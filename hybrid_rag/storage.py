"""SQLite-backed catalog for files, parents, chunks, and meta key/value.

The single source of truth for chunk ordering: ``SELECT chunks ORDER BY
(parent_id, ord)`` — the row index in that ordering equals the row index in
the embeddings array (the ``embed_row``).

BM25 is intentionally NOT persisted to disk: it's rebuilt from ``chunks`` on
every engine load. The previous pickle-on-disk path was a code-execution
sink (CWE-502) for anyone who could write the data dir. Tokenization is
cheap; pickle is forever.

Embedding ``.npz`` I/O lives in ``artifacts.py``; filesystem reconciliation
lives in ``manifest.py``. ``Store`` here is the catalog only — see those
modules for the split rationale.
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Sequence

from .config import db_path, get_data_dir
from .models import ChunkRow


# SQLite has a fixed parameter ceiling (999 on older builds). 500 keeps us
# safely below it for every IN-clause batch we issue.
_SQLITE_IN_BATCH = 500

# The canonical chunk ordering. The row index in this ordering IS the
# `embed_row` — every component that walks chunks for indexing or retrieval
# must use this exact clause.
_CANONICAL_CHUNK_ORDER = "ORDER BY parent_id, ord"

# Column list shared by `get_parent` and `get_parents` — both join `files`
# to surface the source path and filetype on every parent row. ``{text}`` is
# a format placeholder so callers can swap a SUBSTR projection in without
# string-replacing on column names (which silently breaks the moment a
# future column happens to share a substring with "p.text").
_PARENT_WITH_FILE_COLS_TEMPLATE = (
    "p.id, p.file_id, p.ord, p.kind, p.title, p.page_no, {text} AS text, "
    "p.char_len, f.path AS source_path, f.filetype"
)
_PARENT_WITH_FILE_COLS = _PARENT_WITH_FILE_COLS_TEMPLATE.format(text="p.text")

SCHEMA_DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS files (
  id           INTEGER PRIMARY KEY,
  path         TEXT    NOT NULL UNIQUE,
  mtime        REAL    NOT NULL,
  size         INTEGER NOT NULL,
  content_hash TEXT    NOT NULL,
  filetype     TEXT    NOT NULL,
  indexed_at   REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_files_path ON files(path);

CREATE TABLE IF NOT EXISTS parents (
  id        INTEGER PRIMARY KEY,
  file_id   INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  ord       INTEGER NOT NULL,
  kind      TEXT    NOT NULL,
  title     TEXT,
  page_no   INTEGER,
  text      TEXT    NOT NULL,
  char_len  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_parents_file ON parents(file_id);

CREATE TABLE IF NOT EXISTS chunks (
  id                 INTEGER PRIMARY KEY,
  parent_id          INTEGER NOT NULL REFERENCES parents(id) ON DELETE CASCADE,
  ord                INTEGER NOT NULL,
  text               TEXT    NOT NULL,
  embed_row          INTEGER NOT NULL,
  contextual_prefix  TEXT,
  text_for_embedding TEXT,
  text_for_bm25      TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embed_row ON chunks(embed_row);

CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
"""


class Store:
    def __init__(self, data_dir: Path | None = None):
        # Resolution order: explicit arg > env (via get_data_dir()) > default.
        self.data_dir = Path(data_dir) if data_dir is not None else get_data_dir()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        # Serializes the lazy ``connect()`` so concurrent first-callers (the
        # warm-up thread racing a tool call) don't each open a connection and
        # then orphan one. Only protects construction — once ``_conn`` is set,
        # the fast path skips the lock.
        self._conn_lock = threading.Lock()
        # Incremented on every meta write. Read by engines whose TTL-cached
        # `index_version` should invalidate immediately on same-process
        # reindex (test and dev), without paying the SQL read for that
        # check. Cross-process writers don't share this counter — engines
        # in other processes still rely on the TTL to notice the change.
        self._meta_write_seq = 0

    @property
    def db_path(self) -> Path:
        return db_path(self.data_dir)

    def connect(self) -> sqlite3.Connection:
        # Double-checked locking. Fast path: connection already open, no lock.
        # Slow path: acquire the lock so only one thread opens; others wait
        # then re-check. Without this two racing first-callers would each
        # open a sqlite3 connection and one would be orphaned with the PRAGMAs
        # half-applied.
        if self._conn is not None:
            return self._conn
        with self._conn_lock:
            if self._conn is not None:
                return self._conn
            # `check_same_thread=False`: the engine singleton's connection is
            # opened by whichever thread first calls into the store —
            # typically the `on_session_start` warm thread — and then reused
            # from the main thread on every tool call and pre_llm_call hook.
            # SQLite itself is thread-safe; Python's default thread-affinity
            # check would otherwise raise ProgrammingError on every
            # cross-thread use. Writes in this codebase are confined to the
            # indexing path (a separate CLI process), so we don't need
            # additional locking around statements.
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            # PRAGMAs: WAL lets the indexing CLI write while the Hermes
            # process reads concurrently (default DELETE mode would block
            # readers for seconds). `synchronous=NORMAL` skips the per-commit
            # fsync — the catalog is rebuildable from source documents, so a
            # power-loss rollback is acceptable. mmap + larger page cache
            # make the parent/chunk fetches on the hot path effectively
            # zero-syscall.
            conn.executescript(
                "PRAGMA foreign_keys = ON;\n"
                "PRAGMA journal_mode = WAL;\n"
                "PRAGMA synchronous = NORMAL;\n"
                "PRAGMA temp_store = MEMORY;\n"
                "PRAGMA mmap_size = 268435456;\n"   # 256 MiB
                "PRAGMA cache_size = -65536;\n"     # 64 MiB (negative = KiB)
            )
            self.init_schema(conn)
            # Publish last so a partially-initialized connection (PRAGMAs
            # half-applied) is never visible to other threads.
            self._conn = conn
            return self._conn

    def close(self) -> None:
        with self._conn_lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    @contextmanager
    def transaction(self):
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    @contextmanager
    def savepoint(self, name: str = "sp"):
        """Per-unit-of-work atomicity inside an outer transaction.

        SQLite's BEGIN/COMMIT/ROLLBACK doesn't nest; SAVEPOINT does. The
        indexing loop opens one outer transaction over the whole batch
        (collapsing N × 3 commits to one commit) but still needs each
        file's inserts to be independently atomic — if one file's
        extraction blows up midway, only its rows should disappear.

        The name is a literal SQL identifier — keep it ASCII (the caller
        is internal so we don't sanitize).
        """
        conn = self.connect()
        conn.execute(f"SAVEPOINT {name}")
        try:
            yield conn
        except Exception:
            conn.execute(f"ROLLBACK TO SAVEPOINT {name}")
            conn.execute(f"RELEASE SAVEPOINT {name}")
            raise
        else:
            conn.execute(f"RELEASE SAVEPOINT {name}")

    def init_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(SCHEMA_DDL)
        self._migrate_schema(conn)
        conn.commit()

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        """Lazy migrations for existing on-disk DBs.

        Adds the contextual-retrieval columns (`contextual_prefix`,
        `text_for_embedding`, `text_for_bm25`) to `chunks` if they're missing.
        New DBs already have them via the DDL above.
        """
        cols = {r[1] for r in conn.execute("PRAGMA table_info(chunks)").fetchall()}
        for new_col in ("contextual_prefix", "text_for_embedding", "text_for_bm25"):
            if new_col not in cols:
                conn.execute(f"ALTER TABLE chunks ADD COLUMN {new_col} TEXT")

    # --- batched IN-clause helper ---

    def _select_in_batches(
        self,
        ids: Sequence[int],
        select_sql_template: str,
        row_to_kv: Callable[[sqlite3.Row], tuple],
    ) -> dict:
        """Run an ``IN (?, ?, …)`` query batched at ``_SQLITE_IN_BATCH`` and
        merge results into a single dict keyed by ``row_to_kv``.

        ``select_sql_template`` must contain a ``{qmarks}`` placeholder where
        the comma-joined ``?`` list goes — letting callers express any column
        list / join shape while sharing the batching machinery.
        """
        if not ids:
            return {}
        conn = self.connect()
        out: dict = {}
        for start in range(0, len(ids), _SQLITE_IN_BATCH):
            batch = list(ids[start:start + _SQLITE_IN_BATCH])
            qmarks = ",".join("?" * len(batch))
            for r in conn.execute(select_sql_template.format(qmarks=qmarks), batch):
                k, v = row_to_kv(r)
                out[k] = v
        return out

    @contextmanager
    def _maybe_transaction(self, conn):
        """If a connection is passed in (caller owns an open transaction),
        yield it directly. Otherwise open and commit our own. Lets the bulk
        writers be both standalone and composable inside a larger txn."""
        if conn is not None:
            yield conn
            return
        with self.transaction() as own:
            yield own

    def delete_files(self, file_ids: list[int], *, conn=None) -> None:
        if not file_ids:
            return
        with self._maybe_transaction(conn) as c:
            for start in range(0, len(file_ids), _SQLITE_IN_BATCH):
                batch = file_ids[start:start + _SQLITE_IN_BATCH]
                qmarks = ",".join("?" * len(batch))
                c.execute(f"DELETE FROM files WHERE id IN ({qmarks})", batch)

    # --- bulk inserts ---

    def bulk_insert_files(self, rows: list[tuple], *, conn=None) -> dict[str, int]:
        """rows: list of (path, mtime, size, content_hash, filetype, indexed_at).
        Returns {path: file_id}.

        ``conn`` lets a higher-level caller (the indexing loop) collapse
        many bulk-insert calls into a single transaction — see
        ``indexing._apply_inserts``. When ``None``, this opens and commits
        its own transaction.
        """
        out: dict[str, int] = {}
        if not rows:
            return out
        with self._maybe_transaction(conn) as c:
            for r in rows:
                cur = c.execute(
                    "INSERT INTO files(path, mtime, size, content_hash, filetype, indexed_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)", r,
                )
                out[r[0]] = cur.lastrowid
        return out

    def bulk_insert_parents(self, rows: list[tuple], *, conn=None) -> list[int]:
        """rows: list of (file_id, ord, kind, title, page_no, text, char_len).
        Returns list of parent_ids in input order. See `bulk_insert_files`
        for the ``conn`` parameter contract.
        """
        ids: list[int] = []
        if not rows:
            return ids
        with self._maybe_transaction(conn) as c:
            for r in rows:
                cur = c.execute(
                    "INSERT INTO parents(file_id, ord, kind, title, page_no, text, char_len) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)", r,
                )
                ids.append(cur.lastrowid)
        return ids

    def bulk_insert_chunks(self, rows: list[tuple], *, conn=None) -> list[int]:
        """Insert chunks. Each row is a 7-tuple:

            (parent_id, ord, text, embed_row,
             contextual_prefix, text_for_embedding, text_for_bm25)

        The last three columns are NULLable — pass ``None`` for them when
        contextual retrieval is off. Returns chunk_ids in input order. See
        `bulk_insert_files` for the ``conn`` parameter contract.
        """
        ids: list[int] = []
        if not rows:
            return ids
        with self._maybe_transaction(conn) as c:
            for r in rows:
                if len(r) != 7:
                    raise RuntimeError(
                        f"bulk_insert_chunks: expected 7-tuple, got len {len(r)}"
                    )
                cur = c.execute(
                    "INSERT INTO chunks(parent_id, ord, text, embed_row, "
                    "contextual_prefix, text_for_embedding, text_for_bm25) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    r,
                )
                ids.append(cur.lastrowid)
        return ids

    # --- canonical chunk ordering for embedding rebuild ---

    def iter_chunks_ordered(self) -> Iterator[ChunkRow]:
        conn = self.connect()
        # No JOIN with parents needed — every chunk has parent_id NOT NULL by
        # schema and we don't read any parents columns.
        for r in conn.execute(
            "SELECT id, parent_id, ord, text, embed_row, "
            "       contextual_prefix, text_for_embedding, text_for_bm25 "
            f"FROM chunks {_CANONICAL_CHUNK_ORDER}"
        ):
            yield ChunkRow(
                id=r["id"], parent_id=r["parent_id"], ord=r["ord"],
                text=r["text"], embed_row=r["embed_row"],
                contextual_prefix=r["contextual_prefix"],
                text_for_embedding=r["text_for_embedding"],
                text_for_bm25=r["text_for_bm25"],
            )

    def get_chunk_ids_ordered(self) -> list[int]:
        """Just the chunk ids in canonical order. Engine uses this on every
        load to rebuild `_chunk_ids`; pulling only `id` avoids materializing
        the full row × text × prefix payload."""
        conn = self.connect()
        return [r["id"] for r in conn.execute(
            f"SELECT id FROM chunks {_CANONICAL_CHUNK_ORDER}"
        )]

    def bulk_update_embed_rows(self, pairs: list[tuple[int, int]], *, conn=None) -> None:
        """pairs: list of (chunk_id, embed_row). See `bulk_insert_files`
        for the ``conn`` parameter contract."""
        if not pairs:
            return
        with self._maybe_transaction(conn) as c:
            c.executemany("UPDATE chunks SET embed_row = ? WHERE id = ?",
                          [(row, cid) for cid, row in pairs])

    # --- read helpers ---

    def get_chunk(self, chunk_id: int) -> dict | None:
        # Returns the v0.1 columns only. The contextual-retrieval columns
        # (contextual_prefix, text_for_embedding, text_for_bm25) are
        # intentionally omitted: this accessor is for inspectors and tooling
        # that want the raw chunk text, not the retrieval-time augmented form.
        # Use `iter_chunks_ordered` when you need the full row.
        conn = self.connect()
        r = conn.execute(
            "SELECT c.id, c.parent_id, c.ord, c.text, c.embed_row "
            "FROM chunks c WHERE c.id = ?", (chunk_id,),
        ).fetchone()
        return dict(r) if r else None

    def get_parent(self, parent_id: int) -> dict | None:
        conn = self.connect()
        r = conn.execute(
            f"SELECT {_PARENT_WITH_FILE_COLS} "
            "FROM parents p JOIN files f ON f.id = p.file_id WHERE p.id = ?",
            (parent_id,),
        ).fetchone()
        return dict(r) if r else None

    def chunks_for_parent(self, parent_id: int) -> list[dict]:
        # Same intentional omission as `get_chunk`: surfaces raw chunk text
        # (what's actually in the document), not the contextual-augmented
        # form. `rag_drill_down` builds on this to show users the underlying
        # passage; the retrieval-time prefix would be confusing here.
        conn = self.connect()
        rows = conn.execute(
            "SELECT id, parent_id, ord, text, embed_row FROM chunks "
            "WHERE parent_id = ? ORDER BY ord", (parent_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def parent_id_for_chunk(self, chunk_id: int) -> int | None:
        conn = self.connect()
        r = conn.execute("SELECT parent_id FROM chunks WHERE id = ?",
                         (chunk_id,)).fetchone()
        return r["parent_id"] if r else None

    def parent_ids_for_chunks(self, chunk_ids: Iterator[int] | list[int]) -> dict[int, int]:
        """Batched chunk_id → parent_id lookup. Skips ids that don't exist."""
        return self._select_in_batches(
            list(chunk_ids),
            "SELECT id, parent_id FROM chunks WHERE id IN ({qmarks})",
            lambda r: (r["id"], r["parent_id"]),
        )

    def get_parents(
        self,
        parent_ids: Iterator[int] | list[int],
        *,
        text_cap: int | None = None,
    ) -> dict[int, dict]:
        """Batched parent fetch. Returns {parent_id: row dict}, skipping
        missing ids. Joins `files` so source_path/filetype are populated.

        ``text_cap`` lets the rerank-pool stage fetch truncated text
        instead of the full parent body. Parents that don't survive the
        rerank cut never need full text; the survivors are re-fetched
        with ``text_cap=None`` before being returned to the caller. Saves
        about half the text I/O on every query.
        """
        ids = list(parent_ids)
        if not ids:
            return {}
        conn = self.connect()
        # Switch the `text` projection between SUBSTR (truncated) and a
        # plain column reference. Both shapes still return the column as
        # `text` so callers don't branch on it.
        if text_cap is None:
            text_expr = "p.text"
        else:
            text_expr = "SUBSTR(p.text, 1, ?)"
        cols = _PARENT_WITH_FILE_COLS_TEMPLATE.format(text=text_expr)
        out: dict = {}
        for start in range(0, len(ids), _SQLITE_IN_BATCH):
            batch = ids[start:start + _SQLITE_IN_BATCH]
            qmarks = ",".join("?" * len(batch))
            sql = (f"SELECT {cols} FROM parents p "
                   "JOIN files f ON f.id = p.file_id "
                   f"WHERE p.id IN ({qmarks})")
            params = ([text_cap] + batch) if text_cap is not None else batch
            for r in conn.execute(sql, params):
                out[r["id"]] = dict(r)
        return out

    def list_sources(self) -> list[dict]:
        conn = self.connect()
        rows = conn.execute(
            "SELECT f.path, f.filetype, f.indexed_at, "
            "       COUNT(DISTINCT p.id) AS parent_count, "
            "       COUNT(c.id) AS chunk_count "
            "FROM files f "
            "LEFT JOIN parents p ON p.file_id = f.id "
            "LEFT JOIN chunks c ON c.parent_id = p.id "
            "GROUP BY f.id ORDER BY f.path"
        ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        conn = self.connect()
        return {
            "files": conn.execute("SELECT COUNT(*) AS n FROM files").fetchone()["n"],
            "parents": conn.execute("SELECT COUNT(*) AS n FROM parents").fetchone()["n"],
            "chunks": conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"],
            "data_dir": str(self.data_dir),
        }

    # --- meta key/value ---

    def get_meta(self, key: str) -> str | None:
        conn = self.connect()
        r = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return r["value"] if r else None

    def set_meta(self, key: str, value: str) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, str(value)),
            )
        self._meta_write_seq += 1

    @property
    def meta_write_seq(self) -> int:
        """Monotonic counter incremented on every meta write. Engines use
        this as a same-process invalidation signal for their TTL-cached
        version read — see `RAGEngine._current_version`."""
        return self._meta_write_seq

    def iter_bm25_texts_ordered(self) -> Iterator[str]:
        """BM25 text per chunk in canonical order.

        Three-level fallback (matches :pyattr:`ChunkRow.effective_bm25_text`):
        ``text_for_bm25`` > ``text_for_embedding`` > raw ``text``. The middle
        rung exists because for contextual retrieval the BM25 and embedding
        texts are usually identical and storing one copy is enough — the
        indexer leaves ``text_for_bm25`` NULL in that case.
        """
        conn = self.connect()
        for r in conn.execute(
            "SELECT text, text_for_embedding, text_for_bm25 "
            f"FROM chunks {_CANONICAL_CHUNK_ORDER}"
        ):
            yield r["text_for_bm25"] or r["text_for_embedding"] or r["text"]
