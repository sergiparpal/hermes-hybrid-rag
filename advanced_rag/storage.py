"""SQLite-backed catalog for files, parents, chunks + atomic writes for the
embeddings .npz. The single source of truth for chunk ordering: SELECT chunks
ordered by (parent_id, ord) — the row index in that ordering equals the row
index in the embeddings array (the `embed_row`).

BM25 is intentionally NOT persisted to disk: it's rebuilt from `chunks` on
every engine load. The previous pickle-on-disk path was a code-execution
sink (CWE-502) for anyone who could write the data dir. Tokenization is
cheap; pickle is forever.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from .config import bm25_path, db_path, get_data_dir, npz_path


@dataclass
class ChunkRow:
    id: int
    parent_id: int
    ord: int
    text: str
    embed_row: int
    contextual_prefix: str | None = None
    text_for_embedding: str | None = None
    text_for_bm25: str | None = None

    @property
    def effective_embedding_text(self) -> str:
        """Phase 2: `prefix + chunk` when contextual retrieval is active,
        else the raw chunk. Single source of truth so embeddings rebuild and
        any future inspectors agree."""
        return self.text_for_embedding or self.text

    @property
    def effective_bm25_text(self) -> str:
        return self.text_for_bm25 or self.text


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

    @property
    def db_path(self) -> Path:
        return db_path(self.data_dir)

    @property
    def npz_path(self) -> Path:
        return npz_path(self.data_dir)

    @property
    def bm25_path(self) -> Path:
        return bm25_path(self.data_dir)

    def connect(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        self.init_schema(conn)
        self._conn = conn
        return conn

    def close(self) -> None:
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

    def init_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(SCHEMA_DDL)
        self._migrate_schema(conn)
        conn.commit()

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        """Lazy migrations for existing on-disk DBs. Currently:

        - Phase 2 (contextual retrieval): add `contextual_prefix`,
          `text_for_embedding`, `text_for_bm25` columns to `chunks` if
          missing. New DBs already have them via the DDL above.
        """
        cols = {r[1] for r in conn.execute("PRAGMA table_info(chunks)").fetchall()}
        for new_col in ("contextual_prefix", "text_for_embedding", "text_for_bm25"):
            if new_col not in cols:
                conn.execute(f"ALTER TABLE chunks ADD COLUMN {new_col} TEXT")

    # --- manifest diff ---

    def manifest_diff(
        self,
        disk_files: dict[Path, os.stat_result],
        hash_fn: Callable[[Path], str] | None = None,
    ) -> dict:
        """Returns {unchanged, changed, new, deleted}, each a list/dict by path.

        - unchanged: same mtime AND size (and same content_hash, when
          ``hash_fn`` is supplied) as the row in `files`.
        - changed: row exists but mtime, size, or (when checked) content_hash
          differ.
        - new: path not in `files` table.
        - deleted: row exists but path not in `disk_files`.

        ``hash_fn`` is only invoked on the (mtime, size)-match branch, so
        unchanged files dominate the cost: each pays exactly one hash. Files
        with stale (mtime, size) shortcut to "changed" without re-hashing —
        the hash will be recomputed when the file is reindexed anyway.
        """
        conn = self.connect()
        rows = {Path(r["path"]): {"id": r["id"], "mtime": r["mtime"],
                                  "size": r["size"], "content_hash": r["content_hash"]}
                for r in conn.execute("SELECT id, path, mtime, size, content_hash FROM files")}

        unchanged: list[Path] = []
        changed: list[tuple[Path, int]] = []  # (path, file_id)
        new: list[Path] = []
        deleted: list[int] = []

        for path, st in disk_files.items():
            row = rows.get(path)
            if row is None:
                new.append(path)
            elif row["mtime"] == st.st_mtime and row["size"] == st.st_size:
                if hash_fn is None:
                    unchanged.append(path)
                else:
                    disk_hash = hash_fn(path)
                    if disk_hash == row["content_hash"]:
                        unchanged.append(path)
                    else:
                        # in-place edit that preserved (mtime, size) — rare but
                        # real (e.g. `os.utime` after a same-size rewrite).
                        changed.append((path, row["id"]))
            else:
                changed.append((path, row["id"]))

        for path, row in rows.items():
            if path not in disk_files:
                deleted.append(row["id"])

        return {"unchanged": unchanged, "changed": changed,
                "new": new, "deleted": deleted}

    def delete_files(self, file_ids: list[int]) -> None:
        if not file_ids:
            return
        with self.transaction() as conn:
            qmarks = ",".join("?" * len(file_ids))
            conn.execute(f"DELETE FROM files WHERE id IN ({qmarks})", file_ids)

    # --- bulk inserts ---

    def bulk_insert_files(self, rows: list[tuple]) -> dict[str, int]:
        """rows: list of (path, mtime, size, content_hash, filetype, indexed_at).
        Returns {path: file_id}.
        """
        out: dict[str, int] = {}
        if not rows:
            return out
        with self.transaction() as conn:
            for r in rows:
                cur = conn.execute(
                    "INSERT INTO files(path, mtime, size, content_hash, filetype, indexed_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)", r,
                )
                out[r[0]] = cur.lastrowid
        return out

    def bulk_insert_parents(self, rows: list[tuple]) -> list[int]:
        """rows: list of (file_id, ord, kind, title, page_no, text, char_len).
        Returns list of parent_ids in input order.
        """
        ids: list[int] = []
        if not rows:
            return ids
        with self.transaction() as conn:
            for r in rows:
                cur = conn.execute(
                    "INSERT INTO parents(file_id, ord, kind, title, page_no, text, char_len) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)", r,
                )
                ids.append(cur.lastrowid)
        return ids

    def bulk_insert_chunks(self, rows: list[tuple]) -> list[int]:
        """Insert chunks. Each row is either:

        - 4-tuple `(parent_id, ord, text, embed_row)` — legacy / contextual
          retrieval off; new columns stay NULL.
        - 7-tuple `(parent_id, ord, text, embed_row, contextual_prefix,
          text_for_embedding, text_for_bm25)` — Phase 2 contextual rows.

        Rows of mixed length in the same call are fine. Returns chunk_ids."""
        ids: list[int] = []
        if not rows:
            return ids
        with self.transaction() as conn:
            for r in rows:
                if len(r) == 4:
                    cur = conn.execute(
                        "INSERT INTO chunks(parent_id, ord, text, embed_row) "
                        "VALUES (?, ?, ?, ?)",
                        r,
                    )
                elif len(r) == 7:
                    cur = conn.execute(
                        "INSERT INTO chunks(parent_id, ord, text, embed_row, "
                        "contextual_prefix, text_for_embedding, text_for_bm25) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        r,
                    )
                else:
                    raise ValueError(
                        f"bulk_insert_chunks: expected 4- or 7-tuple, got "
                        f"len {len(r)}"
                    )
                ids.append(cur.lastrowid)
        return ids

    # --- canonical chunk ordering for embedding rebuild ---

    def iter_chunks_ordered(self) -> Iterator[ChunkRow]:
        conn = self.connect()
        # No JOIN with parents needed — every chunk has parent_id NOT NULL by
        # schema and we don't read any parents columns. The canonical order
        # (parent_id, ord) is purely a `chunks` ordering.
        for r in conn.execute(
            "SELECT id, parent_id, ord, text, embed_row, "
            "       contextual_prefix, text_for_embedding, text_for_bm25 "
            "FROM chunks ORDER BY parent_id, ord"
        ):
            yield ChunkRow(
                id=r["id"], parent_id=r["parent_id"], ord=r["ord"],
                text=r["text"], embed_row=r["embed_row"],
                contextual_prefix=r["contextual_prefix"],
                text_for_embedding=r["text_for_embedding"],
                text_for_bm25=r["text_for_bm25"],
            )

    def get_chunk_ids_ordered(self) -> list[int]:
        """Just the chunk ids in canonical (parent_id, ord) order. Engine
        uses this on every load to rebuild `_chunk_ids`; pulling only `id`
        avoids materializing the full row × text × prefix payload.
        """
        conn = self.connect()
        return [r["id"] for r in conn.execute(
            "SELECT id FROM chunks ORDER BY parent_id, ord"
        )]

    def bulk_update_embed_rows(self, pairs: list[tuple[int, int]]) -> None:
        """pairs: list of (chunk_id, embed_row)."""
        if not pairs:
            return
        with self.transaction() as conn:
            conn.executemany("UPDATE chunks SET embed_row = ? WHERE id = ?",
                             [(row, cid) for cid, row in pairs])

    # --- read helpers ---

    def get_chunk(self, chunk_id: int) -> dict | None:
        # Returns the v0.1 columns only. Phase 2 columns (contextual_prefix,
        # text_for_embedding, text_for_bm25) are intentionally omitted: this
        # accessor is for inspectors and tooling that want the raw chunk text,
        # not the retrieval-time augmented form. Use `iter_chunks_ordered`
        # when you need the full row.
        conn = self.connect()
        r = conn.execute(
            "SELECT c.id, c.parent_id, c.ord, c.text, c.embed_row "
            "FROM chunks c WHERE c.id = ?", (chunk_id,),
        ).fetchone()
        return dict(r) if r else None

    def get_parent(self, parent_id: int) -> dict | None:
        conn = self.connect()
        r = conn.execute(
            "SELECT p.id, p.file_id, p.ord, p.kind, p.title, p.page_no, p.text, p.char_len, "
            "       f.path AS source_path, f.filetype "
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
        ids = list(chunk_ids)
        if not ids:
            return {}
        conn = self.connect()
        out: dict[int, int] = {}
        # SQLite has a fixed parameter ceiling (default 999); chunk in case the
        # caller hands us a very large list.
        for start in range(0, len(ids), 500):
            batch = ids[start:start + 500]
            qmarks = ",".join("?" * len(batch))
            rows = conn.execute(
                f"SELECT id, parent_id FROM chunks WHERE id IN ({qmarks})", batch,
            ).fetchall()
            for r in rows:
                out[r["id"]] = r["parent_id"]
        return out

    def get_parents(self, parent_ids: Iterator[int] | list[int]) -> dict[int, dict]:
        """Batched parent fetch. Returns {parent_id: row dict}, skipping
        missing ids. Joins `files` so the source_path/filetype are populated."""
        ids = list(parent_ids)
        if not ids:
            return {}
        conn = self.connect()
        out: dict[int, dict] = {}
        for start in range(0, len(ids), 500):
            batch = ids[start:start + 500]
            qmarks = ",".join("?" * len(batch))
            rows = conn.execute(
                "SELECT p.id, p.file_id, p.ord, p.kind, p.title, p.page_no, "
                "       p.text, p.char_len, "
                "       f.path AS source_path, f.filetype "
                f"FROM parents p JOIN files f ON f.id = p.file_id "
                f"WHERE p.id IN ({qmarks})",
                batch,
            ).fetchall()
            for r in rows:
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

    # --- atomic embeddings + bm25 IO ---

    def save_embeddings(self, target_path: Path, embeddings: np.ndarray,
                        chunk_ids: list[int] | None = None) -> None:
        """Write the embeddings array atomically. `chunk_ids` is accepted for
        backwards compatibility but no longer persisted — the canonical
        row-index ↔ chunk-id mapping lives in SQLite (`chunks.embed_row`),
        and the engine reconstructs the list from `iter_chunks_ordered()`.
        """
        del chunk_ids  # dropped from .npz on purpose; kept in the signature
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        # Pass a file handle so numpy doesn't auto-append `.npz` and break our
        # atomic-rename scheme.
        try:
            with open(tmp, "wb") as fh:
                np.savez(fh, embeddings=embeddings)
            os.replace(tmp, target)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    def load_embeddings(self, target_path: Path) -> np.ndarray:
        """Return the embeddings array. Old `.npz` files that still carry a
        `chunk_ids` array load fine — we just ignore that key."""
        with np.load(target_path) as data:
            return data["embeddings"]

    def iter_bm25_texts_ordered(self) -> Iterator[str]:
        """BM25 text per chunk in canonical (parent_id, ord) order.

        Prefers the contextual-composed text (Phase 2) when present, falls
        back to raw chunk text. Used by the engine to rebuild BM25 from
        SQLite at load time — see module docstring for why pickle is gone.
        """
        conn = self.connect()
        for r in conn.execute(
            "SELECT text, text_for_bm25 FROM chunks ORDER BY parent_id, ord"
        ):
            yield r["text_for_bm25"] or r["text"]
