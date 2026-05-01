"""SQLite-backed catalog for files, parents, chunks + atomic writes for the
embeddings .npz and the BM25 pickle. The single source of truth for chunk
ordering: SELECT chunks ordered by (parent_id, ord) — the row index in that
ordering equals the row index in the embeddings array (the `embed_row`).
"""
from __future__ import annotations

import os
import pickle
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import bm25_path, db_path, get_data_dir, npz_path


@dataclass
class ChunkRow:
    id: int
    parent_id: int
    ord: int
    text: str
    embed_row: int


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
  id         INTEGER PRIMARY KEY,
  parent_id  INTEGER NOT NULL REFERENCES parents(id) ON DELETE CASCADE,
  ord        INTEGER NOT NULL,
  text       TEXT    NOT NULL,
  embed_row  INTEGER NOT NULL
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
        conn.commit()

    # --- manifest diff ---

    def manifest_diff(self, disk_files: dict[Path, os.stat_result]) -> dict:
        """Returns {unchanged, changed, new, deleted}, each a list/dict by path.

        - unchanged: same mtime AND size as the row in `files`.
        - changed: row exists but mtime or size differ.
        - new: path not in `files` table.
        - deleted: row exists but path not in `disk_files`.
        """
        conn = self.connect()
        rows = {Path(r["path"]): {"id": r["id"], "mtime": r["mtime"], "size": r["size"]}
                for r in conn.execute("SELECT id, path, mtime, size FROM files")}

        unchanged: list[Path] = []
        changed: list[tuple[Path, int]] = []  # (path, file_id)
        new: list[Path] = []
        deleted: list[int] = []

        for path, st in disk_files.items():
            row = rows.get(path)
            if row is None:
                new.append(path)
            elif row["mtime"] == st.st_mtime and row["size"] == st.st_size:
                unchanged.append(path)
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
        """rows: list of (parent_id, ord, text, embed_row). Returns chunk_ids."""
        ids: list[int] = []
        if not rows:
            return ids
        with self.transaction() as conn:
            for r in rows:
                cur = conn.execute(
                    "INSERT INTO chunks(parent_id, ord, text, embed_row) VALUES (?, ?, ?, ?)",
                    r,
                )
                ids.append(cur.lastrowid)
        return ids

    # --- canonical chunk ordering for embedding rebuild ---

    def iter_chunks_ordered(self) -> Iterator[ChunkRow]:
        conn = self.connect()
        for r in conn.execute(
            "SELECT c.id, c.parent_id, c.ord, c.text, c.embed_row "
            "FROM chunks c JOIN parents p ON p.id = c.parent_id "
            "ORDER BY c.parent_id, c.ord"
        ):
            yield ChunkRow(id=r["id"], parent_id=r["parent_id"], ord=r["ord"],
                           text=r["text"], embed_row=r["embed_row"])

    def bulk_update_embed_rows(self, pairs: list[tuple[int, int]]) -> None:
        """pairs: list of (chunk_id, embed_row)."""
        if not pairs:
            return
        with self.transaction() as conn:
            conn.executemany("UPDATE chunks SET embed_row = ? WHERE id = ?",
                             [(row, cid) for cid, row in pairs])

    # --- read helpers ---

    def get_chunk(self, chunk_id: int) -> dict | None:
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

    # --- atomic embeddings + bm25 IO ---

    def save_embeddings(self, target_path: Path, embeddings: np.ndarray,
                        chunk_ids: list[int]) -> None:
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        # Pass a file handle so numpy doesn't auto-append `.npz` and break our
        # atomic-rename scheme.
        try:
            with open(tmp, "wb") as fh:
                np.savez(fh, embeddings=embeddings,
                         chunk_ids=np.asarray(chunk_ids, dtype=np.int64))
            os.replace(tmp, target)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    def load_embeddings(self, target_path: Path) -> tuple[np.ndarray, list[int]]:
        with np.load(target_path) as data:
            embeddings = data["embeddings"]
            chunk_ids = data["chunk_ids"].tolist()
        return embeddings, chunk_ids

    def save_bm25(self, target_path: Path, bm25_obj) -> None:
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        try:
            with open(tmp, "wb") as f:
                pickle.dump(bm25_obj, f)
            os.replace(tmp, target)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    def load_bm25(self, target_path: Path):
        with open(target_path, "rb") as f:
            return pickle.load(f)
