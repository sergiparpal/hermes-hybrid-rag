"""Walks user-supplied directories, diffs against the catalog, extracts parents
and chunks for new/changed files, then rebuilds embeddings.npz and bm25.pkl
from the canonical SQLite chunk ordering.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

from .chunking import recursive_split
from .config import (
    CHUNK_OVERLAP,
    MAX_CHUNK,
    bm25_path,
    npz_path,
)
from .parents import (
    Parent,
    _enforce_parent_cap,
    extract_md,
    extract_pdf,
    extract_txt,
)
from .retrieval import _tokenize
from .storage import Store

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


def _hash_file(path: Path, chunk_size: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _walk(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in SUPPORTED_SUFFIXES else []
    out: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
            out.append(p)
    return sorted(out)


def _extract_parents(path: Path) -> list[Parent]:
    suf = path.suffix.lower()
    if suf == ".md":
        return extract_md(path.read_text(encoding="utf-8", errors="replace"))
    if suf == ".txt":
        return extract_txt(path.read_text(encoding="utf-8", errors="replace"))
    if suf == ".pdf":
        return extract_pdf(path)
    return []


def _index_file(store: Store, path: Path) -> tuple[int, int, int]:
    """Insert one file's parents and chunks. Returns (file_id, parent_count,
    chunk_count). Caller is responsible for picking up an existing file row's
    deletion before calling this."""
    st = path.stat()
    raw_parents = _extract_parents(path)
    parents = _enforce_parent_cap(raw_parents)

    file_row = (str(path), st.st_mtime, st.st_size, _hash_file(path),
                path.suffix.lower().lstrip("."), time.time())
    file_id = store.bulk_insert_files([file_row])[str(path)]

    if not parents:
        return file_id, 0, 0

    parent_rows = [(file_id, i, p.kind, p.title, p.page_no, p.text, len(p.text))
                   for i, p in enumerate(parents)]
    parent_ids = store.bulk_insert_parents(parent_rows)

    chunk_count = 0
    for pid, parent in zip(parent_ids, parents):
        pieces = recursive_split(parent.text, max_size=MAX_CHUNK,
                                 overlap=CHUNK_OVERLAP) or [parent.text]
        chunk_rows = [(pid, ord_, piece, 0) for ord_, piece in enumerate(pieces)]
        store.bulk_insert_chunks(chunk_rows)
        chunk_count += len(chunk_rows)

    return file_id, len(parents), chunk_count


def index_path(path, force: bool = False, store: Store | None = None,
               embedder=None) -> dict:
    """Index `path`. Returns {files, parents, chunks, skipped, deleted, ...}.

    - `force=True` reindexes every file (deletes existing rows first).
    - `store` and `embedder` injectable for tests; production passes None and
      the caller resolves singletons.
    """
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"index path does not exist: {root}")

    own_store = store or Store()
    if embedder is None:
        from .embeddings import Embedder as _Emb
        embedder = _Emb()

    files = _walk(root)
    disk_map = {p: p.stat() for p in files}
    diff = own_store.manifest_diff(disk_map)

    if force:
        # treat everything found on disk as changed
        changed_now = [(p, _existing_id_for(own_store, p)) for p in files]
        # filter Nones: if not in db yet, route through "new"
        new_now = [p for p, fid in changed_now if fid is None] + diff["new"]
        changed_now = [(p, fid) for p, fid in changed_now if fid is not None]
        deleted_ids = diff["deleted"]
        unchanged_now: list[Path] = []
    else:
        changed_now = diff["changed"]
        new_now = diff["new"]
        deleted_ids = diff["deleted"]
        unchanged_now = diff["unchanged"]

    # deletes first (cascades to parents/chunks)
    obsolete = list(deleted_ids) + [fid for _, fid in changed_now]
    own_store.delete_files(obsolete)

    # then inserts: changed (now treated as new) + new
    to_insert = [p for _, p in [(fid, p) for p, fid in changed_now]] + new_now
    # dedup paths but preserve order
    seen = set()
    ordered_inserts: list[Path] = []
    for p in to_insert:
        if p not in seen:
            seen.add(p)
            ordered_inserts.append(p)

    new_files = 0
    new_parents = 0
    new_chunks = 0
    for p in ordered_inserts:
        try:
            _, pn, cn = _index_file(own_store, p)
        except Exception as e:
            # Skip the file but keep going. Surface the error in the summary.
            print(f"[advanced-rag] failed to index {p}: {e}")
            continue
        new_files += 1
        new_parents += pn
        new_chunks += cn

    rebuild_artifacts(own_store, embedder)

    # If an engine singleton has been created, drop its cached state so the
    # next query reloads from the freshly written .npz / .pkl.
    try:
        from .engine import get_engine  # local import to avoid cycles at module load
        get_engine().reset()
    except Exception:
        pass

    return {
        "indexed_root": str(root),
        "files_added_or_updated": new_files,
        "files_unchanged": len(unchanged_now),
        "files_deleted": len(diff["deleted"]),
        "parents": new_parents,
        "chunks": new_chunks,
        "totals": own_store.stats(),
    }


def _existing_id_for(store: Store, path: Path) -> int | None:
    conn = store.connect()
    r = conn.execute("SELECT id FROM files WHERE path = ?", (str(path),)).fetchone()
    return r["id"] if r else None


def rebuild_artifacts(store: Store, embedder) -> None:
    """Rebuild embeddings.npz and bm25.pkl from the canonical SQLite chunk
    order. Also rewrites each chunk's `embed_row` so row N of the embeddings
    array maps to chunk_ids[N]."""
    from rank_bm25 import BM25Okapi

    rows = list(store.iter_chunks_ordered())
    if not rows:
        # wipe artifacts so an empty index doesn't load stale data
        for p in (npz_path(store.data_dir), bm25_path(store.data_dir)):
            if p.exists():
                p.unlink()
        return

    texts = [r.text for r in rows]
    chunk_ids = [r.id for r in rows]

    embeddings = embedder.encode(texts)
    if embeddings.shape[0] != len(texts):
        raise RuntimeError("embedder returned wrong number of vectors")

    store.save_embeddings(npz_path(store.data_dir), embeddings, chunk_ids)

    tokenized = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    store.save_bm25(bm25_path(store.data_dir), bm25)

    # canonical row-index ↔ chunk_id mapping into SQLite
    store.bulk_update_embed_rows([(cid, row) for row, cid in enumerate(chunk_ids)])
