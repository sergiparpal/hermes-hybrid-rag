"""Walks user-supplied directories, diffs against the catalog, extracts parents
and chunks for new/changed files, then rebuilds embeddings.npz and bm25.pkl
from the canonical SQLite chunk ordering.
"""
from __future__ import annotations

import hashlib
import logging
import os
import stat as _stat
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

log = logging.getLogger(__name__)

from . import contextual
from .chunking import recursive_split
from .config import (
    CHUNK_OVERLAP,
    CONTEXTUAL_CONCURRENCY,
    MAX_CHUNK,
    MAX_INDEX_FILE_BYTES,
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


def _accept_file(p: Path, *, allow_symlink: bool = False) -> bool:
    """Decide whether to index `p`. Skips symlinks by default — they're a
    confused-deputy vector during recursive walks: a hostile or careless
    symlink in an indexed tree (e.g. `~/Documents/notes.md` pointing at
    `~/.ssh/config`) would otherwise land the target's content in the
    catalog, retrievable by every later query. Also caps file size so a
    single huge file can't OOM the process.

    `allow_symlink=True` is for the explicit-single-file path: when the user
    runs `hermes rag index some-symlink.md`, they made that choice
    themselves; we honor it.
    """
    try:
        if not allow_symlink and p.is_symlink():
            return False
        st = p.stat()
    except OSError:
        return False
    if not _stat.S_ISREG(st.st_mode):
        return False
    if p.suffix.lower() not in SUPPORTED_SUFFIXES:
        return False
    if st.st_size > MAX_INDEX_FILE_BYTES:
        log.warning(
            "skipping %s: %d bytes exceeds MAX_INDEX_FILE_BYTES (%d)",
            p, st.st_size, MAX_INDEX_FILE_BYTES,
        )
        print(
            f"[advanced-rag] skipping {p}: size exceeds "
            f"MAX_INDEX_FILE_BYTES ({MAX_INDEX_FILE_BYTES})",
            file=sys.stderr,
        )
        return False
    return True


def _walk(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if _accept_file(root, allow_symlink=True) else []
    out: list[Path] = []
    # followlinks=False also prevents descending into symlinked dirs, closing
    # the symlinked-directory attack that a per-file is_symlink check misses.
    for dirpath, _dirs, files in os.walk(str(root), followlinks=False):
        for name in files:
            p = Path(dirpath) / name
            if _accept_file(p):
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

    use_contextual = contextual.is_contextual_enabled()
    chunk_count = 0
    for pid, parent in zip(parent_ids, parents):
        pieces = recursive_split(parent.text, max_size=MAX_CHUNK,
                                 overlap=CHUNK_OVERLAP) or [parent.text]
        # Drop any whitespace-only piece so we don't insert empty chunk rows
        # — extract_* should already filter these, but a degenerate parent
        # (e.g. a paragraph of only whitespace) can still slip through here.
        pieces = [p for p in pieces if p and p.strip()]
        if not pieces:
            continue

        prefixes: list[str | None] = [None] * len(pieces)
        if use_contextual:
            # Same parent across calls → Anthropic prompt cache amortizes the
            # parent block. Chunks of one parent are issued concurrently so a
            # large corpus doesn't pay N × per-call latency end-to-end; the
            # cache turns the second+ in-flight call into a cheap cache hit.
            with ThreadPoolExecutor(max_workers=CONTEXTUAL_CONCURRENCY) as ex:
                futures = [
                    ex.submit(contextual.generate_contextual_prefix,
                              parent.text, piece)
                    for piece in pieces
                ]
                for i, fut in enumerate(futures):
                    # `generate_contextual_prefix` returns None on any failure
                    # — never raises — so this never crashes the index run.
                    prefixes[i] = fut.result()

        chunk_rows: list[tuple] = []
        for ord_, (piece, prefix) in enumerate(zip(pieces, prefixes)):
            if prefix:
                composed = prefix + "\n\n" + piece
                chunk_rows.append(
                    (pid, ord_, piece, 0, prefix, composed, composed)
                )
            else:
                # Contextual off or prefix generation failed for this chunk —
                # behavior matches v0.1 (4-tuple insert; new columns stay
                # NULL; retrieval falls back to raw `text`).
                chunk_rows.append((pid, ord_, piece, 0))
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

    # `store_owned` tracks whether we constructed the Store ourselves. When
    # the caller passed one (tests, alternate data dirs), we must NOT reset
    # the process-wide engine at the end — that singleton may be bound to a
    # different data_dir and resetting it would cost the next ambient call
    # a cold reload for unrelated reasons.
    store_owned = store is None
    own_store = store or Store()
    if embedder is None:
        from .embeddings import Embedder as _Emb
        embedder = _Emb()

    files = _walk(root)
    disk_map = {p: p.stat() for p in files}
    # Pass `_hash_file` as the (mtime, size) tiebreaker so in-place edits that
    # preserved both fields still get reindexed. Cost: one SHA-256 per file
    # whose (mtime, size) match — files with stale stats short-circuit.
    diff = own_store.manifest_diff(disk_map, hash_fn=_hash_file)

    if force:
        # In force mode, every file on disk is treated as changed (or new if
        # absent from the catalog). One SQL fetch covers both buckets — no
        # per-file `_existing_id_for` round-trip.
        existing_ids = {Path(r["path"]): r["id"]
                        for r in own_store.connect().execute("SELECT id, path FROM files")}
        changed_now = [(p, existing_ids[p]) for p in files if p in existing_ids]
        new_now = [p for p in files if p not in existing_ids]
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

    # changed paths get reinserted; new paths fall straight through
    ordered_inserts = [p for p, _ in changed_now] + new_now

    new_files = 0
    new_parents = 0
    new_chunks = 0
    for p in ordered_inserts:
        try:
            _, pn, cn = _index_file(own_store, p)
        except Exception as e:
            # Skip the file but keep going. Warning goes to stderr so the CLI's
            # JSON summary on stdout stays parseable.
            log.warning("failed to index %s: %s", p, e)
            print(f"[advanced-rag] failed to index {p}: {e}", file=sys.stderr)
            continue
        new_files += 1
        new_parents += pn
        new_chunks += cn

    rebuild_artifacts(own_store, embedder)

    # If an engine singleton has been created, drop its cached state so the
    # next query reloads from the freshly written .npz / .pkl. Only reset
    # when we own the store: caller-supplied stores belong to a different
    # data_dir (tests, isolated runs) and the singleton would be unrelated.
    if store_owned:
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


def rebuild_artifacts(store: Store, embedder) -> None:
    """Rebuild embeddings.npz from the canonical SQLite chunk order. Also
    rewrites each chunk's `embed_row` so row N of the embeddings array maps
    to chunk_ids[N]. BM25 is no longer persisted — see storage.py."""
    rows = list(store.iter_chunks_ordered())

    # Drop any legacy bm25.pkl regardless of whether the new index is empty.
    # Leaving it on disk would be a stale, unreferenced pickle file — and a
    # cohabiting attacker's planted file (the exact threat we removed pickle
    # to address) would otherwise survive a re-index unnoticed.
    legacy_bm25 = bm25_path(store.data_dir)
    if legacy_bm25.exists():
        try:
            legacy_bm25.unlink()
        except OSError:
            pass

    if not rows:
        npz_p = npz_path(store.data_dir)
        if npz_p.exists():
            npz_p.unlink()
        return

    embed_texts = [r.effective_embedding_text for r in rows]
    chunk_ids = [r.id for r in rows]

    embeddings = embedder.encode(embed_texts)
    if embeddings.shape[0] != len(embed_texts):
        raise RuntimeError("embedder returned wrong number of vectors")

    store.save_embeddings(npz_path(store.data_dir), embeddings)

    store.bulk_update_embed_rows([(cid, row) for row, cid in enumerate(chunk_ids)])

    model_name = getattr(embedder, "model_name", None) or getattr(
        embedder, "_model_name", "unknown"
    )
    store.set_meta("embed_model", str(model_name))
    store.set_meta("embed_dim", str(int(embeddings.shape[1])))
