"""Walks user-supplied directories, diffs against the catalog, extracts
parents and chunks for new/changed files, then rebuilds ``embeddings.npz``
from the canonical SQLite chunk ordering. BM25 is rebuilt at engine load
time from the same canonical order; no pickle on disk.
"""
from __future__ import annotations

import hashlib
import logging
import os
import stat as _stat
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from . import contextual, manifest
from .artifacts import ArtifactStore
from .chunking import recursive_split
from .config import CHUNK_OVERLAP, MAX_CHUNK, MAX_INDEX_FILE_BYTES
from .contextual import CONTEXTUAL_CONCURRENCY
from .extractors import DEFAULT_REGISTRY
from .models import Parent
from .parents import _enforce_parent_cap
from .storage import Store

log = logging.getLogger(__name__)


# Backwards-compatible export. Prefer calling ``_supported_suffixes()`` from
# inside this module — the property on the registry is re-evaluated on every
# read so third-party extractors registered at runtime become visible to
# ``_accept_file`` immediately. A module-level snapshot froze the set at
# import time and quietly skipped late-registered suffixes during the walk.
SUPPORTED_SUFFIXES = DEFAULT_REGISTRY.supported_suffixes


def _supported_suffixes() -> frozenset[str]:
    return DEFAULT_REGISTRY.supported_suffixes


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
    if p.suffix.lower() not in _supported_suffixes():
        return False
    if st.st_size > MAX_INDEX_FILE_BYTES:
        log.warning(
            "skipping %s: %d bytes exceeds MAX_INDEX_FILE_BYTES (%d)",
            p, st.st_size, MAX_INDEX_FILE_BYTES,
        )
        print(
            f"[hybrid-rag] skipping {p}: size exceeds "
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
    return DEFAULT_REGISTRY.extract(path)


# --- per-file write helpers ----------------------------------------------


def _persist_file_row(store: Store, path: Path, *, conn=None) -> int:
    """Insert one `files` row and return its id."""
    st = path.stat()
    row = (str(path), st.st_mtime, st.st_size, _hash_file(path),
           path.suffix.lower().lstrip("."), time.time())
    return store.bulk_insert_files([row], conn=conn)[str(path)]


def _persist_parent_rows(store: Store, file_id: int,
                         parents: list[Parent], *, conn=None) -> list[int]:
    """Insert `parents` rows under `file_id`, return ids in input order."""
    rows = [(file_id, i, p.kind, p.title, p.page_no, p.text, len(p.text))
            for i, p in enumerate(parents)]
    return store.bulk_insert_parents(rows, conn=conn)


def _generate_contextual_prefixes(
    parent_text: str, pieces: list[str],
) -> list[str | None]:
    """Concurrent fan-out for one parent's chunks.

    All requests for chunks of one parent share an Anthropic prompt-cache
    entry (the parent block is the cached content), so issuing them in
    parallel multiplies throughput without multiplying token cost.
    `generate_contextual_prefix` swallows its own exceptions, so a flaky
    call returns None instead of raising.
    """
    with ThreadPoolExecutor(max_workers=CONTEXTUAL_CONCURRENCY) as ex:
        futures = [
            ex.submit(contextual.generate_contextual_prefix, parent_text, piece)
            for piece in pieces
        ]
        return [f.result() for f in futures]


def _chunk_one_parent(store: Store, parent_id: int, parent: Parent,
                      *, use_contextual: bool, conn=None) -> int:
    """Split `parent` into chunks, optionally generate contextual prefixes,
    insert into `chunks`. Returns the number of rows inserted (0 if the
    parent yielded no usable pieces)."""
    pieces = recursive_split(parent.text, max_size=MAX_CHUNK,
                             overlap=CHUNK_OVERLAP) or [parent.text]
    # Drop any whitespace-only piece so we don't insert empty chunk rows —
    # extract_* should already filter these, but a degenerate parent (a
    # paragraph of only whitespace) can still slip through here.
    pieces = [p for p in pieces if p and p.strip()]
    if not pieces:
        return 0

    prefixes: list[str | None]
    if use_contextual:
        prefixes = _generate_contextual_prefixes(parent.text, pieces)
    else:
        prefixes = [None] * len(pieces)

    chunk_rows: list[tuple] = []
    for ord_, (piece, prefix) in enumerate(zip(pieces, prefixes)):
        if prefix:
            composed = prefix + "\n\n" + piece
            # `text_for_bm25` is left NULL when it equals `text_for_embedding`
            # — readers fall through (see `ChunkRow.effective_bm25_text`).
            # Saves one copy per chunk on disk; today BM25 and dense always
            # use the same composed text under contextual retrieval.
            chunk_rows.append(
                (parent_id, ord_, piece, 0, prefix, composed, None)
            )
        else:
            chunk_rows.append((parent_id, ord_, piece, 0, None, None, None))
    store.bulk_insert_chunks(chunk_rows, conn=conn)
    return len(chunk_rows)


def _index_file(store: Store, path: Path, *, conn=None) -> tuple[int, int, int]:
    """Insert one file's parents and chunks. Returns (file_id, parent_count,
    chunk_count). Caller is responsible for picking up an existing file row's
    deletion before calling this."""
    raw_parents = _extract_parents(path)
    parents = _enforce_parent_cap(raw_parents)

    file_id = _persist_file_row(store, path, conn=conn)
    if not parents:
        return file_id, 0, 0

    parent_ids = _persist_parent_rows(store, file_id, parents, conn=conn)
    use_contextual = contextual.is_contextual_enabled()

    chunk_count = 0
    for pid, parent in zip(parent_ids, parents):
        chunk_count += _chunk_one_parent(
            store, pid, parent, use_contextual=use_contextual, conn=conn,
        )
    return file_id, len(parents), chunk_count


# --- top-level orchestrator ----------------------------------------------


@dataclass
class _Changeset:
    """The four buckets `index_path` operates on after the manifest diff."""
    changed: list[tuple[Path, int]]   # (path, file_id) — must delete then re-insert
    new: list[Path]                   # path → fresh insert
    deleted_ids: list[int]            # file_ids whose paths are gone from disk
    unchanged: list[Path]             # nothing to do — reported in summary


def _compute_changeset(store: Store, files: list[Path], diff: dict,
                       force: bool) -> _Changeset:
    if not force:
        return _Changeset(
            changed=diff["changed"], new=diff["new"],
            deleted_ids=diff["deleted"], unchanged=diff["unchanged"],
        )
    # Force mode: treat every file on disk as changed (or new). One SQL
    # fetch covers both buckets — no per-file round-trip.
    existing_ids = {Path(r["path"]): r["id"]
                    for r in store.connect().execute(
                        "SELECT id, path FROM files")}
    return _Changeset(
        changed=[(p, existing_ids[p]) for p in files if p in existing_ids],
        new=[p for p in files if p not in existing_ids],
        deleted_ids=diff["deleted"],
        unchanged=[],
    )


def _apply_inserts(store: Store, paths: list[Path]) -> tuple[int, int, int]:
    """Index each path under one outer transaction with per-file savepoints.

    Pre-A1 behavior: each file opened three independent transactions (one
    per `bulk_insert_*` call) — so a 1000-file index did ~3000 fsyncs even
    in WAL mode. Now the whole batch is one transaction, and a file that
    fails mid-way rolls back via its savepoint without affecting the
    successful files. Net: O(N) commits collapse to O(1).

    Returns (files, parents, chunks) totals.
    """
    files = parents_total = chunks = 0
    if not paths:
        return 0, 0, 0
    with store.transaction() as conn:
        for p in paths:
            try:
                with store.savepoint("file"):
                    _, pn, cn = _index_file(store, p, conn=conn)
            except Exception as e:
                # Savepoint rolled back this file's partial rows; the other
                # successful files in the same outer transaction still
                # commit at the end. Warning to stderr so the CLI's JSON
                # summary on stdout stays parseable.
                log.warning("failed to index %s: %s", p, e)
                print(f"[hybrid-rag] failed to index {p}: {e}", file=sys.stderr)
                continue
            files += 1
            parents_total += pn
            chunks += cn
    return files, parents_total, chunks


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
    # Pass `_hash_file` as the (mtime, size) tiebreaker so in-place edits that
    # preserved both fields still get reindexed. Cost: one SHA-256 per file
    # whose (mtime, size) match — files with stale stats short-circuit.
    diff = manifest.diff(own_store, disk_map, hash_fn=_hash_file)
    changeset = _compute_changeset(own_store, files, diff, force)

    # Deletes first (cascades to parents/chunks).
    obsolete = list(changeset.deleted_ids) + [fid for _, fid in changeset.changed]
    own_store.delete_files(obsolete)

    # Changed paths get reinserted; new paths fall straight through.
    ordered_inserts = [p for p, _ in changeset.changed] + changeset.new
    new_files, new_parents, new_chunks = _apply_inserts(own_store, ordered_inserts)

    rebuild_artifacts(own_store, embedder)

    return {
        "indexed_root": str(root),
        "files_added_or_updated": new_files,
        "files_unchanged": len(changeset.unchanged),
        "files_deleted": len(diff["deleted"]),
        "parents": new_parents,
        "chunks": new_chunks,
        "totals": own_store.stats(),
    }


def _build_bm25_state(bm25_texts: list[str]) -> dict:
    """Build a BM25Okapi over `bm25_texts` and serialize its state to a
    JSON-safe dict.

    The dict is the exact set of attributes our `rank_bm25.BM25Okapi`
    reconstruction in `engine.py` expects — keep the two in lockstep
    when the shape changes.
    """
    from rank_bm25 import BM25Okapi

    from .retrieval import _tokenize

    tokenized = [_tokenize(t) for t in bm25_texts]
    bm25 = BM25Okapi(tokenized)
    return {
        "corpus_size": int(bm25.corpus_size),
        "avgdl": float(bm25.avgdl),
        "average_idf": float(bm25.average_idf),
        "k1": float(bm25.k1),
        "b": float(bm25.b),
        "epsilon": float(bm25.epsilon),
        # `doc_freqs` is a list of per-doc {term: freq} dicts. JSON
        # round-trips this losslessly (all keys are tokenizer-emitted
        # ASCII strings, all values are ints).
        "doc_freqs": [dict(d) for d in bm25.doc_freqs],
        "idf": dict(bm25.idf),
        "doc_len": [int(x) for x in bm25.doc_len],
    }


def rebuild_artifacts(store: Store, embedder) -> None:
    """Rebuild embeddings.npz and the BM25 sidecar from the canonical
    SQLite chunk order. Also rewrites each chunk's ``embed_row`` so row N
    of the embeddings array maps to chunk_ids[N]. Persisting the BM25
    state at index time means engine load is a JSON decode (~100 ms on
    100K chunks) instead of a re-tokenize + re-build (1-3 s).

    Always bumps ``index_version`` — including the empty-corpus path —
    so a live engine that previously loaded a non-empty index notices
    the corpus is now empty and reloads, instead of serving cached
    results against deleted chunks.
    """
    rows = list(store.iter_chunks_ordered())
    artifacts = ArtifactStore(store.data_dir)
    artifacts.unlink_legacy_bm25()

    if not rows:
        artifacts.delete()
    else:
        embed_texts = [r.effective_embedding_text for r in rows]
        bm25_texts = [r.effective_bm25_text for r in rows]
        chunk_ids = [r.id for r in rows]

        embeddings = embedder.encode(embed_texts)
        if embeddings.shape[0] != len(embed_texts):
            raise RuntimeError("embedder returned wrong number of vectors")

        artifacts.save(embeddings)
        artifacts.save_bm25_state(_build_bm25_state(bm25_texts))
        store.bulk_update_embed_rows([(cid, row) for row, cid in enumerate(chunk_ids)])

        model_name = getattr(embedder, "model_name", None) or "unknown"
        store.set_meta("embed_model", str(model_name))
        store.set_meta("embed_dim", str(int(embeddings.shape[1])))

    # Bump the version counter so any RAGEngine pointed at this data dir
    # reloads on its next call. Replaces the old explicit engine.reset()
    # call from index_path — see engine._ensure_loaded for the consumer.
    store.set_meta("index_version", str(time.monotonic_ns()))
