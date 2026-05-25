"""Process-wide RAG engine. Holds the (lazily loaded) BM25, embeddings array,
chunk_id list, embedder, and store, and exposes ``hybrid_search`` as the
public retrieval entry point. Callers never reach into ``_bm25`` /
``_embeddings`` directly — they go through the method, which folds the
lazy-load behind the public surface.

BM25 is rebuilt from SQLite at load time rather than read from a pickle —
see ``storage.py`` module docstring for the security reasoning.
"""
from __future__ import annotations

import logging
import threading
import time

import numpy as np

from . import validation
from .artifacts import ArtifactStore
from .models import Hit
from .storage import Store

log = logging.getLogger(__name__)

_INSTANCE = None
_INSTANCE_LOCK = threading.Lock()


# Backwards-compatible alias: tests and downstream code import this name
# from ``engine``. Internally it's just the validation module's error type.
EngineLoadError = validation.IndexConsistencyError


class RAGEngine:
    # Time-to-live for the cached `index_version` meta read. Every query
    # consults this value to decide whether to reload — without the cache
    # that's one SQLite roundtrip per `hybrid_search` (×5 variants per
    # `rag_search`, plus one per ambient turn). A 1-second TTL caps the
    # observable reindex visibility lag at the same granularity as
    # `state._CACHE_TTL`; the indexing CLI doesn't expect sub-second
    # propagation anyway.
    _VERSION_CACHE_TTL = 1.0

    def __init__(self, store: Store | None = None, embedder=None):
        self._store = store or Store()
        self._embedder = embedder
        self._bm25 = None
        self._embeddings: np.ndarray | None = None
        self._chunk_ids: list[int] = []
        self._loaded = False
        # The index version that was on disk the last time we loaded. A
        # reindex bumps the meta key; the next call sees the mismatch and
        # transparently reloads — no caller has to invoke `reset()`.
        self._loaded_version: str | None = None
        self._version_cache: str | None = None
        self._version_cache_ts: float = 0.0
        # The Store's meta-write counter at the time we filled the cache.
        # Same-process writes (test, in-process reindex) bump the counter
        # so the cache invalidates immediately; cross-process writers
        # don't share it and fall back to the TTL.
        self._version_cache_seq: int = -1
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def store(self) -> Store:
        return self._store

    def hybrid_search(
        self,
        query: str,
        *,
        qvec: np.ndarray | None = None,
        k_pool: int = 30,
    ) -> list[Hit]:
        """BM25 + dense top-k fused with RRF. Lazy-loads on first call so
        callers never need to invoke ``_ensure_loaded`` themselves.

        ``qvec`` is the optional pre-computed query vector — supplied by the
        ambient path when convo memory mixes the current query embedding with
        prior turns. When ``None``, the configured embedder encodes ``query``.
        BM25 always operates on the literal ``query`` regardless, so lexical
        search isn't contaminated by mixed embeddings.
        """
        # Local import to avoid a top-level cycle. retrieval reuses our
        # internals via `engine_like.bm25/embeddings/chunk_ids/...`.
        from .retrieval import hybrid_search as _retrieval_hybrid_search

        self._ensure_loaded()
        return _retrieval_hybrid_search(self, query, qvec=qvec, k_pool=k_pool)

    def hybrid_search_chunk_ids(
        self,
        query: str,
        *,
        qvec: np.ndarray | None = None,
        k_pool: int = 30,
    ) -> list[int]:
        """Chunk-IDs-only counterpart to ``hybrid_search``. Used by the
        explicit pipeline's variant loop — per-variant parent resolution
        is wasted work since only the post-fusion list needs parents."""
        from .retrieval import hybrid_search_chunk_ids as _retrieval_chunk_ids

        self._ensure_loaded()
        return _retrieval_chunk_ids(self, query, qvec=qvec, k_pool=k_pool)

    def encode_query(self, query: str) -> np.ndarray | None:
        """Encode `query` to a single L2-normalized vector, or None if the
        embedder returned empty. Used by the ambient path's convo-memory
        mixer."""
        self._ensure_loaded()
        if self._embedder is None:
            return None
        batch = self._embedder.encode([query])
        if batch.shape[0] == 0:
            return None
        return batch[0]

    def has_embeddings(self) -> bool:
        """True iff there's at least one row available to score against."""
        self._ensure_loaded()
        return self._embeddings is not None and self._embeddings.shape[0] > 0

    def reset(self) -> None:
        with self._lock:
            # Drop ``_loaded`` FIRST so any fast-path reader that races us
            # without the lock sees "not loaded" before the underlying
            # arrays are torn down. The fast path in ``_ensure_loaded``
            # checks ``self._loaded`` outside the lock; clearing arrays
            # first would briefly expose ``_loaded=True`` paired with
            # ``_embeddings=None`` and crash retrieval with a NoneType
            # error instead of cleanly falling through to the slow path.
            self._loaded = False
            self._loaded_version = None
            self._bm25 = None
            self._embeddings = None
            self._chunk_ids = []
            # Invalidate the version cache too so the next call re-reads
            # the meta key fresh — otherwise a test that calls `reset()`
            # then writes a new version within the same TTL window would
            # see the cached stale value and skip reloading.
            self._version_cache = None
            self._version_cache_ts = 0.0
            self._version_cache_seq = -1

    # ------------------------------------------------------------------
    # Read-only views — used by retrieval.py to score queries. These are
    # protocol surface, not a free invitation to mutate engine state.
    # ------------------------------------------------------------------

    @property
    def embedder(self):
        return self._embedder

    @property
    def embeddings(self):
        return self._embeddings

    @property
    def chunk_ids(self) -> list[int]:
        return self._chunk_ids

    @property
    def bm25(self):
        return self._bm25

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _make_default_embedder(self):
        from .embeddings import Embedder
        return Embedder()

    def _current_version(self, *, force: bool = False) -> str | None:
        """Cached read of the ``index_version`` meta key.

        Even a sub-millisecond SELECT adds up when it fires on every
        retrieval call (each `rag_search` runs 5 variant searches, each
        ambient turn runs one). The cache has two invalidation channels:

        1. **Same-process writer**: the Store's ``meta_write_seq`` counter
           increments on every ``set_meta`` call. We compare it; mismatch
           forces a fresh read. Tests and in-process reindex see new
           versions immediately.
        2. **Cross-process writer**: another process (the indexing CLI)
           updates meta without touching our counter. We fall back to a
           1-second TTL — that's the maximum staleness window the
           ambient hook can observe before noticing a fresh index.

        ``force=True`` bypasses both checks — used inside `_ensure_loaded`
        after the load completes.
        """
        seq = self._store.meta_write_seq
        now = time.monotonic()
        if (not force
                and seq == self._version_cache_seq
                and (now - self._version_cache_ts) < self._VERSION_CACHE_TTL):
            return self._version_cache
        v = self._store.get_meta("index_version")
        self._version_cache = v
        self._version_cache_ts = now
        self._version_cache_seq = seq
        return v

    def _ensure_loaded(self) -> None:
        # Fast path: already loaded AND the on-disk version still matches.
        # The version read goes through a 1-second TTL cache so the steady
        # state doesn't pay one SQLite SELECT per query. Indexing still
        # invalidates by bumping the meta key — the worst-case lag for the
        # engine to notice is the TTL.
        current_version = self._current_version()
        if self._loaded and current_version == self._loaded_version:
            return
        with self._lock:
            # Re-read inside the lock (bypassing the TTL cache so two
            # threads racing into the slow path don't both reload).
            current_version = self._current_version(force=True)
            if self._loaded and current_version == self._loaded_version:
                return
            if self._embedder is None:
                self._embedder = self._make_default_embedder()

            artifacts = ArtifactStore(self._store.data_dir)
            if artifacts.exists():
                self._embeddings = artifacts.load()
                self._chunk_ids = self._store.get_chunk_ids_ordered()
                self._bm25 = self._load_or_rebuild_bm25(artifacts)
            else:
                self._embeddings = None
                self._chunk_ids = []
                self._bm25 = None

            try:
                validation.validate(
                    self._embeddings, self._chunk_ids, self._store, self._embedder,
                )
            except validation.IndexConsistencyError:
                self._embeddings = None
                self._chunk_ids = []
                self._bm25 = None
                self._loaded = False
                self._loaded_version = None
                raise
            self._loaded = True
            self._loaded_version = current_version

    def _load_or_rebuild_bm25(self, artifacts: ArtifactStore):
        """Prefer the persisted BM25 state — way cheaper than re-tokenizing.

        Three failure modes fall through to the from-SQLite rebuild:
          * sidecar missing (older install, or `bm25_state.json` was deleted)
          * sidecar present but its `corpus_size` disagrees with the loaded
            chunk_ids (artifact desync after a partial rebuild)
          * any decode/reconstruction error — we log and rebuild
        """
        if artifacts.bm25_state_exists():
            try:
                state = artifacts.load_bm25_state()
                bm25 = _bm25_from_state(state)
                if bm25 is not None and bm25.corpus_size == len(self._chunk_ids):
                    return bm25
                log.warning(
                    "BM25 sidecar corpus_size %s != chunk count %s — "
                    "rebuilding from SQLite",
                    getattr(bm25, "corpus_size", "?"), len(self._chunk_ids),
                )
            except Exception as e:
                log.warning("BM25 sidecar load failed (%s); rebuilding", e)
        return self._build_bm25()

    def _build_bm25(self):
        """Build BM25Okapi from the SQLite chunks table. Returns None for an
        empty corpus. Identical tokenizer to query time — ``retrieval._tokenize``
        is the single source so index- and query-side tokens stay aligned.

        Used only as a fallback today; the index path persists the BM25
        state to a sidecar so engine load can skip the tokenize step.
        """
        from rank_bm25 import BM25Okapi

        from .retrieval import _tokenize

        tokenized = [_tokenize(t) for t in self._store.iter_bm25_texts_ordered()]
        if not tokenized:
            return None
        return BM25Okapi(tokenized)


def _bm25_from_state(state: dict):
    """Reconstruct a `BM25Okapi` from a serialized state dict.

    We bypass `BM25Okapi.__init__` (which expects a tokenized corpus and
    runs `_initialize` + `_calc_idf` over it) by using `__new__` and
    populating attributes directly — those two methods are exactly the
    work we paid for at index time and don't want to repeat on every
    engine load. Returns None on schema mismatch so the caller can fall
    back to the rebuild path.
    """
    from rank_bm25 import BM25Okapi
    required = (
        "corpus_size", "avgdl", "average_idf", "k1", "b", "epsilon",
        "doc_freqs", "idf", "doc_len",
    )
    if not all(k in state for k in required):
        return None
    bm = BM25Okapi.__new__(BM25Okapi)
    bm.corpus_size = int(state["corpus_size"])
    bm.avgdl = float(state["avgdl"])
    bm.average_idf = float(state["average_idf"])
    bm.k1 = float(state["k1"])
    bm.b = float(state["b"])
    bm.epsilon = float(state["epsilon"])
    bm.doc_freqs = state["doc_freqs"]
    bm.idf = state["idf"]
    bm.doc_len = state["doc_len"]
    bm.tokenizer = None
    return bm


def get_engine() -> RAGEngine:
    global _INSTANCE
    if _INSTANCE is None:
        with _INSTANCE_LOCK:
            if _INSTANCE is None:
                _INSTANCE = RAGEngine()
    return _INSTANCE


def set_engine_for_tests(engine: RAGEngine) -> None:
    """Test-only helper. Replaces the process singleton with ``engine``.
    Use :func:`reset_for_tests` to clear instead."""
    global _INSTANCE
    _INSTANCE = engine


def reset_for_tests() -> None:
    """Test-only helper. Clears the process singleton."""
    global _INSTANCE
    _INSTANCE = None
