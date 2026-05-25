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
            self._bm25 = None
            self._embeddings = None
            self._chunk_ids = []
            self._loaded = False
            self._loaded_version = None

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

    def _ensure_loaded(self) -> None:
        # Fast path: already loaded AND the on-disk version still matches.
        # The meta read is one tiny SELECT — sub-millisecond — and lets
        # indexing invalidate by writing the meta key instead of calling
        # back into engine.reset() (which would couple the two modules).
        current_version = self._store.get_meta("index_version")
        if self._loaded and current_version == self._loaded_version:
            return
        with self._lock:
            current_version = self._store.get_meta("index_version")
            if self._loaded and current_version == self._loaded_version:
                return
            if self._embedder is None:
                self._embedder = self._make_default_embedder()

            artifacts = ArtifactStore(self._store.data_dir)
            if artifacts.exists():
                self._embeddings = artifacts.load()
                self._chunk_ids = self._store.get_chunk_ids_ordered()
                self._bm25 = self._build_bm25()
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

    def _build_bm25(self):
        """Build BM25Okapi from the SQLite chunks table. Returns None for an
        empty corpus. Identical tokenizer to query time — ``retrieval._tokenize``
        is the single source so index- and query-side tokens stay aligned.
        """
        from rank_bm25 import BM25Okapi

        from .retrieval import _tokenize

        tokenized = [_tokenize(t) for t in self._store.iter_bm25_texts_ordered()]
        if not tokenized:
            return None
        return BM25Okapi(tokenized)


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
