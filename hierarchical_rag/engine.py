"""Process-wide RAG engine. Holds the (lazily loaded) BM25, embeddings array,
chunk_id list, embedder, and store. `reset()` drops cached state so a re-index
flushes the next query.
"""
from __future__ import annotations

import threading

from .config import bm25_path, npz_path
from .storage import Store

_INSTANCE = None
_INSTANCE_LOCK = threading.Lock()


class RAGEngine:
    def __init__(self, store: Store | None = None, embedder=None):
        self._store = store or Store()
        self._embedder = embedder
        self._bm25 = None
        self._embeddings = None
        self._chunk_ids: list[int] = []
        self._loaded = False
        self._lock = threading.Lock()

    @property
    def store(self) -> Store:
        return self._store

    def _make_default_embedder(self):
        from .embeddings import Embedder
        return Embedder()

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            if self._embedder is None:
                self._embedder = self._make_default_embedder()

            npz_p = npz_path(self._store.data_dir)
            bm25_p = bm25_path(self._store.data_dir)

            if npz_p.exists():
                embeddings, chunk_ids = self._store.load_embeddings(npz_p)
                self._embeddings = embeddings
                self._chunk_ids = list(chunk_ids)
            else:
                self._embeddings = None
                self._chunk_ids = []

            if bm25_p.exists():
                self._bm25 = self._store.load_bm25(bm25_p)
            else:
                self._bm25 = None

            self._loaded = True

    def reset(self) -> None:
        with self._lock:
            self._bm25 = None
            self._embeddings = None
            self._chunk_ids = []
            self._loaded = False


def get_engine() -> RAGEngine:
    global _INSTANCE
    if _INSTANCE is None:
        with _INSTANCE_LOCK:
            if _INSTANCE is None:
                _INSTANCE = RAGEngine()
    return _INSTANCE


def set_engine_for_tests(engine: RAGEngine | None) -> None:
    """Test-only helper. Replaces the singleton (or clears it with None)."""
    global _INSTANCE
    _INSTANCE = engine
