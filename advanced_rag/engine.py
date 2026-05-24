"""Process-wide RAG engine. Holds the (lazily loaded) BM25, embeddings array,
chunk_id list, embedder, and store. `reset()` drops cached state so a re-index
flushes the next query.

BM25 is rebuilt from SQLite at load time rather than read from a pickle —
see `storage.py` module docstring for the security reasoning.
"""
from __future__ import annotations

import logging
import threading

from .config import npz_path
from .storage import Store

log = logging.getLogger(__name__)

_INSTANCE = None
_INSTANCE_LOCK = threading.Lock()


class EngineLoadError(RuntimeError):
    """Raised when the on-disk index artifacts are inconsistent. Surfaces a
    partial-failure scenario (e.g. .npz updated but bm25.pkl stale, or
    embed_row drift) instead of letting it manifest as a silent IndexError
    deep inside retrieval.
    """


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

            if npz_p.exists():
                self._embeddings = self._store.load_embeddings(npz_p)
                self._chunk_ids = self._store.get_chunk_ids_ordered()
                self._bm25 = self._build_bm25()
            else:
                self._embeddings = None
                self._chunk_ids = []
                self._bm25 = None

            self._check_consistency()
            self._loaded = True

    def _build_bm25(self):
        """Build BM25Okapi from the SQLite chunks table. Returns None for an
        empty corpus. Identical tokenizer to query time — `retrieval._tokenize`
        is the single source so index- and query-side tokens stay aligned.
        """
        from rank_bm25 import BM25Okapi

        from .retrieval import _tokenize

        tokenized = [_tokenize(t) for t in self._store.iter_bm25_texts_ordered()]
        if not tokenized:
            return None
        return BM25Okapi(tokenized)

    def _check_consistency(self) -> None:
        """Refuse to serve queries if the loaded artifacts disagree about
        cardinality — a partial rebuild can leave .npz and the SQLite chunks
        table in inconsistent states."""
        if self._embeddings is None:
            return
        n_emb = int(self._embeddings.shape[0])
        n_ids = len(self._chunk_ids)
        if n_emb != n_ids:
            self._embeddings = None
            self._chunk_ids = []
            self._bm25 = None
            raise EngineLoadError(
                f"embeddings array has {n_emb} rows but SQLite has "
                f"{n_ids} chunks — re-run `hermes rag index <path> --force`."
            )

        # Embedding-model alignment: catch the silent corruption case where
        # the .npz was built with a different model than the currently
        # configured one. Dim mismatch is fatal; pure-id mismatch (same dim,
        # different family) is loud-but-non-fatal.
        on_disk_dim = self._store.get_meta("embed_dim")
        live_dim = int(self._embeddings.shape[1])
        if on_disk_dim is not None:
            try:
                disk_dim = int(on_disk_dim)
            except ValueError:
                disk_dim = None
            if disk_dim is not None and disk_dim != live_dim:
                self._embeddings = None
                self._chunk_ids = []
                self._bm25 = None
                raise EngineLoadError(
                    f"embeddings.npz dim {live_dim} disagrees with stored "
                    f"meta dim {disk_dim} — re-run "
                    "`hermes rag index <path> --force`."
                )

        configured_dim = getattr(self._embedder, "dim", None)
        if configured_dim and configured_dim != live_dim:
            self._embeddings = None
            self._chunk_ids = []
            self._bm25 = None
            raise EngineLoadError(
                f"configured embedder dim {configured_dim} disagrees with "
                f".npz dim {live_dim} — re-run "
                "`hermes rag index <path> --force` "
                "(or unset HERMES_RAG_EMBED_MODEL / HERMES_RAG_EMBED_DIM)."
            )

        on_disk_model = self._store.get_meta("embed_model")
        configured_model = getattr(self._embedder, "model_name", None) or getattr(
            self._embedder, "_model_name", None
        )
        if on_disk_model and configured_model and on_disk_model != configured_model:
            log.warning(
                "embedding-model drift: index was built with %r but the "
                "current configuration is %r. Dimensions match so retrieval "
                "will still run, but quality may degrade until a "
                "`hermes rag index --force` rebuilds the .npz.",
                on_disk_model, configured_model,
            )

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
