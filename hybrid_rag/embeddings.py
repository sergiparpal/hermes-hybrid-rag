"""Lazy embedder wrapper. Model selected by HERMES_RAG_EMBED_MODEL (default
BAAI/bge-m3). The model isn't loaded until the first encode() call.

Tests inject a stub embedder rather than this real one — keeps
`sentence-transformers` out of the dev install.
"""
from __future__ import annotations

import os

import numpy as np


# Modern multilingual default. Override at runtime with HERMES_RAG_EMBED_MODEL
# (any sentence-transformers-compatible id). Switching models requires a full
# `hermes rag index --force` — the .npz dim won't match otherwise.
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"
# Known dimensionalities. Used to pre-allocate the `(0, dim)` empty-result
# array in `Embedder.encode([])` without loading the model. Anything missing
# here is auto-detected on first load by querying the model itself.
EMBED_MODEL_DIMS: dict[str, int] = {
    "all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-m3": 1024,
}


def get_embed_model() -> str:
    """Configured embedding model id. HERMES_RAG_EMBED_MODEL overrides the
    default; unset → DEFAULT_EMBED_MODEL."""
    env = os.environ.get("HERMES_RAG_EMBED_MODEL")
    return env.strip() if env and env.strip() else DEFAULT_EMBED_MODEL


def get_embed_dim() -> int | None:
    """Optional manual dimension override via HERMES_RAG_EMBED_DIM. Returns
    None when unset/invalid so the Embedder auto-detects from the loaded
    model."""
    env = os.environ.get("HERMES_RAG_EMBED_DIM")
    if not env:
        return None
    try:
        n = int(env)
        return n if n > 0 else None
    except ValueError:
        return None


class Embedder:
    def __init__(self, model_name: str | None = None,
                 dim: int | None = None):
        # Resolve at construction time so each Embedder pins one (model, dim)
        # pair — important because the engine compares this against the meta
        # row written at index time.
        self._model_name = model_name if model_name is not None else get_embed_model()
        self._model = None
        # Precedence: explicit ctor arg > HERMES_RAG_EMBED_DIM > known-models
        # table > auto-detect on first encode.
        env_dim = get_embed_dim() if dim is None else dim
        self._dim: int | None = env_dim if env_dim else EMBED_MODEL_DIMS.get(self._model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int | None:
        """Vector dimension. Returns None only if the model hasn't been loaded
        yet AND wasn't pre-registered. Once `encode()` has run for any model
        once, this becomes available even for unknown ids."""
        return self._dim

    def _load_model(self):
        if self._model is not None:
            return self._model
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self._model_name)
        # Discover the real dim once and remember it. Models we don't know
        # up front (any user-supplied id) get auto-registered here.
        try:
            real = int(self._model.get_sentence_embedding_dimension())
        except Exception:
            real = None
        if real:
            if self._dim and self._dim != real:
                raise RuntimeError(
                    f"HERMES_RAG_EMBED_DIM={self._dim} disagrees with model "
                    f"{self._model_name!r} actual dim {real}. Unset the env "
                    "var or re-run `hermes rag index --force`."
                )
            self._dim = real
        return self._model

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            # SentenceTransformer raises on an empty list; short-circuit. The
            # shape uses the resolved dim so callers swapping models don't get
            # a silent (0, 384) misfit.
            dim = self._dim
            if dim is None:
                self._load_model()
                dim = self._dim
            if dim is None:
                raise RuntimeError(
                    f"could not determine embedding dim for "
                    f"{self._model_name!r}: model load did not expose "
                    "`get_sentence_embedding_dimension`. Set HERMES_RAG_EMBED_DIM "
                    "explicitly to skip auto-detect."
                )
            return np.zeros((0, dim), dtype=np.float32)
        self._load_model()
        vecs = self._model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32)
