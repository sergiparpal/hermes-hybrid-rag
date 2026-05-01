"""Lazy MiniLM wrapper. The model isn't loaded until the first encode() call.

Tests inject a stub embedder rather than this real one — keeps `sentence-transformers`
out of the dev install.
"""
from __future__ import annotations

import numpy as np

from .config import EMBED_MODEL


class Embedder:
    def __init__(self, model_name: str = EMBED_MODEL):
        self._model_name = model_name
        self._model = None

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        if not texts:
            # SentenceTransformer raises on an empty list; short-circuit.
            return np.zeros((0, 384), dtype=np.float32)
        vecs = self._model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32)
