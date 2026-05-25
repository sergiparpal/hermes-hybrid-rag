"""Post-load consistency checks for the index artifacts.

Separated from ``engine.py`` so the engine focuses on state ownership and
this module focuses on integrity — they have different reasons to change.
A partial rebuild can leave ``.npz`` and the SQLite ``chunks`` table in
inconsistent states; catching the mismatch here is much better than letting
it surface as an ``IndexError`` deep in retrieval.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


class IndexConsistencyError(RuntimeError):
    """Raised when on-disk index artifacts disagree about cardinality or
    model identity. Aliased as ``EngineLoadError`` in ``engine.py`` for
    backwards-compatible imports from tests."""


def validate(
    embeddings: np.ndarray | None,
    chunk_ids: list[int],
    store,
    embedder,
) -> None:
    """Raise ``IndexConsistencyError`` if any artifact disagrees.

    Returns ``None`` on success. The order matters: cardinality first (cheap),
    then dim checks against disk meta and against the configured embedder,
    then a non-fatal model-drift warning.
    """
    if embeddings is None:
        return
    _check_cardinality(embeddings, chunk_ids)
    _check_disk_dim(embeddings, store)
    _check_configured_dim(embeddings, embedder)
    _check_model_drift(store, embedder)


def _check_cardinality(embeddings: np.ndarray, chunk_ids: list[int]) -> None:
    n_emb = int(embeddings.shape[0])
    n_ids = len(chunk_ids)
    if n_emb != n_ids:
        raise IndexConsistencyError(
            f"embeddings array has {n_emb} rows but SQLite has "
            f"{n_ids} chunks — re-run `hermes rag index <path> --force`."
        )


def _check_disk_dim(embeddings: np.ndarray, store) -> None:
    """Catch the silent corruption case where the .npz was built with a
    different model than the currently configured one."""
    on_disk_dim = store.get_meta("embed_dim")
    if on_disk_dim is None:
        return
    try:
        disk_dim = int(on_disk_dim)
    except ValueError:
        return
    live_dim = int(embeddings.shape[1])
    if disk_dim != live_dim:
        raise IndexConsistencyError(
            f"embeddings.npz dim {live_dim} disagrees with stored "
            f"meta dim {disk_dim} — re-run "
            "`hermes rag index <path> --force`."
        )


def _check_configured_dim(embeddings: np.ndarray, embedder) -> None:
    configured_dim = getattr(embedder, "dim", None)
    if not configured_dim:
        return
    live_dim = int(embeddings.shape[1])
    if configured_dim != live_dim:
        raise IndexConsistencyError(
            f"configured embedder dim {configured_dim} disagrees with "
            f".npz dim {live_dim} — re-run "
            "`hermes rag index <path> --force` "
            "(or unset HERMES_RAG_EMBED_MODEL / HERMES_RAG_EMBED_DIM)."
        )


def _check_model_drift(store, embedder) -> None:
    """Dim matches but the model name doesn't — quality may degrade. Loud,
    but non-fatal: we still serve queries."""
    on_disk_model = store.get_meta("embed_model")
    configured_model = getattr(embedder, "model_name", None)
    if on_disk_model and configured_model and on_disk_model != configured_model:
        log.warning(
            "embedding-model drift: index was built with %r but the "
            "current configuration is %r. Dimensions match so retrieval "
            "will still run, but quality may degrade until a "
            "`hermes rag index --force` rebuilds the .npz.",
            on_disk_model, configured_model,
        )
