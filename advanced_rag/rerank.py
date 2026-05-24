"""Reranker. Tries Cohere first (if COHERE_API_KEY set), falls back to a local
cross-encoder, falls back to identity. Failures never reach the caller.
"""
from __future__ import annotations

import logging
import os

from .config import COHERE_RERANK_MODEL, RERANK_MODEL
from .retrieval import ParentResult

log = logging.getLogger(__name__)

_CROSS = None  # module-level cache for the local cross-encoder


def _try_cohere(query: str, parents: list[ParentResult], top_k: int) -> list[ParentResult] | None:
    if not os.environ.get("COHERE_API_KEY"):
        return None
    try:
        import cohere
    except ImportError:
        return None

    try:
        client_cls = getattr(cohere, "ClientV2", None) or cohere.Client
        # Let the SDK pull the key from COHERE_API_KEY itself. Passing it
        # positionally invites it into constructor-arg traces / reprs.
        client = client_cls()
        response = client.rerank(
            model=COHERE_RERANK_MODEL,
            query=query,
            documents=[p.text for p in parents],
            top_n=min(top_k, len(parents)),
        )
        ordered: list[ParentResult] = []
        for r in response.results:
            idx = getattr(r, "index", None)
            score = getattr(r, "relevance_score", None)
            # Guard against bogus indices in either direction; negative values
            # would silently wrap around to index from the end of `parents`.
            if not isinstance(idx, int) or idx < 0 or idx >= len(parents):
                continue
            p = parents[idx]
            p.rerank_score = float(score) if score is not None else None
            ordered.append(p)
        return ordered
    except Exception as e:
        log.warning("Cohere rerank failed, falling back: %s", e)
        return None


def _try_local_cross_encoder(query: str, parents: list[ParentResult], top_k: int) -> list[ParentResult] | None:
    global _CROSS
    try:
        if _CROSS is None:
            from sentence_transformers import CrossEncoder
            _CROSS = CrossEncoder(RERANK_MODEL)
        pairs = [(query, p.text[:2000]) for p in parents]
        scores = _CROSS.predict(pairs)
        ranked = sorted(zip(parents, scores), key=lambda kv: -float(kv[1]))[:top_k]
        out: list[ParentResult] = []
        for p, s in ranked:
            p.rerank_score = float(s)
            out.append(p)
        return out
    except Exception as e:
        log.warning("local cross-encoder rerank failed: %s", e)
        return None


def rerank(query: str, parents: list[ParentResult], top_k: int) -> list[ParentResult]:
    """Rerank `parents` by relevance to `query` and return the top `top_k`.

    **Mutates input.** The chosen reranker writes its score back onto each
    returned ``ParentResult`` via ``p.rerank_score = ...``. Callers that need
    to rerank the same list more than once (e.g. A/B testing models) should
    pass copies; otherwise a stale ``rerank_score`` from the previous call
    will leak into the next.
    """
    if not parents:
        return []
    cohere_out = _try_cohere(query, parents, top_k)
    # Truthy check (not `is not None`): an empty Cohere result must fall
    # through to the local reranker rather than hand the user back nothing.
    if cohere_out:
        return cohere_out
    local_out = _try_local_cross_encoder(query, parents, top_k)
    if local_out:
        return local_out
    # identity fallback
    return parents[:top_k]


def rerank_local(query: str, parents: list[ParentResult], top_k: int) -> list[ParentResult]:
    """Local-cross-encoder-only variant used by the **ambient** path (Phase 3).

    Cohere is intentionally never called from the ambient path: a per-turn
    HTTP round-trip would defeat the purpose of the cheap injection layer.
    The explicit `rag_search` path keeps using `rerank` (Cohere → local →
    identity).
    """
    if not parents:
        return []
    local_out = _try_local_cross_encoder(query, parents, top_k)
    if local_out:
        return local_out
    return parents[:top_k]


def warm_local_cross_encoder() -> None:
    """Pre-load the local cross-encoder so the first ambient rerank is hot.

    Called from `on_session_start` (in a background thread). Silent on any
    failure — cold load on the first ambient rerank is the fallback.
    """
    global _CROSS
    if _CROSS is not None:
        return
    try:
        from sentence_transformers import CrossEncoder
        _CROSS = CrossEncoder(RERANK_MODEL)
    except Exception as e:
        log.debug("cross-encoder warm-up failed (will retry on demand): %s", e)
