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
        client = client_cls(os.environ["COHERE_API_KEY"])
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
            if idx is None or idx >= len(parents):
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
    if not parents:
        return []
    cohere_out = _try_cohere(query, parents, top_k)
    if cohere_out is not None:
        return cohere_out
    local_out = _try_local_cross_encoder(query, parents, top_k)
    if local_out is not None:
        return local_out
    # identity fallback
    return parents[:top_k]
