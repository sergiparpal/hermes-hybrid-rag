"""Reranker. Tries Cohere first (if COHERE_API_KEY set), falls back to a local
cross-encoder, falls back to identity. Failures never reach the caller.

The two reranker entry points (`rerank` and `rerank_local`) are
non-mutating: input parents are never modified in place. Each returned
``ParentResult`` is a fresh copy carrying the new ``rerank_score`` — so a
caller can rerank the same list with different scorers (A/B testing) and
trust that the inputs aren't leaking state between calls.
"""
from __future__ import annotations

import dataclasses
import logging
import os
from operator import itemgetter

from .models import ParentResult

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COHERE_RERANK_MODEL = "rerank-english-v3.0"

# Per-parent text length capped before scoring. The cross-encoder is
# trained on ~512-token passages; feeding 2000 chars just lets the model
# internally truncate at a higher cost. The explicit `rag_search` path
# uses the wider cap so longer context can sway the score — it pays the
# Cohere round-trip anyway. The ambient path runs every turn and must
# stay cheap, so it sets a tighter cap (`AMBIENT_RERANK_TEXT_CHARS`).
DEFAULT_RERANK_TEXT_CHARS = 2000
AMBIENT_RERANK_TEXT_CHARS = 512

log = logging.getLogger(__name__)

_CROSS = None  # module-level cache for the local cross-encoder


def _scored(parent: ParentResult, score: float | None) -> ParentResult:
    """Return a copy of ``parent`` with ``rerank_score`` set to ``score``.

    `dataclasses.replace` is the non-mutating equivalent of
    ``p.rerank_score = score`` — letting the reranker pipeline build a
    fresh ordering without leaking state back into the caller's list.
    """
    return dataclasses.replace(parent, rerank_score=score)


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
            ordered.append(_scored(
                parents[idx],
                float(score) if score is not None else None,
            ))
        return ordered
    except Exception as e:
        log.warning("Cohere rerank failed, falling back: %s", e)
        return None


def _try_local_cross_encoder(
    query: str,
    parents: list[ParentResult],
    top_k: int,
    *,
    text_cap: int = DEFAULT_RERANK_TEXT_CHARS,
) -> list[ParentResult] | None:
    global _CROSS
    try:
        if _CROSS is None:
            from sentence_transformers import CrossEncoder
            _CROSS = CrossEncoder(RERANK_MODEL)
        pairs = [(query, p.text[:text_cap]) for p in parents]
        scores = _CROSS.predict(pairs)
        ranked = sorted(
            zip(parents, scores), key=itemgetter(1), reverse=True,
        )[:top_k]
        return [_scored(p, float(s)) for p, s in ranked]
    except Exception as e:
        log.warning("local cross-encoder rerank failed: %s", e)
        return None


def rerank(
    query: str,
    parents: list[ParentResult],
    top_k: int,
    *,
    text_cap: int = DEFAULT_RERANK_TEXT_CHARS,
) -> list[ParentResult]:
    """Rerank `parents` by relevance to `query` and return the top `top_k`.

    Non-mutating: returned ``ParentResult`` objects are fresh copies — the
    input list and its elements are untouched. Callers running multiple
    rerank passes (A/B model comparison) can pass the same list twice
    without stale scores leaking across calls.

    ``text_cap`` is applied only by the local cross-encoder fallback —
    Cohere is handed each parent's ``text`` as-is. Upstream
    (``chunks_to_parents``) already pre-truncates pool text to
    ``DEFAULT_RERANK_TEXT_CHARS`` to bound the rerank-stage I/O, so in
    practice Cohere also sees capped text; if you ever want Cohere to
    score full bodies, lift the pre-truncation upstream first.
    """
    if not parents:
        return []
    cohere_out = _try_cohere(query, parents, top_k)
    # Truthy check (not `is not None`): an empty Cohere result must fall
    # through to the local reranker rather than hand the user back nothing.
    if cohere_out:
        return cohere_out
    local_out = _try_local_cross_encoder(query, parents, top_k, text_cap=text_cap)
    if local_out:
        return local_out
    # Identity fallback. Still return fresh copies so the contract holds
    # uniformly — callers never get back the same object they passed in.
    return [_scored(p, None) for p in parents[:top_k]]


def rerank_local(
    query: str,
    parents: list[ParentResult],
    top_k: int,
    *,
    text_cap: int = DEFAULT_RERANK_TEXT_CHARS,
) -> list[ParentResult]:
    """Local-cross-encoder-only variant used by the **ambient** path.

    Cohere is intentionally never called from the ambient path: a per-turn
    HTTP round-trip would defeat the purpose of the cheap injection layer.
    The explicit `rag_search` path keeps using `rerank` (Cohere → local →
    identity). Ambient callers pass `text_cap=AMBIENT_RERANK_TEXT_CHARS`
    to keep per-turn scoring cheap.
    """
    if not parents:
        return []
    local_out = _try_local_cross_encoder(query, parents, top_k, text_cap=text_cap)
    if local_out:
        return local_out
    return [_scored(p, None) for p in parents[:top_k]]


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
