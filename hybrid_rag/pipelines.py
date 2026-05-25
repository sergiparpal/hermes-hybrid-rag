"""End-to-end retrieval pipelines.

Two pipelines live here because they share ~70% of their shape but disagree
on the trimmings:

- ``ExplicitPipeline`` — invoked by ``rag_search``. Runs query expansion,
  multi-variant hybrid search, second-level RRF fusion on chunks, parent
  rollup, and the Cohere / cross-encoder rerank cascade. When
  ``HERMES_RAG_CRAG=1`` an LLM judge can trigger exactly one
  reformulate-and-retry pass.
- ``AmbientPipeline`` — invoked by ``hooks.ambient_pre_llm_call``. No query
  expansion, no CRAG; local-only reranker; gated by a relevance threshold;
  output wrapped in ``<retrieved_document>`` blocks.

Pulling both out of ``tools.py`` / ``hooks.py`` lets the modules above this
layer remain thin: ``tools.tool_rag_search`` becomes a JSON-shaped wrapper
over ``ExplicitPipeline.run``; ``hooks.ambient_pre_llm_call`` becomes a
guard + ``AmbientPipeline.run``.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from operator import itemgetter

from . import convo, crag, expansion, formatting, rerank, retrieval
from .config import RAG_SEARCH_CHUNK_POOL, RAG_SEARCH_PARENT_POOL
from .models import Hit, ParentResult

# --- Ambient tunables (consumer is the AmbientPipeline below) ---

AMBIENT_TOP_PARENTS = 3
AMBIENT_TOKEN_CAP = 1500
AMBIENT_SCORE_THRESHOLD = 0.25
# Top-K pool that survives the lightweight ambient rerank.
AMBIENT_RERANK_POOL = 10


# --- Explicit pipeline ---


@dataclass
class ExplicitResult:
    """One full ``ExplicitPipeline.run`` outcome. CRAG fields are populated
    only when CRAG ran AND triggered a retry."""
    parents: list[ParentResult] = field(default_factory=list)
    expansions_used: int = 0
    crag_reformulated_query: str | None = None
    crag_reason: str | None = None


class ExplicitPipeline:
    """Encapsulates the ``rag_search`` flow. Construct once per engine; call
    ``run(query, k)`` per request. Stateless across calls."""

    def __init__(self, engine):
        self._engine = engine

    def run(self, query: str, k: int) -> ExplicitResult:
        first = self._one_pass(query, k)

        if not (crag.is_enabled() and first.parents):
            return first

        verdict = crag.judge_retrieval(query, first.parents)
        if verdict.get("sufficient", True):
            return first

        # The judge said "not sufficient" — the reason is surfaced to the
        # caller regardless of whether reformulation succeeds. Useful for
        # logging when the retry can't run.
        reason = verdict.get("reason", "")
        new_q = crag.reformulate_query(query, first.parents, reason)
        if not (new_q and new_q.strip() and new_q.strip() != query.strip()):
            return ExplicitResult(
                parents=first.parents,
                expansions_used=first.expansions_used,
                crag_reformulated_query=None,
                crag_reason=reason,
            )

        retry = self._one_pass(new_q, k)
        return ExplicitResult(
            parents=retry.parents,
            expansions_used=retry.expansions_used,
            crag_reformulated_query=new_q,
            crag_reason=reason,
        )

    def _one_pass(self, query: str, k: int) -> ExplicitResult:
        """Single expansion → hybrid → fusion → rollup → rerank pass."""
        eng = self._engine
        store = eng.store

        variants = expansion.expand_query(query)
        per_variant: list[list[int]] = []
        for v in variants:
            hits = eng.hybrid_search(v, k_pool=RAG_SEARCH_CHUNK_POOL)
            per_variant.append([h.chunk_id for h in hits])

        fused = retrieval.rrf_fuse(per_variant)
        if not fused:
            return ExplicitResult(parents=[], expansions_used=len(variants))

        # Partial sort over the fused map, then batch the parent_id lookup —
        # avoids N individual SQL roundtrips on the inner loop.
        top_chunks = heapq.nlargest(
            RAG_SEARCH_CHUNK_POOL, fused.items(), key=itemgetter(1)
        )
        parent_by_chunk = store.parent_ids_for_chunks(
            [cid for cid, _ in top_chunks]
        )
        materialized: list[Hit] = [
            Hit(chunk_id=cid, score=float(score), parent_id=pid)
            for cid, score in top_chunks
            if (pid := parent_by_chunk.get(cid)) is not None
        ]

        parents = retrieval.chunks_to_parents(
            eng, materialized, top=RAG_SEARCH_PARENT_POOL,
        )
        reranked = rerank.rerank(query, parents, top_k=k)
        return ExplicitResult(parents=reranked, expansions_used=len(variants))


# --- Ambient pipeline ---


class AmbientPipeline:
    """Encapsulates the per-turn ambient injection flow. The hook layer
    (``hooks.ambient_pre_llm_call``) handles the Hermes signature shape
    and the ``state.is_ambient_enabled`` gate; everything else lives here.
    """

    def __init__(self, engine):
        self._engine = engine

    def run(self, user_message: str, *,
            session_id: str | None = None) -> str | None:
        """Return the prompt-injectable context string, or ``None`` when
        the pipeline declined to inject (empty index, no hits, threshold
        not cleared, or rerank rejected)."""
        engine = self._engine
        if not engine.has_embeddings():
            return None

        hits = self._search(user_message, session_id)
        if not hits:
            return None
        parents = retrieval.chunks_to_parents(
            engine, hits, top=AMBIENT_RERANK_POOL,
        )
        if not parents:
            return None

        # Local-only rerank — never Cohere on the per-turn path; an HTTP
        # round-trip would defeat the cheap-injection premise.
        parents = rerank.rerank_local(
            user_message, parents, top_k=AMBIENT_TOP_PARENTS,
        )
        if not parents:
            return None

        # Threshold applies to the post-rerank score. Identity fallback (no
        # cross-encoder) keeps `rerank_score=None`, so `effective_score`
        # falls back to RRF. RRF scores are tiny (~0.03-0.06), so the
        # 0.25 default effectively gates ambient OFF when the cross-encoder
        # is unavailable. Intentional — no reranker means low confidence.
        if parents[0].effective_score < AMBIENT_SCORE_THRESHOLD:
            return None

        context = formatting.format_context(parents, token_cap=AMBIENT_TOKEN_CAP)
        return context or None

    def _search(self, user_message: str, session_id: str | None) -> list[Hit]:
        """Hybrid search for the ambient path. When convo memory is enabled,
        mix the current query embedding with the previous turns' embeddings
        before dense scoring; BM25 always operates on the literal current
        message so lexical search isn't contaminated."""
        engine = self._engine
        if convo.is_enabled() and session_id:
            cur = engine.encode_query(user_message)
            if cur is None:
                return engine.hybrid_search(user_message, k_pool=30)
            ring = convo.get_ring(session_id)
            mixed = convo.mix_with_history(cur, ring)
            convo.push(session_id, cur)
            return engine.hybrid_search(user_message, qvec=mixed, k_pool=30)
        return engine.hybrid_search(user_message, k_pool=30)
