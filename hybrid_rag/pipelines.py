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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from operator import itemgetter

from . import convo, crag, expansion, formatting, rerank, retrieval
from .config import RAG_SEARCH_CHUNK_POOL, RAG_SEARCH_PARENT_POOL
from .models import Hit, ParentResult

# Cap on the worker threads used to fan out variant searches in
# `ExplicitPipeline._one_pass`. Past 5 the marginal benefit fades — the
# default expansion produces 5 variants (q + 3 paraphrases + HyDE), and
# the per-variant work is dominated by a numpy matmul that already uses
# multiple BLAS threads under the covers.
_VARIANT_WORKER_CAP = 5

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

        # One LLM call instead of the historical two (judge then
        # reformulate). The merged shape lets the model either declare
        # "sufficient" and stop, or emit the rewrite in the same response.
        # ~500 ms saved on the CRAG-enabled path; behavior unchanged on
        # failure paths (sufficient=True, no retry).
        verdict = crag.judge_and_reformulate(query, first.parents)
        if verdict.get("sufficient", True):
            return first

        reason = verdict.get("reason", "")
        new_q = verdict.get("rewritten_query")
        if not (new_q and new_q.strip() and new_q.strip() != query.strip()):
            return ExplicitResult(
                parents=first.parents,
                expansions_used=first.expansions_used,
                crag_reformulated_query=None,
                crag_reason=reason,
            )

        retry = self._one_pass(new_q, k)
        # If the rewrite returned nothing, prefer the first-pass parents over
        # silently handing the caller an empty result. The judge declared
        # them insufficient, but "imperfect" still beats "empty" — and the
        # caller can still see the verdict via ``crag_reason``.
        kept_parents = retry.parents if retry.parents else first.parents
        return ExplicitResult(
            parents=kept_parents,
            expansions_used=retry.expansions_used,
            crag_reformulated_query=new_q,
            crag_reason=reason,
        )

    def _search_variants(self, variants: list[str]) -> list[list[int]]:
        """Run `hybrid_search_chunk_ids` for each variant. The variants are
        independent — different query strings, same engine state — so we
        fan out across a thread pool. numpy releases the GIL during the
        cosine matmul and BM25's inner ops, so this gives near-linear
        speedup up to the BLAS-thread cap.

        Pre-dedupe by literal query string before fanning out. Expansion
        already dedupes case-insensitively, so in steady state every
        variant is unique and this short-circuit is a no-op. The point is
        defense: a future tweak to `expand_query` (or a model that emits
        byte-identical paraphrases) doesn't quietly cost an extra search.
        Token-equivalent strings are NOT deduped here — they share BM25
        rankings but diverge on the dense side.

        The single-variant case skips the pool overhead (which is ~5 ms
        of teardown when nothing is queued).
        """
        eng = self._engine
        # dict.fromkeys preserves first-seen order and dedupes by exact key.
        unique = list(dict.fromkeys(variants))

        def _search(v):
            return eng.hybrid_search_chunk_ids(v, k_pool=RAG_SEARCH_CHUNK_POOL)

        if len(unique) <= 1:
            by_query = {v: _search(v) for v in unique}
        else:
            with ThreadPoolExecutor(
                max_workers=min(len(unique), _VARIANT_WORKER_CAP),
            ) as ex:
                results = list(ex.map(_search, unique))
            by_query = dict(zip(unique, results))

        return [by_query[v] for v in variants]

    def _one_pass(self, query: str, k: int) -> ExplicitResult:
        """Single expansion → hybrid → fusion → rollup → rerank pass."""
        eng = self._engine
        store = eng.store

        variants = expansion.expand_query(query)
        # `hybrid_search_chunk_ids` skips per-variant parent resolution —
        # we only need the chunk rankings here for the second-level RRF.
        # The parent IDs get resolved once below for the fused top list.
        per_variant = self._search_variants(variants)

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

        # Rerank-pool stage gets truncated text — most of these parents
        # won't survive the rerank cut, so reading their full bodies from
        # SQLite is wasted I/O. The top-k that DO survive get rehydrated
        # with full text below.
        parents = retrieval.chunks_to_parents(
            eng, materialized, top=RAG_SEARCH_PARENT_POOL,
            text_cap=rerank.DEFAULT_RERANK_TEXT_CHARS,
        )
        reranked = rerank.rerank(query, parents, top_k=k)
        reranked = retrieval.rehydrate_parent_text(eng, reranked)
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
        # Rerank-pool fetch with the same tight text_cap the cross-encoder
        # will consume below — no point reading more than the scorer can
        # use. Survivors get rehydrated to full text before formatting so
        # the injected context isn't artificially short.
        parents = retrieval.chunks_to_parents(
            engine, hits, top=AMBIENT_RERANK_POOL,
            text_cap=rerank.AMBIENT_RERANK_TEXT_CHARS,
        )
        if not parents:
            return None

        # Local-only rerank — never Cohere on the per-turn path; an HTTP
        # round-trip would defeat the cheap-injection premise. The
        # ambient text cap is tighter than `rag_search`'s so the
        # per-turn cross-encoder pass stays inside the latency budget.
        parents = rerank.rerank_local(
            user_message, parents, top_k=AMBIENT_TOP_PARENTS,
            text_cap=rerank.AMBIENT_RERANK_TEXT_CHARS,
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

        # Rehydrate full text now that we know which parents survived.
        # `format_context` packs into AMBIENT_TOKEN_CAP — without this,
        # the injected blocks would only carry the truncated rerank text.
        parents = retrieval.rehydrate_parent_text(engine, parents)
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
