"""Hybrid retrieval primitives: tokenizer, RRF fusion, hybrid search, and
chunk → parent rollup.

The same `_tokenize` is used at index time and at query time — keeping them in
one place is the only way to keep BM25 scoring honest. Presentation
(``<retrieved_document>`` wrapping, sanitization) lives in ``formatting.py``.
"""
from __future__ import annotations

import heapq
import logging
import re
from collections import defaultdict
from operator import itemgetter
from typing import Iterable

import numpy as np

from .config import RRF_K
from .models import Hit, ParentResult

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric runs only. SAME tokenizer at index and query time."""
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def rrf_fuse(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion. For each ranking (a list of chunk_ids in order):
        score[id] += 1 / (k + rank + 1)
    where rank is 0-indexed (so rank+1 is the 1-indexed rank).
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            fused[item_id] = fused.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return fused


def _top_k_descending(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the top-`k` entries in `scores`, descending. ``O(N)``
    partial selection then a small sort over the head — the dense and BM25
    top-k loops both ride this."""
    n = scores.shape[0]
    k = min(k, n)
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


def _bm25_topk(engine, query_tokens: list[str], k: int) -> list[int]:
    if engine.bm25 is None or not query_tokens:
        return []
    scores = engine.bm25.get_scores(query_tokens)
    if len(scores) == 0:
        return []
    idx = _top_k_descending(scores, k)
    return [engine.chunk_ids[i] for i in idx if scores[i] > 0]


def _dense_topk(engine, query: str, qvec: np.ndarray | None, k: int) -> list[int]:
    """Dense top-k. Encodes ``query`` itself when ``qvec`` is None; otherwise
    uses the caller-supplied vector (the ambient path under
    ``HERMES_RAG_AMBIENT_CONVO_MEMORY=1`` mixes prior-turn embeddings before
    scoring).

    Validation already refused to load a dim-mismatched index, so a mismatch
    here means live state was corrupted mid-process. We log an error and
    return ``[]`` rather than crash — the caller proceeds with BM25 only.
    """
    if engine.embeddings is None or engine.embeddings.shape[0] == 0:
        return []
    if qvec is None:
        # Defensive: an empty/whitespace query encodes to an arbitrary
        # vector whose top matches would be noise. Production callers
        # (tool layer, ambient hook) already reject these upstream, but
        # the engine method is also reachable directly from tests and
        # third-party code.
        if not query or not query.strip():
            return []
        batch = engine.embedder.encode([query])  # (1, dim), L2-normalized
        if batch.shape[0] == 0:
            return []
        qvec = batch[0]
    if qvec.size == 0 or qvec.ndim != 1:
        return []
    if qvec.shape[0] != engine.embeddings.shape[1]:
        log.error(
            "dense top-k aborted: query vector dim %d != index dim %d. "
            "Engine consistency check should have caught this — investigate.",
            qvec.shape[0], engine.embeddings.shape[1],
        )
        return []
    sims = engine.embeddings @ qvec
    idx = _top_k_descending(sims, k)
    return [engine.chunk_ids[i] for i in idx]


def hybrid_search(
    engine, query: str, *, qvec: np.ndarray | None = None, k_pool: int = 30,
) -> list[Hit]:
    """BM25 + dense, fused with RRF. Returns top ``k_pool`` Hits with
    ``parent_id`` resolved from the SQLite store.

    When ``qvec`` is provided, it replaces the embedder call on the dense
    side (the ambient convo-memory path uses this to mix prior-turn vectors
    in). BM25 always operates on the literal ``query`` regardless.

    Note: this function is engine-state oriented — pass anything that
    exposes ``.bm25 / .embeddings / .chunk_ids / .embedder / .store``. Tests
    use a ``FakeEngine`` for that reason. Production callers should prefer
    ``RAGEngine.hybrid_search(...)`` which folds in the lazy-load.
    """
    tokens = _tokenize(query)
    bm25_ranked = _bm25_topk(engine, tokens, k_pool * 2)
    dense_ranked = _dense_topk(engine, query, qvec, k_pool * 2)
    return _materialize_hits(engine, bm25_ranked, dense_ranked, k_pool)


def hybrid_search_chunk_ids(
    engine, query: str, *, qvec: np.ndarray | None = None, k_pool: int = 30,
) -> list[int]:
    """BM25 + dense fused into chunk-id rankings, no parent resolution.

    The explicit `rag_search` pipeline calls this **once per variant**
    (q + paraphrases + HyDE) and then re-fuses the rankings into a single
    chunk list. Parent IDs are only needed for the FINAL fused result,
    so resolving them per variant — as the regular ``hybrid_search`` does
    — is wasted SQL work. This function skips that step; callers do one
    batched ``parent_ids_for_chunks`` after the second-level RRF.
    """
    tokens = _tokenize(query)
    bm25_ranked = _bm25_topk(engine, tokens, k_pool * 2)
    dense_ranked = _dense_topk(engine, query, qvec, k_pool * 2)
    fused = rrf_fuse([bm25_ranked, dense_ranked])
    if not fused:
        return []
    top = heapq.nlargest(k_pool, fused.items(), key=itemgetter(1))
    return [cid for cid, _ in top]


def _materialize_hits(
    engine, bm25_ranked: list[int], dense_ranked: list[int], k_pool: int,
) -> list[Hit]:
    fused = rrf_fuse([bm25_ranked, dense_ranked])
    if not fused:
        return []
    # heapq.nlargest is O(N log k); full sort is O(N log N). Cheap improvement
    # given `fused` can hold thousands of (chunk_id, score) pairs.
    top = heapq.nlargest(k_pool, fused.items(), key=itemgetter(1))
    # One SQL roundtrip for the whole batch instead of N+1 per-chunk lookups.
    parent_by_chunk = engine.store.parent_ids_for_chunks([cid for cid, _ in top])
    out: list[Hit] = []
    for cid, score in top:
        pid = parent_by_chunk.get(cid)
        if pid is None:
            continue
        out.append(Hit(chunk_id=cid, score=score, parent_id=pid))
    return out


def chunks_to_parents(
    engine, hits: Iterable[Hit], top: int,
    *, text_cap: int | None = None,
) -> list[ParentResult]:
    """MAX-rollup: a parent's score is the highest fused score across its
    matched chunks. (Avoids penalizing parents whose other children are
    unrelated, which SUM/MEAN would do.)

    ``text_cap`` truncates the per-parent ``text`` field at the SQL layer.
    Pipelines use this for the rerank-pool stage — most of those parents
    won't survive the rerank cut, so fetching their full 8000-char body
    is wasted I/O. The survivors are re-fetched with full text before
    being returned to the caller — see ``rehydrate_parent_text``.
    """
    by_parent: dict[int, float] = defaultdict(lambda: float("-inf"))
    for h in hits:
        if h.score > by_parent[h.parent_id]:
            by_parent[h.parent_id] = h.score

    ranked = heapq.nlargest(top, by_parent.items(), key=itemgetter(1))
    # Single batched fetch for all top parent rows.
    parent_rows = engine.store.get_parents(
        [pid for pid, _ in ranked], text_cap=text_cap,
    )
    out: list[ParentResult] = []
    for pid, score in ranked:
        row = parent_rows.get(pid)
        if row is None:
            continue
        out.append(ParentResult(
            parent_id=pid,
            title=row.get("title"),
            kind=row["kind"],
            page_no=row.get("page_no"),
            text=row["text"],
            source_path=row["source_path"],
            score=float(score),
        ))
    return out


def rehydrate_parent_text(engine, parents: list[ParentResult]) -> list[ParentResult]:
    """Replace each parent's (possibly truncated) text with the full
    SQLite copy. Used after rerank when the pool was fetched with a
    `text_cap` — the survivors should reach the caller with full bodies.

    Returns fresh ParentResult instances (the input list and its
    elements are not mutated)."""
    if not parents:
        return []
    full = engine.store.get_parents([p.parent_id for p in parents])
    out: list[ParentResult] = []
    for p in parents:
        row = full.get(p.parent_id)
        if row is None or "text" not in row:
            # Parent vanished between the two fetches (race with a concurrent
            # delete). Keep the truncated version rather than dropping it.
            out.append(p)
            continue
        out.append(ParentResult(
            parent_id=p.parent_id,
            title=p.title,
            kind=p.kind,
            page_no=p.page_no,
            text=row["text"],
            source_path=p.source_path,
            score=p.score,
            rerank_score=p.rerank_score,
        ))
    return out


