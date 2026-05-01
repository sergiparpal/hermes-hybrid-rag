"""Hybrid retrieval primitives: tokenizer, RRF fusion, hybrid search,
chunk → parent rollup, and ambient-context formatting.

The same `_tokenize` is used at index time and at query time — keeping them in
one place is the only way to keep BM25 scoring honest.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .config import RRF_K

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric runs only. SAME tokenizer at index and query time."""
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


@dataclass
class Hit:
    chunk_id: int
    score: float
    parent_id: int


@dataclass
class ParentResult:
    parent_id: int
    title: str | None
    kind: str
    page_no: int | None
    text: str
    source_path: str
    score: float
    rerank_score: float | None = None


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


def _bm25_topk(engine, query_tokens: list[str], k: int) -> list[int]:
    if engine._bm25 is None or not query_tokens:
        return []
    scores = engine._bm25.get_scores(query_tokens)
    if len(scores) == 0:
        return []
    k = min(k, len(scores))
    if k <= 0:
        return []
    # argpartition is O(N), then sort the small head by exact score.
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [engine._chunk_ids[i] for i in idx if scores[i] > 0]


def _dense_topk(engine, query: str, k: int) -> list[int]:
    if engine._embeddings is None or engine._embeddings.shape[0] == 0:
        return []
    qvec = engine._embedder.encode([query])  # (1, dim), L2-normalized
    if qvec.shape[0] == 0:
        return []
    sims = engine._embeddings @ qvec[0]
    n = sims.shape[0]
    k = min(k, n)
    if k <= 0:
        return []
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return [engine._chunk_ids[i] for i in idx]


def hybrid_search(engine, query: str, k_pool: int = 30) -> list[Hit]:
    """BM25 + dense, fused with RRF. Returns top k_pool Hits with parent_id
    resolved from the SQLite store."""
    tokens = _tokenize(query)
    bm25_ranked = _bm25_topk(engine, tokens, k_pool * 2)
    dense_ranked = _dense_topk(engine, query, k_pool * 2)
    fused = rrf_fuse([bm25_ranked, dense_ranked])
    if not fused:
        return []
    top = sorted(fused.items(), key=lambda kv: -kv[1])[:k_pool]
    out: list[Hit] = []
    for cid, score in top:
        pid = engine._store.parent_id_for_chunk(cid)
        if pid is None:
            continue
        out.append(Hit(chunk_id=cid, score=score, parent_id=pid))
    return out


def chunks_to_parents(engine, hits: Iterable[Hit], top: int) -> list[ParentResult]:
    """MAX-rollup: a parent's score is the highest fused score across its
    matched chunks. (Avoids penalizing parents whose other children are
    unrelated, which SUM/MEAN would do.)"""
    by_parent: dict[int, float] = {}
    for h in hits:
        prev = by_parent.get(h.parent_id)
        if prev is None or h.score > prev:
            by_parent[h.parent_id] = h.score

    ranked = sorted(by_parent.items(), key=lambda kv: -kv[1])[:top]
    out: list[ParentResult] = []
    for pid, score in ranked:
        row = engine._store.get_parent(pid)
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


def format_context(parents: list[ParentResult], token_cap: int = 1500) -> str:
    """Pack parents as ## title / body sections, truncating by char-budget
    (~4 chars/token). Returns "" if nothing fits."""
    char_budget = token_cap * 4
    pieces: list[str] = []
    used = 0
    for p in parents:
        title = p.title or f"{p.kind} (parent {p.parent_id})"
        block = f"## {title}\n{p.text}\n"
        if used + len(block) > char_budget:
            remaining = char_budget - used
            if remaining > 200:
                truncated = block[: remaining - 1].rstrip() + "…"
                pieces.append(truncated)
            break
        pieces.append(block)
        used += len(block) + 1
    return "\n".join(pieces).strip()
