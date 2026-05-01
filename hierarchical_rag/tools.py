"""Pure tool handlers. Each wraps its body in try/except and returns a JSON
string — never raises out to the caller."""
from __future__ import annotations

import json

from . import expansion, retrieval, rerank
from .engine import get_engine
from .retrieval import Hit
from .storage import Store


def _resolve(store=None, engine=None) -> tuple[Store, object]:
    eng = engine if engine is not None else get_engine()
    eng._ensure_loaded()
    st = store if store is not None else eng.store
    return st, eng


def _err(e: Exception) -> str:
    return json.dumps({"error": str(e), "type": type(e).__name__})


def tool_rag_search(args: dict, store=None, engine=None) -> str:
    """Full pipeline: expand → hybrid search per variant → second-level RRF on
    chunks → parent rollup (MAX) → rerank → top-k.

    Returns JSON: {"results": [{...}, ...], "expansions_used": int}.
    """
    try:
        if not isinstance(args, dict):
            return _err(TypeError(f"args must be dict, got {type(args).__name__}"))
        q = args.get("query")
        if not q or not isinstance(q, str) or not q.strip():
            return _err(ValueError("query is required and must be a non-empty string"))
        k = int(args.get("k", 5))

        st, eng = _resolve(store, engine)

        variants = expansion.expand_query(q)
        per_variant: list[list[int]] = []
        for v in variants:
            hits = retrieval.hybrid_search(eng, v, k_pool=30)
            per_variant.append([h.chunk_id for h in hits])

        fused = retrieval.rrf_fuse(per_variant)
        if not fused:
            return json.dumps({"results": [], "expansions_used": len(variants)})

        # top-30 chunks by fused score, then materialize Hit objs (parent_id from store)
        top_chunks = sorted(fused.items(), key=lambda kv: -kv[1])[:30]
        materialized: list[Hit] = []
        for cid, score in top_chunks:
            pid = st.parent_id_for_chunk(cid)
            if pid is None:
                continue
            materialized.append(Hit(chunk_id=cid, score=float(score), parent_id=pid))

        parents = retrieval.chunks_to_parents(eng, materialized, top=10)
        reranked = rerank.rerank(q, parents, top_k=k)

        results = []
        for p in reranked:
            results.append({
                "parent_id": p.parent_id,
                "title": p.title,
                "source_path": p.source_path,
                "score": p.score,
                "rerank_score": p.rerank_score,
                "kind": p.kind,
                "page_no": p.page_no,
                "text": p.text,
            })
        return json.dumps({"results": results, "expansions_used": len(variants)})
    except Exception as e:
        return _err(e)


def tool_rag_drill_down(args: dict, store=None, engine=None) -> str:
    """Return {"parent": {...}, "chunks": [...]} for a parent_id."""
    try:
        if not isinstance(args, dict):
            return _err(TypeError(f"args must be dict, got {type(args).__name__}"))
        pid_raw = args.get("parent_id")
        if pid_raw is None:
            return _err(ValueError("parent_id is required"))
        try:
            pid = int(pid_raw)
        except (TypeError, ValueError):
            return _err(ValueError(f"parent_id must be an integer, got {pid_raw!r}"))

        st = store if store is not None else (engine.store if engine is not None else get_engine().store)
        parent = st.get_parent(pid)
        if parent is None:
            return json.dumps({"error": f"parent_id {pid} not found",
                               "type": "NotFoundError"})
        chunks = st.chunks_for_parent(pid)
        return json.dumps({"parent": parent, "chunks": chunks})
    except Exception as e:
        return _err(e)


def tool_rag_list_sources(args: dict, store=None, engine=None) -> str:
    """Return {"sources": [{"path", "filetype", "indexed_at",
    "parent_count", "chunk_count"}, ...]}."""
    try:
        st = store if store is not None else (engine.store if engine is not None else get_engine().store)
        return json.dumps({"sources": st.list_sources()})
    except Exception as e:
        return _err(e)
