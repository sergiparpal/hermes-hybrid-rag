import sys

import pytest

import hybrid_rag.rerank as rerank_mod
from hybrid_rag.rerank import rerank
from hybrid_rag.models import ParentResult


def _parents():
    return [
        ParentResult(parent_id=1, title="A", kind="section", page_no=None,
                     text="alpha alpha", source_path="/x.md", score=0.9),
        ParentResult(parent_id=2, title="B", kind="section", page_no=None,
                     text="beta beta", source_path="/y.md", score=0.8),
        ParentResult(parent_id=3, title="C", kind="section", page_no=None,
                     text="gamma gamma", source_path="/z.md", score=0.7),
    ]


def test_cohere_path_used_when_key_present(mock_cohere):
    mock_cohere._scores = [0.1, 0.9, 0.5]  # B should rank first
    out = rerank("query", _parents(), top_k=2)
    assert [p.parent_id for p in out] == [2, 3]
    assert out[0].rerank_score == 0.9


def test_falls_back_to_local_when_cohere_raises(mock_cohere, mock_cross_encoder):
    mock_cohere._raise = RuntimeError("api down")
    mock_cross_encoder._scores = [-0.1, 5.0, 1.0]  # B should rank first locally
    out = rerank("query", _parents(), top_k=2)
    assert [p.parent_id for p in out] == [2, 3]
    assert out[0].rerank_score == 5.0


def test_falls_back_to_local_when_no_cohere_key(monkeypatch, mock_cross_encoder):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    mock_cross_encoder._scores = [3.0, 1.0, 2.0]
    out = rerank("query", _parents(), top_k=3)
    assert [p.parent_id for p in out] == [1, 3, 2]


def test_identity_fallback_when_everything_breaks(monkeypatch):
    # No COHERE_API_KEY, no cohere module installed, sentence_transformers
    # raises on CrossEncoder construction.
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "cohere", None)
    import types
    st = types.ModuleType("sentence_transformers")

    class Boom:
        def __init__(self, *a, **kw):
            raise RuntimeError("no model on disk")
    st.CrossEncoder = Boom
    monkeypatch.setitem(sys.modules, "sentence_transformers", st)
    rerank_mod._CROSS = None

    parents = _parents()
    out = rerank("query", parents, top_k=2)
    # identity fallback: same order, truncated to top_k
    assert [p.parent_id for p in out] == [1, 2]


def test_empty_input_returns_empty(mock_cohere):
    assert rerank("query", [], top_k=5) == []


def test_cohere_response_with_negative_idx_is_skipped(mock_cohere):
    """A malformed Cohere response carrying idx=-1 must not silently rank a
    wrong parent via Python's negative indexing — it must be skipped."""
    import types
    mock_cohere.Client = lambda *a, **kw: types.SimpleNamespace(
        rerank=lambda **kw: types.SimpleNamespace(results=[
            types.SimpleNamespace(index=-1, relevance_score=0.99),
            types.SimpleNamespace(index=1, relevance_score=0.5),
        ]),
    )
    mock_cohere.ClientV2 = mock_cohere.Client

    out = rerank("query", _parents(), top_k=3)
    assert [p.parent_id for p in out] == [2]
    assert out[0].rerank_score == 0.5


def test_falls_through_when_cohere_returns_empty(mock_cohere, mock_cross_encoder):
    """A Cohere call that succeeds but returns zero results must still
    trigger the local fallback — otherwise the user sees an empty answer
    despite having parents to rank."""
    # Force Cohere to return empty results (override the mock)
    import types
    mock_cohere._scores = []
    mock_cohere.Client = lambda *a, **kw: types.SimpleNamespace(
        rerank=lambda **kw: types.SimpleNamespace(results=[]),
    )
    mock_cohere.ClientV2 = mock_cohere.Client
    mock_cross_encoder._scores = [0.5, 5.0, 1.0]  # B should rank first locally

    out = rerank("query", _parents(), top_k=2)
    assert [p.parent_id for p in out] == [2, 3]
    assert out[0].rerank_score == 5.0
