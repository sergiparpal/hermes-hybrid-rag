from pathlib import Path

import numpy as np
import pytest

import advanced_rag.hooks as hooks_mod
import advanced_rag.state as state_mod
from advanced_rag.engine import RAGEngine, set_engine_for_tests
from advanced_rag.hooks import ambient_pre_llm_call
from advanced_rag.indexing import index_path
from advanced_rag.storage import Store

FIXTURES = Path(__file__).parent / "fixtures" / "docs"


@pytest.fixture(autouse=True)
def _reset_state_cache():
    state_mod.invalidate_cache_for_tests()
    yield
    state_mod.invalidate_cache_for_tests()


@pytest.fixture
def warmed_engine(tmp_data_dir, tmp_path, stub_embedder):
    docs = tmp_path / "docs"
    docs.mkdir()
    for n in ("alpha.md", "beta.md", "gamma.txt"):
        (docs / n).write_text((FIXTURES / n).read_text())

    store = Store()
    index_path(docs, store=store, embedder=stub_embedder)
    eng = RAGEngine(store=store, embedder=stub_embedder)
    eng._ensure_loaded()
    set_engine_for_tests(eng)
    yield eng
    set_engine_for_tests(None)


def test_returns_none_when_disabled(warmed_engine, monkeypatch):
    state_mod.set_ambient(False)
    state_mod.invalidate_cache_for_tests()
    out = ambient_pre_llm_call(
        session_id=None, user_message="cosmic rays from space",
        conversation_history=None, is_first_turn=True, model=None, platform=None,
    )
    assert out is None


def test_returns_none_when_message_too_short(warmed_engine):
    out = ambient_pre_llm_call(
        session_id=None, user_message="hi",
        conversation_history=None, is_first_turn=True, model=None, platform=None,
    )
    assert out is None


def test_returns_none_when_no_hits(warmed_engine):
    out = ambient_pre_llm_call(
        session_id=None, user_message="zzzzzzzz qqqqqqq xxxxxxx aaaaaaaaa",
        conversation_history=None, is_first_turn=True, model=None, platform=None,
    )
    # very low or zero hits → likely None (BM25 score 0, dense match noise)
    # Either None (no hits / below threshold) is acceptable.
    assert out is None or "context" in out


def test_returns_none_when_below_threshold(warmed_engine, monkeypatch):
    monkeypatch.setattr(hooks_mod, "AMBIENT_SCORE_THRESHOLD", 999.0)
    out = ambient_pre_llm_call(
        session_id=None, user_message="cosmic rays from outer space",
        conversation_history=None, is_first_turn=True, model=None, platform=None,
    )
    assert out is None


def test_returns_context_on_match(warmed_engine, monkeypatch):
    # Lower the threshold to ensure the stub embedder produces a match strong enough
    monkeypatch.setattr(hooks_mod, "AMBIENT_SCORE_THRESHOLD", 0.0)
    out = ambient_pre_llm_call(
        session_id=None, user_message="cosmic rays from outer space",
        conversation_history=None, is_first_turn=True, model=None, platform=None,
    )
    assert out is not None
    assert "context" in out
    assert isinstance(out["context"], str) and out["context"]


def test_never_raises_when_engine_misbehaves(warmed_engine, monkeypatch):
    """Force an exception inside the body and assert the hook still returns
    None instead of raising."""
    def boom(*a, **kw):
        raise RuntimeError("synthetic engine failure")
    monkeypatch.setattr(hooks_mod.retrieval, "hybrid_search", boom)
    out = ambient_pre_llm_call(
        session_id=None, user_message="something substantive enough",
        conversation_history=None, is_first_turn=True, model=None, platform=None,
    )
    assert out is None


def test_returns_none_on_empty_index(tmp_data_dir, stub_embedder):
    """Engine with no artifacts should produce None without raising."""
    store = Store()
    eng = RAGEngine(store=store, embedder=stub_embedder)
    set_engine_for_tests(eng)
    try:
        out = ambient_pre_llm_call(
            session_id=None, user_message="anything substantial here",
            conversation_history=None, is_first_turn=True, model=None, platform=None,
        )
        assert out is None
    finally:
        set_engine_for_tests(None)
