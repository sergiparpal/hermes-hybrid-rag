"""Phase 4 — CRAG-lite (critique + retry on the explicit path).

Covers the four paths the spec calls out:
- sufficient first try → no retry
- insufficient → reformulate → retry → return
- Anthropic unavailable → skip CRAG entirely
- CRAG disabled (default) → skip CRAG entirely
- ambient path is NEVER affected by HERMES_RAG_CRAG
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

import advanced_rag.crag as crag_mod
import advanced_rag.hooks as hooks_mod
import advanced_rag.state as state_mod
from advanced_rag import convo
from advanced_rag.engine import RAGEngine, set_engine_for_tests
from advanced_rag.hooks import ambient_pre_llm_call
from advanced_rag.indexing import index_path
from advanced_rag.storage import Store
from advanced_rag.tools import tool_rag_search

FIXTURES = Path(__file__).parent / "fixtures" / "docs"


@pytest.fixture(autouse=True)
def _isolate():
    crag_mod._reset_client_for_tests()
    state_mod.invalidate_cache_for_tests()
    convo.reset_for_tests()
    yield
    crag_mod._reset_client_for_tests()
    state_mod.invalidate_cache_for_tests()
    convo.reset_for_tests()


@pytest.fixture
def warmed_engine(tmp_data_dir, tmp_path, stub_embedder, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    for n in ("alpha.md", "beta.md", "gamma.txt"):
        (docs / n).write_text((FIXTURES / n).read_text())
    store = Store()
    index_path(docs, store=store, embedder=stub_embedder)
    eng = RAGEngine(store=store, embedder=stub_embedder)
    eng._ensure_loaded()
    set_engine_for_tests(eng)
    # Don't let query expansion or Cohere kick in during search tests.
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    yield eng
    set_engine_for_tests(None)


# --- Anthropic mock that scripts judge + reformulate ---

def _install_scripted_anthropic(monkeypatch, *, responses: list[str]):
    """Install a fake `anthropic` module where each successive
    `messages.create` returns the next scripted string."""
    mod = types.ModuleType("anthropic")
    state = {"i": 0, "calls": []}

    class _Msg:
        def __init__(self, t):
            self.content = [types.SimpleNamespace(text=t)]

    class _Messages:
        def create(self, **kwargs):
            state["calls"].append(kwargs)
            t = responses[state["i"]] if state["i"] < len(responses) else responses[-1]
            state["i"] += 1
            return _Msg(t)

    class _Client:
        def __init__(self, *a, **kw):
            self.messages = _Messages()

    mod.Anthropic = _Client
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    # Force fresh client constructions in both expansion and crag modules.
    crag_mod._reset_client_for_tests()
    from advanced_rag import expansion as _exp
    _exp._reset_anthropic_client_for_tests()
    return state


# --- env toggle ---

def test_crag_off_by_default(monkeypatch):
    monkeypatch.delenv("HERMES_RAG_CRAG", raising=False)
    assert crag_mod.is_enabled() is False


def test_crag_toggle_via_env(monkeypatch):
    monkeypatch.setenv("HERMES_RAG_CRAG", "1")
    assert crag_mod.is_enabled() is True


# --- isolated judge/reformulate behavior ---

def test_judge_returns_sufficient_when_no_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = crag_mod.judge_retrieval("q", [])
    assert out["sufficient"] is True


def test_judge_returns_sufficient_when_sdk_missing(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setitem(sys.modules, "anthropic", None)
    crag_mod._reset_client_for_tests()
    out = crag_mod.judge_retrieval("q", [])
    assert out["sufficient"] is True


def test_reformulate_returns_none_when_no_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert crag_mod.reformulate_query("q", [], "x") is None


def test_judge_parses_fenced_json(monkeypatch):
    _install_scripted_anthropic(monkeypatch, responses=[
        '```json\n{"sufficient": false, "reason": "missing X"}\n```'
    ])
    out = crag_mod.judge_retrieval("q", [])
    assert out["sufficient"] is False
    assert "missing X" in out["reason"]


def test_judge_swallows_sdk_error(monkeypatch):
    mod = types.ModuleType("anthropic")

    class _Messages:
        def create(self, **kw):
            raise RuntimeError("boom")

    class _Client:
        def __init__(self, *a, **kw):
            self.messages = _Messages()

    mod.Anthropic = _Client
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    crag_mod._reset_client_for_tests()
    out = crag_mod.judge_retrieval("q", [])
    assert out["sufficient"] is True


# --- integration: tool_rag_search ---

def test_crag_disabled_skips_critique(warmed_engine, monkeypatch):
    """Default-off path: no Anthropic call, response carries null CRAG fields."""
    monkeypatch.delenv("HERMES_RAG_CRAG", raising=False)
    state = _install_scripted_anthropic(monkeypatch, responses=[
        '{"sufficient": false, "reason": "unused"}',
        "unused reformulation",
    ])
    out = json.loads(tool_rag_search({"query": "cosmic rays"}))
    assert out["crag_reformulated_query"] is None
    assert out["crag_reason"] is None
    # CRAG should not have called Anthropic at all (expansion path uses the
    # ANTHROPIC_API_KEY env too, but that path is gated by expansion's own
    # logic — we only assert no *crag* call was made by checking the
    # response shape).


def test_crag_sufficient_no_retry(warmed_engine, monkeypatch):
    monkeypatch.setenv("HERMES_RAG_CRAG", "1")
    # ANTHROPIC_API_KEY is set by the scripted installer below.
    state = _install_scripted_anthropic(monkeypatch, responses=[
        # First call to /messages on the search path is query expansion —
        # return valid JSON so expansion can proceed.
        '{"paraphrases": [], "hyde": ""}',
        # Second call is CRAG judge — verdict: sufficient.
        '{"sufficient": true, "reason": "complete coverage"}',
    ])
    out = json.loads(tool_rag_search({"query": "cosmic rays"}))
    assert out["crag_reformulated_query"] is None
    assert out["crag_reason"] is None


def test_crag_insufficient_triggers_one_retry(warmed_engine, monkeypatch):
    monkeypatch.setenv("HERMES_RAG_CRAG", "1")
    state = _install_scripted_anthropic(monkeypatch, responses=[
        # 1st: expansion on original query.
        '{"paraphrases": [], "hyde": ""}',
        # 2nd: CRAG judge → insufficient.
        '{"sufficient": false, "reason": "no cosmic context"}',
        # 3rd: reformulation.
        "cosmic ray atmospheric showers",
        # 4th: expansion on reformulated query.
        '{"paraphrases": [], "hyde": ""}',
        # No second judge — hard cap one retry.
    ])
    out = json.loads(tool_rag_search({"query": "cosmic rays"}))
    assert out["crag_reformulated_query"] == "cosmic ray atmospheric showers"
    assert "no cosmic context" in (out["crag_reason"] or "")
    # Exactly one retry → exactly 4 Anthropic calls.
    assert len(state["calls"]) == 4


def test_crag_reformulation_failure_returns_first_pass(warmed_engine, monkeypatch):
    """If reformulation returns empty / fails, the caller falls back to the
    first-pass results — no retry."""
    monkeypatch.setenv("HERMES_RAG_CRAG", "1")
    _install_scripted_anthropic(monkeypatch, responses=[
        '{"paraphrases": [], "hyde": ""}',
        '{"sufficient": false, "reason": "thin"}',
        # Empty reformulation → None
        "",
    ])
    out = json.loads(tool_rag_search({"query": "cosmic rays"}))
    assert out["crag_reformulated_query"] is None
    # Reason still surfaces so the caller can log it.
    assert "thin" in (out["crag_reason"] or "")


def test_crag_no_anthropic_key_skips_silently(warmed_engine, monkeypatch):
    """HERMES_RAG_CRAG=1 but no ANTHROPIC_API_KEY → CRAG is a no-op."""
    monkeypatch.setenv("HERMES_RAG_CRAG", "1")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    crag_mod._reset_client_for_tests()
    out = json.loads(tool_rag_search({"query": "cosmic rays"}))
    assert out["crag_reformulated_query"] is None
    assert out["crag_reason"] is None


# --- ambient path is never affected ---

def test_ambient_path_never_invokes_crag(warmed_engine, monkeypatch,
                                         mock_cross_encoder):
    """HERMES_RAG_CRAG=1 must not change ambient behavior. The ambient hook
    must not call judge_retrieval / reformulate_query."""
    monkeypatch.setenv("HERMES_RAG_CRAG", "1")
    monkeypatch.setattr(hooks_mod, "AMBIENT_SCORE_THRESHOLD", 0.0)

    judge_calls = {"n": 0}
    reformulate_calls = {"n": 0}

    real_judge = crag_mod.judge_retrieval
    real_reformulate = crag_mod.reformulate_query

    def spy_judge(*a, **kw):
        judge_calls["n"] += 1
        return real_judge(*a, **kw)

    def spy_reformulate(*a, **kw):
        reformulate_calls["n"] += 1
        return real_reformulate(*a, **kw)

    monkeypatch.setattr(crag_mod, "judge_retrieval", spy_judge)
    monkeypatch.setattr(crag_mod, "reformulate_query", spy_reformulate)

    mock_cross_encoder._scores = [5.0] * 10
    out = ambient_pre_llm_call(
        session_id="s", user_message="cosmic rays from outer space",
        conversation_history=None, is_first_turn=True,
    )
    assert out is not None  # ambient still works
    assert judge_calls["n"] == 0
    assert reformulate_calls["n"] == 0
