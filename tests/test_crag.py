"""CRAG-lite (critique + retry on the explicit path).

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

import hybrid_rag.crag as crag_mod
import hybrid_rag.hooks as hooks_mod
import hybrid_rag.pipelines as pipelines_mod
import hybrid_rag.state as state_mod
from hybrid_rag import _anthropic, convo
from hybrid_rag.engine import RAGEngine, reset_for_tests, set_engine_for_tests
from hybrid_rag.hooks import ambient_pre_llm_call
from hybrid_rag.indexing import index_path
from hybrid_rag.storage import Store
from hybrid_rag.tools import tool_rag_search

FIXTURES = Path(__file__).parent / "fixtures" / "docs"


@pytest.fixture(autouse=True)
def _isolate():
    _anthropic.reset_for_tests()
    state_mod.reset_for_tests()
    convo.reset_for_tests()
    yield
    _anthropic.reset_for_tests()
    state_mod.reset_for_tests()
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
    reset_for_tests()


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
    # Force a fresh client construction (expansion + crag now share one).
    _anthropic.reset_for_tests()
    return state


# --- env toggle ---

def test_crag_off_by_default(monkeypatch):
    monkeypatch.delenv("HERMES_RAG_CRAG", raising=False)
    assert crag_mod.is_enabled() is False


def test_crag_toggle_via_env(monkeypatch):
    monkeypatch.setenv("HERMES_RAG_CRAG", "1")
    assert crag_mod.is_enabled() is True


# --- isolated judge/reformulate behavior ---

def test_judge_and_reformulate_normalizes_whitespace(monkeypatch):
    """Multi-line LLM output collapses to a single line of tokens — keeps the
    rewritten query honest before it's fed back to BM25 / dense retrieval."""
    _install_scripted_anthropic(monkeypatch, responses=[
        '{"sufficient": false, "reason": "r", '
        '"rewritten_query": "\\n  rewritten\\tquery\\n  with\\nlines\\n"}',
    ])
    out = crag_mod.judge_and_reformulate("orig", [])
    assert out["rewritten_query"] == "rewritten query with lines"


def test_judge_and_reformulate_rejects_pathological_length(monkeypatch):
    """A model that returns a wall of text instead of a query is rejected —
    defense-in-depth bound on what we'll feed back into retrieval."""
    long_q = "word " * 200
    _install_scripted_anthropic(monkeypatch, responses=[
        '{"sufficient": false, "reason": "r", '
        f'"rewritten_query": "{long_q.strip()}"}}',
    ])
    out = crag_mod.judge_and_reformulate("orig", [])
    assert out["rewritten_query"] is None


def test_judge_and_reformulate_parses_fenced_json(monkeypatch):
    _install_scripted_anthropic(monkeypatch, responses=[
        '```json\n{"sufficient": false, "reason": "missing X", '
        '"rewritten_query": "better q"}\n```'
    ])
    out = crag_mod.judge_and_reformulate("q", [])
    assert out["sufficient"] is False
    assert "missing X" in out["reason"]
    assert out["rewritten_query"] == "better q"


def test_judge_and_reformulate_sufficient_path(monkeypatch):
    """Sufficient verdict: no rewrite emitted, one LLM call total."""
    state = _install_scripted_anthropic(monkeypatch, responses=[
        '{"sufficient": true, "reason": "covered"}',
    ])
    out = crag_mod.judge_and_reformulate("q", [])
    assert out["sufficient"] is True
    assert out["rewritten_query"] is None
    assert len(state["calls"]) == 1


def test_judge_and_reformulate_insufficient_path(monkeypatch):
    """Insufficient verdict: rewrite rides on the same response."""
    state = _install_scripted_anthropic(monkeypatch, responses=[
        '{"sufficient": false, "reason": "missing", '
        '"rewritten_query": "better phrasing"}',
    ])
    out = crag_mod.judge_and_reformulate("q", [])
    assert out["sufficient"] is False
    assert out["rewritten_query"] == "better phrasing"
    assert "missing" in out["reason"]
    assert len(state["calls"]) == 1


def test_judge_and_reformulate_no_api_key_is_noop(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _anthropic.reset_for_tests()
    out = crag_mod.judge_and_reformulate("q", [])
    # Fail-open: caller treats this as no retry.
    assert out["sufficient"] is True
    assert out["rewritten_query"] is None


def test_judge_and_reformulate_swallows_sdk_error(monkeypatch):
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
    _anthropic.reset_for_tests()
    out = crag_mod.judge_and_reformulate("q", [])
    # Fail-open: caller treats this as no retry.
    assert out["sufficient"] is True
    assert out["rewritten_query"] is None


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
        # Second call is the merged CRAG judge+reformulate — sufficient
        # response stops here (no rewrite needed).
        '{"sufficient": true, "reason": "complete coverage"}',
    ])
    out = json.loads(tool_rag_search({"query": "cosmic rays"}))
    assert out["crag_reformulated_query"] is None
    assert out["crag_reason"] is None
    assert len(state["calls"]) == 2


def test_crag_insufficient_triggers_one_retry(warmed_engine, monkeypatch):
    monkeypatch.setenv("HERMES_RAG_CRAG", "1")
    state = _install_scripted_anthropic(monkeypatch, responses=[
        # 1st: expansion on original query.
        '{"paraphrases": [], "hyde": ""}',
        # 2nd: merged judge+reformulate. The rewritten query rides in the
        # same response — A2 collapsed two LLM calls into one.
        '{"sufficient": false, "reason": "no cosmic context", '
        '"rewritten_query": "cosmic ray atmospheric showers"}',
        # 3rd: expansion on reformulated query.
        '{"paraphrases": [], "hyde": ""}',
        # No second judge — hard cap one retry.
    ])
    out = json.loads(tool_rag_search({"query": "cosmic rays"}))
    assert out["crag_reformulated_query"] == "cosmic ray atmospheric showers"
    assert "no cosmic context" in (out["crag_reason"] or "")
    # One judge + one reformulate previously cost 4 calls (expand, judge,
    # reformulate, expand-retry); the merged shape brings it down to 3.
    assert len(state["calls"]) == 3


def test_crag_reformulation_failure_returns_first_pass(warmed_engine, monkeypatch):
    """If the merged call returns insufficient but no usable rewrite, the
    caller falls back to the first-pass results — no retry."""
    monkeypatch.setenv("HERMES_RAG_CRAG", "1")
    _install_scripted_anthropic(monkeypatch, responses=[
        '{"paraphrases": [], "hyde": ""}',
        # Empty rewritten_query → treated as "no rewrite available".
        '{"sufficient": false, "reason": "thin", "rewritten_query": ""}',
    ])
    out = json.loads(tool_rag_search({"query": "cosmic rays"}))
    assert out["crag_reformulated_query"] is None
    # Reason still surfaces so the caller can log it.
    assert "thin" in (out["crag_reason"] or "")


def test_crag_no_anthropic_key_skips_silently(warmed_engine, monkeypatch):
    """HERMES_RAG_CRAG=1 but no ANTHROPIC_API_KEY → CRAG is a no-op."""
    monkeypatch.setenv("HERMES_RAG_CRAG", "1")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _anthropic.reset_for_tests()
    out = json.loads(tool_rag_search({"query": "cosmic rays"}))
    assert out["crag_reformulated_query"] is None
    assert out["crag_reason"] is None


# --- ambient path is never affected ---

def test_ambient_path_never_invokes_crag(warmed_engine, monkeypatch,
                                         mock_cross_encoder):
    """HERMES_RAG_CRAG=1 must not change ambient behavior. The ambient hook
    must not call ``judge_and_reformulate``."""
    monkeypatch.setenv("HERMES_RAG_CRAG", "1")
    monkeypatch.setattr(pipelines_mod, "AMBIENT_SCORE_THRESHOLD", 0.0)

    merged_calls = {"n": 0}
    real_merged = crag_mod.judge_and_reformulate

    def spy_merged(*a, **kw):
        merged_calls["n"] += 1
        return real_merged(*a, **kw)

    monkeypatch.setattr(crag_mod, "judge_and_reformulate", spy_merged)

    mock_cross_encoder._scores = [5.0] * 10
    out = ambient_pre_llm_call(
        session_id="s", user_message="cosmic rays from outer space",
        conversation_history=None, is_first_turn=True,
    )
    assert out is not None  # ambient still works
    assert merged_calls["n"] == 0
