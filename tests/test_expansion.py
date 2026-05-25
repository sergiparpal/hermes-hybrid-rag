import sys

import pytest

from hybrid_rag.expansion import expand_query


def test_fallback_when_no_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert expand_query("hello world") == ["hello world"]


def test_fallback_when_anthropic_missing(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setitem(sys.modules, "anthropic", None)
    # importing `None` raises TypeError; expansion treats any import failure as
    # the package being unavailable.
    assert expand_query("hello world") == ["hello world"]


def test_happy_path_returns_query_paraphrases_and_hyde(mock_anthropic):
    out = expand_query("hello world")
    assert out[0] == "hello world"
    assert "a" in out
    assert "b" in out
    assert "c" in out
    assert out[-1] == "h"


def test_strips_markdown_code_fences(mock_anthropic):
    mock_anthropic.Anthropic().messages.create  # warm
    # Build a fresh client that returns fenced JSON
    import types
    mock_anthropic.Anthropic = lambda *a, **kw: types.SimpleNamespace(
        messages=types.SimpleNamespace(create=lambda **kw: types.SimpleNamespace(
            content=[types.SimpleNamespace(text='```json\n{"paraphrases": ["x"], "hyde": "y"}\n```')]
        ))
    )
    out = expand_query("hello world")
    assert "x" in out
    assert "y" in out


def test_falls_back_on_sdk_exception(mock_anthropic):
    import types
    mock_anthropic.Anthropic = lambda *a, **kw: types.SimpleNamespace(
        messages=types.SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    )
    assert expand_query("hello world") == ["hello world"]


def test_falls_back_on_unparseable_json(mock_anthropic):
    import types
    mock_anthropic.Anthropic = lambda *a, **kw: types.SimpleNamespace(
        messages=types.SimpleNamespace(create=lambda **kw: types.SimpleNamespace(
            content=[types.SimpleNamespace(text="not json at all")]
        ))
    )
    assert expand_query("hello world") == ["hello world"]


def test_empty_query_returns_empty_list():
    # Both empty and whitespace-only collapse to "no expansion" — a
    # whitespace string would otherwise feed noise into a hybrid_search
    # variant if a future caller skipped the tool layer's validation.
    assert expand_query("") == []
    assert expand_query("   ") == []


def test_paraphrases_dedupe_against_query_and_each_other(mock_anthropic):
    import types
    mock_anthropic.Anthropic = lambda *a, **kw: types.SimpleNamespace(
        messages=types.SimpleNamespace(create=lambda **kw: types.SimpleNamespace(
            content=[types.SimpleNamespace(
                text='{"paraphrases": ["hello world", "FOO ", "foo", "bar"], "hyde": "h"}'
            )]
        ))
    )
    out = expand_query("hello world")
    # original query stays; "hello world" duplicate dropped; "FOO " and "foo"
    # collapse to one entry; "bar" comes through; hyde appended.
    assert out[0] == "hello world"
    lower = [s.lower() for s in out]
    assert lower.count("hello world") == 1
    assert lower.count("foo") == 1
    assert "bar" in out
    assert out[-1] == "h"


def test_hyde_dedupes_against_query_and_paraphrases(mock_anthropic):
    """If the model echoes the query into `hyde`, it must not survive the
    dedup — wasting a hybrid_search variant on an identical string only
    hurts latency."""
    import types
    mock_anthropic.Anthropic = lambda *a, **kw: types.SimpleNamespace(
        messages=types.SimpleNamespace(create=lambda **kw: types.SimpleNamespace(
            content=[types.SimpleNamespace(
                text='{"paraphrases": ["alt"], "hyde": "hello world"}'
            )]
        ))
    )
    out = expand_query("hello world")
    # original + 1 paraphrase, with the duplicated hyde dropped.
    assert out == ["hello world", "alt"]


def test_anthropic_client_is_cached_across_calls(mock_anthropic):
    """L9: the SDK client is reused; one constructor call covers many queries."""
    constructions = {"n": 0}
    real_client = mock_anthropic.Anthropic

    def counting_ctor(*a, **kw):
        constructions["n"] += 1
        return real_client(*a, **kw)

    mock_anthropic.Anthropic = counting_ctor
    expand_query("first")
    expand_query("second")
    expand_query("third")
    assert constructions["n"] == 1
