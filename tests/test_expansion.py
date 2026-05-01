import sys

import pytest

from advanced_rag.expansion import expand_query


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
    assert expand_query("") == []
    assert expand_query("   ") == ["   "]
