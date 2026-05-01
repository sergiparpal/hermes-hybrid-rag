import json

import hierarchical_rag.state as state_mod
from hierarchical_rag.slash import slash_rag


def setup_function(_):
    state_mod.invalidate_cache_for_tests()


def test_empty_returns_status_and_help(tmp_data_dir):
    out = slash_rag("")
    assert "Ambient RAG is on" in out
    assert "/rag on" in out and "/rag off" in out and "/rag stats" in out


def test_on_enables(tmp_data_dir):
    state_mod.set_ambient(False)
    state_mod.invalidate_cache_for_tests()
    out = slash_rag("on")
    assert out == "Ambient RAG: on"
    assert state_mod.is_ambient_enabled() is True


def test_off_disables(tmp_data_dir):
    state_mod.set_ambient(True)
    state_mod.invalidate_cache_for_tests()
    out = slash_rag("off")
    assert out == "Ambient RAG: off"
    assert state_mod.is_ambient_enabled() is False


def test_stats_returns_json(tmp_data_dir):
    out = slash_rag("stats")
    payload = json.loads(out)
    assert "files" in payload
    assert "parents" in payload
    assert "chunks" in payload


def test_unknown_returns_help(tmp_data_dir):
    out = slash_rag("bogus")
    assert "/rag on" in out


def test_extra_kwargs_ignored(tmp_data_dir):
    """Hermes may pass session_id or other kwargs; slash_rag must accept them."""
    out = slash_rag("", session_id="abc", platform="x")
    assert "Ambient RAG" in out
