from pathlib import Path

import pytest
import yaml

import hybrid_rag
from hybrid_rag import adapters


def test_register_wires_three_tools(fake_ctx):
    hybrid_rag.register(fake_ctx)
    names = [t["name"] for t in fake_ctx.tools]
    assert sorted(names) == ["rag_drill_down", "rag_list_sources", "rag_search"]
    for t in fake_ctx.tools:
        assert t["toolset"] == "rag"
        assert callable(t["handler"])
        assert "name" in t["schema"]


def test_register_wires_pre_llm_call_hook(fake_ctx):
    hybrid_rag.register(fake_ctx)
    events = [h["event"] for h in fake_ctx.hooks]
    assert "pre_llm_call" in events
    fn = next(h["fn"] for h in fake_ctx.hooks if h["event"] == "pre_llm_call")
    assert callable(fn)


def test_register_wires_on_session_start_hook(fake_ctx):
    hybrid_rag.register(fake_ctx)
    events = [h["event"] for h in fake_ctx.hooks]
    assert "on_session_start" in events
    # exactly two hooks: pre_llm_call and on_session_start
    assert sorted(events) == ["on_session_start", "pre_llm_call"]


def test_register_wires_cli_command(fake_ctx):
    hybrid_rag.register(fake_ctx)
    assert len(fake_ctx.cli_commands) == 1
    cmd = fake_ctx.cli_commands[0]
    assert cmd["name"] == "rag"
    assert callable(cmd["setup"])
    assert callable(cmd["handler"])


def test_register_wires_slash_command(fake_ctx):
    hybrid_rag.register(fake_ctx)
    assert len(fake_ctx.commands) == 1
    cmd = fake_ctx.commands[0]
    assert cmd["name"] == "rag"
    assert callable(cmd["handler"])


def test_register_wires_skill_with_path_object(fake_ctx):
    hybrid_rag.register(fake_ctx)
    assert len(fake_ctx.skills) == 1
    s = fake_ctx.skills[0]
    assert s["name"] == "rag-usage"
    # Hermes calls path.exists() at hermes_cli/plugins.py:577 — must be a Path,
    # not a string. A string would raise AttributeError.
    assert isinstance(s["path"], Path), \
        f"register_skill needs a pathlib.Path, got {type(s['path']).__name__}"
    assert s["path"].exists(), f"SKILL.md not found at {s['path']}"


def test_tool_wrapper_calls_underlying_function():
    calls = {"args": None}

    def fake_tool(args):
        calls["args"] = args
        return "ok"

    wrapper = adapters.make_tool_wrapper(fake_tool)
    out = wrapper({"hello": 1}, extra="ignored")
    assert out == "ok"
    assert calls["args"] == {"hello": 1}


def test_slash_wrapper_takes_single_positional_arg(tmp_data_dir):
    """cli.py:6599 calls plugin_handler(user_args) — one positional str, no kwargs."""
    wrapper = adapters.make_slash_handler()
    out = wrapper("")
    assert "Ambient RAG" in out
    out2 = wrapper("stats")
    # stats returns JSON; prove it doesn't crash on a non-empty input
    assert out2 is not None


def test_hook_wrapper_is_keyword_only(tmp_data_dir):
    """Hermes invokes hooks as cb(**kwargs). Wrapper must accept the verified
    kwargs (run_agent.py:10619) plus extras like sender_id."""
    wrapper = adapters.make_hook_wrapper()
    out = wrapper(
        session_id="s1",
        user_message="hi",
        conversation_history=None,
        is_first_turn=True,
        model="claude-haiku",
        platform="cli",
        sender_id="user-42",  # extra kwarg from run_agent.py — must be absorbed
    )
    assert out is None  # message too short → None


def test_hook_wrapper_handles_legacy_kwargs(tmp_data_dir):
    """Some plugins still call pre_llm_call with a different kwarg shape.
    The wrapper should not crash on extras / missing fields."""
    wrapper = adapters.make_hook_wrapper()
    # call with only some of the expected kwargs
    out = wrapper(user_message="x")
    assert out is None  # too short → None, no crash


def test_session_warm_hook_spawns_background_thread(tmp_data_dir, monkeypatch):
    """make_session_warm_hook should return a callable that spawns a thread
    invoking get_engine()._ensure_loaded — and never blocks or raises."""
    import threading
    from hybrid_rag import engine as engine_mod

    # Warm-up is one-shot per process; reset for an isolated test.
    adapters.reset_for_tests()

    calls = {"ensure": 0}

    class _StubEngine:
        def has_embeddings(self):
            calls["ensure"] += 1
            return False

    monkeypatch.setattr(engine_mod, "get_engine", lambda: _StubEngine())

    warm = adapters.make_session_warm_hook()
    warm(session_id="abc", model="claude-haiku", platform="cli")

    # Wait briefly for the daemon thread to run. If anything blocks or raises,
    # this would hang or fail.
    for t in threading.enumerate():
        if t.daemon and t.name != "MainThread":
            t.join(timeout=2.0)

    assert calls["ensure"] >= 1


def test_session_warm_hook_swallows_exceptions(monkeypatch):
    """Engine warming must never raise out — failures fall back to cold-load
    on the first ambient call."""
    import threading
    from hybrid_rag import engine as engine_mod

    adapters.reset_for_tests()

    def _boom():
        raise RuntimeError("simulated MiniLM load failure")

    class _Boom:
        def has_embeddings(self):
            _boom()

    monkeypatch.setattr(engine_mod, "get_engine", lambda: _Boom())

    warm = adapters.make_session_warm_hook()
    # If the closure raised synchronously, this would propagate.
    warm(session_id="abc", model="m", platform="p")

    for t in threading.enumerate():
        if t.daemon and t.name != "MainThread":
            t.join(timeout=2.0)


def test_plugin_yaml_parses_and_has_expected_keys():
    p = Path(__file__).parent.parent / "hybrid_rag" / "plugin.yaml"
    data = yaml.safe_load(p.read_text())
    assert data["name"] == "hybrid-rag"
    assert "version" in data
    tools = data.get("provides_tools", [])
    assert sorted(tools) == ["rag_drill_down", "rag_list_sources", "rag_search"]
    hooks = data.get("provides_hooks", [])
    assert sorted(hooks) == ["on_session_start", "pre_llm_call"]


def test_plugin_yaml_does_not_misrepresent_optional_env_as_required():
    """All env vars used by the plugin are optional (Cohere, Anthropic, and the
    data-dir override). They must NOT appear under `requires_env` — that field
    is reserved for actual hard dependencies and is what `hermes plugin list`
    surfaces. The optional set lives under our `optional_env` documentation
    key, which Hermes ignores."""
    p = Path(__file__).parent.parent / "hybrid_rag" / "plugin.yaml"
    data = yaml.safe_load(p.read_text())
    assert not data.get("requires_env"), \
        "requires_env must be empty/absent — the plugin runs with zero env vars"
    optional = data.get("optional_env", [])
    assert "COHERE_API_KEY" in optional
    assert "ANTHROPIC_API_KEY" in optional
    assert "HERMES_RAG_DATA_DIR" in optional
    for item in optional:
        assert isinstance(item, str), \
            f"optional_env entry should be a string, got {type(item).__name__}: {item!r}"


def test_requirements_files_are_in_sync():
    """`hybrid_rag/requirements.txt` is intentionally a copy of the repo-root
    file (so a single rsync of the inner package carries deps to runtime). If
    they drift, the deployment would silently install the wrong set."""
    root = Path(__file__).parent.parent
    a = (root / "requirements.txt").read_text()
    b = (root / "hybrid_rag" / "requirements.txt").read_text()
    assert a == b, (
        "requirements.txt and hybrid_rag/requirements.txt have diverged. "
        "Update one to match the other (or both, to whatever the new spec is)."
    )


def test_skill_md_has_frontmatter():
    p = Path(__file__).parent.parent / "hybrid_rag" / "skills" / "rag-usage" / "SKILL.md"
    text = p.read_text()
    assert text.startswith("---\n")
    end = text.find("\n---", 4)
    assert end > 0, "SKILL.md frontmatter not closed with ---"
    fm = yaml.safe_load(text[4:end])
    assert fm["name"] == "rag-usage"
    assert "description" in fm and fm["description"]
