"""Thin closures wrapping pure handlers to whatever shape Hermes wants.

This is the only Hermes-coupled module besides `__init__.py::register`. If the
real Hermes API differs from `HERMES_API.md`, edit closures here — pure logic
stays untouched. All shapes below are verified against the Hermes source.
"""
from __future__ import annotations


def make_cli_setup():
    def _setup(parser):
        from .cli import setup_rag_parser
        setup_rag_parser(parser)
    return _setup


def make_cli_handler():
    def _handle(args):
        from .cli import handle_rag
        return handle_rag(args)
    return _handle


def make_slash_handler():
    """Hermes invokes plugin slash handlers with a single positional str arg.
    Source: cli.py:6599 calls plugin_handler(user_args). No kwargs are passed,
    so per-session toggle is impossible — v0.1 uses a process-global toggle.
    """
    def _slash(rest: str):
        from .slash import slash_rag
        return slash_rag(rest)
    return _slash


def make_tool_wrapper(fn):
    def _tool(args, **kwargs):
        return fn(args)
    return _tool


def make_hook_wrapper():
    """Wrap ambient_pre_llm_call to match Hermes's pre_llm_call invocation.

    Hermes calls hooks as cb(**kwargs). Real kwargs (run_agent.py:10619):
    session_id, user_message, conversation_history, is_first_turn, model,
    platform, sender_id. The legacy variant (still in use by some plugins)
    passes a different set; keyword-only with defaults handles both.
    """
    def _hook(*, session_id="", user_message="", conversation_history=None,
              is_first_turn=False, model="", platform="", **kwargs):
        from .hooks import ambient_pre_llm_call
        return ambient_pre_llm_call(
            session_id=session_id,
            user_message=user_message,
            conversation_history=conversation_history or [],
            is_first_turn=is_first_turn,
            model=model,
            platform=platform,
            **kwargs,
        )
    return _hook


def make_session_warm_hook():
    """Warm the engine on new sessions. Background thread — never blocks
    session_start. Hermes fires this only on brand-new sessions
    (run_agent.py:10519), so the cost is paid once per session.
    """
    def _warm(*, session_id="", model="", platform="", **_):
        import threading

        def _bg():
            try:
                from .engine import get_engine
                get_engine()._ensure_loaded()
            except Exception:
                # Cold load on first ambient call is the fallback. Never raise.
                pass

        threading.Thread(target=_bg, daemon=True).start()
    return _warm
