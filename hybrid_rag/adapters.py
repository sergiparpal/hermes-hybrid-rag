"""Thin closures wrapping pure handlers to whatever shape Hermes wants.

This is the only Hermes-coupled module besides `__init__.py::register`. If the
real Hermes API differs from `HERMES_API.md`, edit closures here — pure logic
stays untouched. All shapes below are verified against the Hermes source.
"""
from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)


def _safe(label: str, fn):
    """Call ``fn()`` and swallow any exception with a debug log.

    Used by the warm-up hook where we want best-effort preloading and never a
    raise that could disturb session start.
    """
    try:
        fn()
    except Exception as e:
        log.debug("%s failed: %s", label, e)


# `on_session_start` fires per new session. Warming is idempotent at the engine
# and reranker layers, but we still gate it with a one-shot flag so a
# long-running Hermes process doesn't spawn one thread per session.
_WARM_LAUNCHED = False
_WARM_LOCK = threading.Lock()


def reset_for_tests() -> None:
    """Test helper. Clears the one-shot warm flag so successive tests can
    re-trigger the warm-up path."""
    global _WARM_LAUNCHED
    with _WARM_LOCK:
        _WARM_LAUNCHED = False


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
    platform, sender_id. The pure hook only consumes a subset; the rest
    flow through ``**kwargs`` so signature drift in Hermes doesn't break
    the wire.
    """
    def _hook(*, session_id="", user_message="", conversation_history=None,
              model="", platform="", **kwargs):
        from .hooks import ambient_pre_llm_call
        return ambient_pre_llm_call(
            session_id=session_id,
            user_message=user_message,
            conversation_history=conversation_history or [],
            model=model,
            platform=platform,
            **kwargs,
        )
    return _hook


def make_session_warm_hook():
    """Warm the engine + ambient reranker on new sessions. Background thread
    — never blocks session_start. Hermes fires this for every brand-new
    session (run_agent.py:10519); we gate with a one-shot flag so a
    long-running process spawns the warm thread at most once.

    The ambient path reranks every turn with the local cross-encoder;
    preload that here too so the first per-turn rerank doesn't pay the
    model-download / model-load cost.
    """
    def _warm(*, session_id="", model="", platform="", **_):
        global _WARM_LAUNCHED
        with _WARM_LOCK:
            if _WARM_LAUNCHED:
                return
            _WARM_LAUNCHED = True

        def _bg():
            from .engine import get_engine
            from .rerank import warm_local_cross_encoder
            # has_embeddings() triggers the lazy load. Idempotent — a no-op
            # if another caller already loaded.
            _safe("session warm-up", lambda: get_engine().has_embeddings())
            _safe("cross-encoder warm-up", warm_local_cross_encoder)

        threading.Thread(target=_bg, daemon=True).start()
    return _warm
