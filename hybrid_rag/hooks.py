"""Ambient pre-LLM-call hook. Thin glue between Hermes's ``pre_llm_call``
signature and the ``AmbientPipeline`` that does the actual work.

This module's job is exactly: (1) absorb Hermes kwargs and surface drift,
(2) check the state toggle and message-length gate, (3) delegate. The
pipeline lives in ``pipelines.py``; presentation in ``formatting.py``.

Must never raise — the body is wrapped in ``try/except`` and returns
``None`` on any failure path.
"""
from __future__ import annotations

import logging
import threading

from . import state
from .engine import get_engine
from .pipelines import AmbientPipeline

log = logging.getLogger(__name__)

_MIN_MESSAGE_LEN = 8

# Track Hermes hook kwargs we don't consume — surface them once so that an
# upstream signature change adding a useful field doesn't go unnoticed.
_HOOK_KNOWN_KWARGS = frozenset({
    "session_id", "user_message", "conversation_history", "model", "platform",
})
_HOOK_SEEN_EXTRA_KWARGS: set[str] = set()
_HOOK_KWARG_LOG_LOCK = threading.Lock()


def _log_unfamiliar_kwargs(kwargs: dict) -> None:
    """One-shot debug log per never-before-seen kwarg name. Helps spot a
    Hermes upgrade that started passing something the hook should be
    reading. Cheap — we only log first occurrence."""
    extras = set(kwargs) - _HOOK_KNOWN_KWARGS - _HOOK_SEEN_EXTRA_KWARGS
    if not extras:
        return
    with _HOOK_KWARG_LOG_LOCK:
        new = extras - _HOOK_SEEN_EXTRA_KWARGS
        if not new:
            return
        _HOOK_SEEN_EXTRA_KWARGS.update(new)
        log.debug("ambient_pre_llm_call: ignoring new kwargs %s", sorted(new))


def ambient_pre_llm_call(
    *,
    session_id: str | None = None,
    user_message: str = "",
    conversation_history=None,
    model: str | None = None,
    platform: str | None = None,
    **kwargs,
):
    """Return ``{"context": str}`` to inject ambient context, or ``None``
    to do nothing. Never raises.

    Hermes passes additional kwargs that this hook doesn't use today
    (``is_first_turn``, ``sender_id``, etc.); they're absorbed by
    ``**kwargs`` so upstream signature drift never breaks the wire.
    First-seen extras are logged once at debug so an upgrade that starts
    passing something *useful* doesn't go unnoticed.
    """
    try:
        if kwargs:
            _log_unfamiliar_kwargs(kwargs)
        if not state.is_ambient_enabled(session_id):
            return None
        if not user_message or len(user_message.strip()) < _MIN_MESSAGE_LEN:
            return None

        context = AmbientPipeline(get_engine()).run(
            user_message, session_id=session_id,
        )
        return {"context": context} if context else None
    except Exception as e:
        log.warning("ambient_pre_llm_call failed: %s", e)
        return None
