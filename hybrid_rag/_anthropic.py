"""Shared Anthropic SDK plumbing.

A single cached client (one connection pool, one prompt-cache state) plus
helpers for the response-shape patterns the three LLM-using modules
(``expansion``, ``crag``, ``contextual``) all need. Optional dep — every
function returns a fallback value when the SDK isn't installed or no API key
is configured; callers never get an exception.
"""
from __future__ import annotations

import os
import re
import threading

ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"

_CLIENT = None
_LOCK = threading.Lock()

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def get_client():
    """Return the shared ``anthropic.Anthropic`` client, or ``None`` if the
    SDK is missing, ``ANTHROPIC_API_KEY`` is unset, or instantiation fails.

    Double-checked locking so concurrent first-callers don't race on import.
    """
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    with _LOCK:
        if _CLIENT is not None:
            return _CLIENT
        try:
            import anthropic
        except ImportError:
            return None
        try:
            _CLIENT = anthropic.Anthropic()
        except Exception:
            return None
        return _CLIENT


def reset_for_tests() -> None:
    """Drop the cached client. Tests that swap ``sys.modules['anthropic']``
    between cases need this so the next ``get_client()`` re-imports."""
    global _CLIENT
    with _LOCK:
        _CLIENT = None


def extract_text(msg) -> str:
    """Concatenate every ``.text`` part of an Anthropic message response."""
    return "".join(getattr(part, "text", "") for part in msg.content)


def strip_json_fences(text: str) -> str:
    """Strip the ```json ... ``` (or bare ``` ... ```) wrapper if present.
    Returns the inner text stripped; otherwise the input stripped."""
    m = _JSON_FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()
