"""`/rag`, `/rag on|off`, `/rag stats` — pure slash-command dispatcher returning
the string Hermes will display."""
from __future__ import annotations

import json

from . import state as _state_default
from .storage import Store as _Store_default

_HELP = (
    "Hierarchical RAG slash commands:\n"
    "  /rag           — show ambient toggle state\n"
    "  /rag on        — enable ambient context injection (default)\n"
    "  /rag off       — disable ambient context injection\n"
    "  /rag stats     — show indexed-file counts\n"
    "\n"
    "Note: the on/off toggle is process-global, not per-session. In gateway\n"
    "deployments (Discord, Slack, …) every session sharing this Hermes\n"
    "process sees the same setting."
)


def slash_rag(rest: str = "", *, state_mod=None, store_factory=None,
              session_id: str | None = None, **_kwargs) -> str:
    state_mod = state_mod or _state_default
    store_factory = store_factory or _Store_default

    parts = (rest or "").strip().split()
    cmd = parts[0].lower() if parts else ""

    if cmd == "":
        on = state_mod.is_ambient_enabled(session_id)
        return f"Ambient RAG is {'on' if on else 'off'}.\n\n{_HELP}"
    if cmd == "on":
        state_mod.set_ambient(True, session_id=session_id)
        return "Ambient RAG: on"
    if cmd == "off":
        state_mod.set_ambient(False, session_id=session_id)
        return "Ambient RAG: off"
    if cmd == "stats":
        try:
            stats = store_factory().stats()
            return json.dumps(stats, indent=2, sort_keys=True)
        except Exception as e:
            return f"error reading stats: {e}"
    return _HELP
