"""Ambient pre-LLM-call hook. Injects up to AMBIENT_TOP_PARENTS parents into
the prompt when the user message looks substantive and the top result clears
the threshold. Must never raise — return None on any failure path.
"""
from __future__ import annotations

import logging

from . import retrieval, state
from .config import (
    AMBIENT_SCORE_THRESHOLD,
    AMBIENT_TOKEN_CAP,
    AMBIENT_TOP_PARENTS,
)
from .engine import get_engine

log = logging.getLogger(__name__)

_MIN_MESSAGE_LEN = 8


def ambient_pre_llm_call(
    *,
    session_id: str | None = None,
    user_message: str = "",
    conversation_history=None,
    is_first_turn: bool = False,
    model: str | None = None,
    platform: str | None = None,
    **kwargs,
):
    """Return `{"context": str}` to inject ambient context, or `None` to do
    nothing. Never raises."""
    try:
        if not state.is_ambient_enabled(session_id):
            return None
        if not user_message or len(user_message.strip()) < _MIN_MESSAGE_LEN:
            return None

        engine = get_engine()
        engine._ensure_loaded()
        if engine._embeddings is None or engine._embeddings.shape[0] == 0:
            return None

        hits = retrieval.hybrid_search(engine, user_message, k_pool=30)
        if not hits:
            return None
        parents = retrieval.chunks_to_parents(
            engine, hits, top=AMBIENT_TOP_PARENTS,
        )
        if not parents or parents[0].score < AMBIENT_SCORE_THRESHOLD:
            return None

        context = retrieval.format_context(parents, token_cap=AMBIENT_TOKEN_CAP)
        if not context:
            return None
        return {"context": context}
    except Exception as e:
        log.warning("ambient_pre_llm_call failed: %s", e)
        return None
