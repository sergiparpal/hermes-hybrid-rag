"""Ambient conversational memory (opt-in).

When `HERMES_RAG_AMBIENT_CONVO_MEMORY=1`, the ambient retrieval path mixes
the current user turn's query embedding with the embeddings of the previous
1–2 user turns. This helps with follow-ups ("explain more about that")
where the chunk text in the corpus matches the prior topic rather than the
literal current message.

Trade-off: when the user changes topic, the prior embeddings contaminate
retrieval. That's why it is off by default.

Ring buffer is keyed by session id and lives only in memory — it never
persists across process restarts.
"""
from __future__ import annotations

import threading

import numpy as np

from .config import env_flag

# Weights apply to current/previous/older user turn embeddings, normalized
# before mixing. Owned here because this module is the only consumer.
AMBIENT_CONVO_MEMORY_WEIGHTS = (1.0, 0.25, 0.1)

_LOCK = threading.Lock()
_RINGS: dict[str, list[np.ndarray]] = {}
_RING_SIZE = len(AMBIENT_CONVO_MEMORY_WEIGHTS)  # current + N priors


def is_enabled() -> bool:
    return env_flag("HERMES_RAG_AMBIENT_CONVO_MEMORY")


def push(session_id: str, vec: np.ndarray) -> None:
    """Insert `vec` as the newest entry for `session_id`. Older entries
    drop off the tail when the buffer reaches RING_SIZE."""
    if not session_id:
        return
    with _LOCK:
        ring = _RINGS.setdefault(session_id, [])
        ring.insert(0, np.asarray(vec, dtype=np.float32))
        if len(ring) > _RING_SIZE:
            del ring[_RING_SIZE:]


def get_ring(session_id: str) -> list[np.ndarray]:
    """Snapshot copy of the ring buffer for `session_id`, newest first."""
    if not session_id:
        return []
    with _LOCK:
        return list(_RINGS.get(session_id, []))


def reset_for_tests() -> None:
    with _LOCK:
        _RINGS.clear()


def mix_with_history(
    current: np.ndarray,
    history: list[np.ndarray],
    weights: tuple[float, ...] = AMBIENT_CONVO_MEMORY_WEIGHTS,
) -> np.ndarray:
    """Linearly combine `current` with up to `len(weights)-1` history vectors
    (newest first), then L2-normalize. Weights are normalized to sum to 1.

    If `history` is empty, returns `current` unchanged (already normalized
    by the embedder)."""
    if not history:
        return current
    vecs = [current] + list(history)[: len(weights) - 1]
    ws = list(weights[: len(vecs)])
    total = float(sum(ws))
    if total <= 0:
        return current
    ws = [w / total for w in ws]
    out = np.zeros_like(current, dtype=np.float32)
    for v, w in zip(vecs, ws):
        # Defensive: skip any malformed entry (e.g. dim drift across an
        # in-process reindex) rather than crashing the ambient hook.
        if v.shape == current.shape:
            out += np.asarray(v, dtype=np.float32) * w
    norm = float(np.linalg.norm(out))
    if norm == 0.0:
        return current
    return (out / norm).astype(np.float32)
