"""Recursive text splitter — packs text into <=max_size pieces, recursing on
finer separators when a part overflows, and falling back to fixed-size hard
splits with overlap when no separator works.
"""
from __future__ import annotations

from .config import MAX_CHUNK, CHUNK_OVERLAP


def _hard_split(text: str, max_size: int, overlap: int) -> list[str]:
    if not text:
        return []
    if overlap >= max_size:
        overlap = max_size // 2
    step = max(1, max_size - overlap)
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        out.append(text[i : i + max_size])
        i += step
    return out


def _split_with_separator(text: str, sep: str) -> list[str]:
    if sep == "":
        return list(text)
    parts = text.split(sep)
    if len(parts) == 1:
        return parts
    out: list[str] = []
    for idx, p in enumerate(parts):
        if idx < len(parts) - 1:
            out.append(p + sep)
        else:
            out.append(p)
    return [p for p in out if p]


def recursive_split(
    text: str,
    max_size: int = MAX_CHUNK,
    overlap: int = CHUNK_OVERLAP,
    separators: tuple = ("\n\n", "\n", ". ", " ", ""),
) -> list[str]:
    """Greedy pack of split parts; recurses on remaining separator list when a
    single part overflows; falls through to fixed-size hard split with overlap
    when no separator works.

    **Bound:** with ``overlap == 0`` every chunk satisfies ``len(c) <= max_size``.
    With ``overlap > 0`` the post-pass prepends the previous chunk's tail to
    each successor and accepts merges up to ``max_size + overlap`` chars — so
    callers must size buffers against ``max_size + overlap``, not ``max_size``
    alone. The parent cap (``MAX_PARENT_CHARS``) absorbs the slop downstream.
    """
    if max_size <= 0:
        raise ValueError(f"max_size must be positive, got {max_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")
    if text is None:
        return []
    if not text.strip():
        return []
    if len(text) <= max_size:
        return [text]

    sep, rest = separators[0], separators[1:]
    parts = _split_with_separator(text, sep)

    chunks: list[str] = []
    buf = ""

    def flush():
        nonlocal buf
        if buf:
            chunks.append(buf)
            buf = ""

    for part in parts:
        if len(part) > max_size:
            flush()
            if rest:
                chunks.extend(recursive_split(part, max_size, overlap, rest))
            else:
                chunks.extend(_hard_split(part, max_size, overlap))
            continue
        if len(buf) + len(part) <= max_size:
            buf += part
        else:
            flush()
            buf = part
    flush()

    if overlap > 0 and len(chunks) > 1:
        out: list[str] = [chunks[0]]
        for prev, cur in zip(chunks, chunks[1:]):
            tail = prev[-overlap:] if len(prev) > overlap else prev
            merged = (tail + cur) if not cur.startswith(tail) else cur
            if len(merged) <= max_size + overlap:
                out.append(merged)
            else:
                out.append(cur)
        return out
    return chunks
