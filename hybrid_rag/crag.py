"""CRAG-lite — Critique + retry layer on the explicit `rag_search` path.

After the full pipeline produces a set of parents, an LLM judges whether
they're sufficient to answer the query. If not, the query is reformulated
once and the pipeline runs again. **Hard cap: exactly one retry.** No second
critique, no loops.

Off by default (`HERMES_RAG_CRAG=1` to opt in). Anthropic unavailable → CRAG
is skipped silently and the original parents are returned.
"""
from __future__ import annotations

import json
import logging
import re

from . import _anthropic
from ._anthropic import ANTHROPIC_MODEL
from .config import env_flag
from .models import ParentResult

log = logging.getLogger(__name__)

_JUDGE_AND_REFORMULATE_PROMPT = (
    "You are evaluating whether retrieved document excerpts are sufficient "
    "to answer a user's query, and rewriting the query if they are not.\n\n"
    "Query: {q}\n\n"
    "Retrieved excerpts:\n{excerpts}\n\n"
    "Respond with a single JSON object (no surrounding text, no code "
    "fences). Two shapes:\n\n"
    "Sufficient — the excerpts can answer the query:\n"
    '  {{"sufficient": true, "reason": "<one sentence>"}}\n\n'
    "Insufficient — supply a rewritten query (≤25 words, no quotes) that "
    "would be more likely to retrieve the missing evidence:\n"
    '  {{"sufficient": false, "reason": "<one sentence>", '
    '"rewritten_query": "<the rewrite>"}}'
)

_WHITESPACE_RE = re.compile(r"\s+")
# Hard cap on the rewritten query length. The prompt asks for ≤25 words; this
# is a defense-in-depth bound that also rejects an LLM that returns a wall of
# text instead of a query.
_REFORMULATED_MAX_CHARS = 500


def is_enabled() -> bool:
    return env_flag("HERMES_RAG_CRAG")


def _excerpt_block(parents: list[ParentResult], max_chars: int = 500) -> str:
    if not parents:
        return "(no excerpts retrieved)"
    pieces = []
    for i, p in enumerate(parents):
        title = p.title or f"{p.kind} (parent {p.parent_id})"
        body = (p.text or "")[:max_chars]
        pieces.append(f"[{i + 1}] {title}\n{body}")
    return "\n\n".join(pieces)


def _clean_rewritten_query(text: str) -> str | None:
    """Trim a model's rewritten-query output. Returns None for empty,
    overlong, or otherwise unusable values — the caller treats that as
    "no retry" rather than feeding garbage into BM25."""
    # Strip whatever the model wrapped the rewrite in: ASCII quotes,
    # backticks, smart quotes (U+2018-201D), and surrounding whitespace.
    text = text.strip("\"' \n\t`‘’“”")
    # Collapse any internal whitespace (newlines, tabs) into single spaces
    # — the rewritten query is fed into BM25 tokenization, so multi-line
    # noise just dilutes the score.
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if not text or len(text) > _REFORMULATED_MAX_CHARS:
        return None
    return text


def judge_and_reformulate(
    query: str,
    parents: list[ParentResult],
    *,
    client=None,
    model: str = ANTHROPIC_MODEL,
) -> dict:
    """Single-call CRAG: judge sufficiency AND, if insufficient, return a
    rewritten query in one round-trip.

    Returns ``{"sufficient": bool, "reason": str, "rewritten_query": str|None}``.

    Replaces an earlier two-call judge → reformulate cascade — saves ~500 ms
    on the CRAG-enabled path at the cost of a slightly longer prompt that
    always asks for both shapes. The model decides early and only emits the
    rewrite when needed, so the net saving holds.

    On any failure (no API key, no SDK, network error, malformed response)
    returns ``{"sufficient": True, "reason": "<details>",
    "rewritten_query": None}`` so the caller treats CRAG as a no-op.
    """
    cli = client if client is not None else _anthropic.get_client()
    if cli is None:
        return {"sufficient": True, "reason": "anthropic unavailable",
                "rewritten_query": None}

    excerpts = _excerpt_block(parents)
    try:
        msg = cli.messages.create(
            model=model,
            max_tokens=384,
            messages=[{"role": "user",
                       "content": _JUDGE_AND_REFORMULATE_PROMPT.format(
                           q=query, excerpts=excerpts)}],
        )
        payload = json.loads(_anthropic.strip_json_fences(_anthropic.extract_text(msg)))
        sufficient = bool(payload.get("sufficient", True))
        reason = str(payload.get("reason", ""))
        rewritten = payload.get("rewritten_query")
        if sufficient or not isinstance(rewritten, str):
            return {"sufficient": sufficient, "reason": reason,
                    "rewritten_query": None}
        return {"sufficient": False, "reason": reason,
                "rewritten_query": _clean_rewritten_query(rewritten)}
    except Exception as e:
        log.warning("CRAG judge_and_reformulate failed: %s", e)
        return {"sufficient": True, "reason": f"call failed: {e!r}",
                "rewritten_query": None}
