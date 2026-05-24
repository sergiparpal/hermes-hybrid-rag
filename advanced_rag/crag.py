"""CRAG-lite — Critique + retry layer on the explicit `rag_search` path.

After the full pipeline produces a set of parents, an LLM judges whether
they're sufficient to answer the query. If not, the query is reformulated
once and the pipeline runs again. **Hard cap: exactly one retry.** No second
critique, no loops.

Off by default (`HERMES_RAG_CRAG=1` to opt in). Anthropic unavailable → CRAG
is skipped silently and the original parents are returned. Per the design,
this is a robustness layer — it does not replace better retrieval (and
should only be enabled after Phase 2 contextual retrieval is in use).
"""
from __future__ import annotations

import json
import logging
import os
import re

from .config import ANTHROPIC_MODEL
from .retrieval import ParentResult

log = logging.getLogger(__name__)

_JUDGE_PROMPT = (
    "You are evaluating whether retrieved document excerpts are sufficient "
    "to answer a user's query.\n\n"
    "Query: {q}\n\n"
    "Retrieved excerpts:\n{excerpts}\n\n"
    "Respond with a single JSON object (no surrounding text, no code "
    "fences):\n"
    '{{"sufficient": true | false, "reason": "<one sentence>"}}'
)

_REFORMULATE_PROMPT = (
    "You are rewriting a search query that did not find sufficient evidence "
    "in a document corpus.\n\n"
    "Original query: {q}\n\n"
    "Brief reason the first retrieval was insufficient: {reason}\n\n"
    "Rewrite the query to be more likely to retrieve the right documents. "
    "Keep it under 25 words. Output ONLY the rewritten query — no "
    "explanation, no quotes."
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_WHITESPACE_RE = re.compile(r"\s+")
# Hard cap on the rewritten query length. The prompt asks for ≤25 words; this
# is a defense-in-depth bound that also rejects an LLM that returns a wall of
# text instead of a query.
_REFORMULATED_MAX_CHARS = 500

_CLIENT = None


def is_enabled() -> bool:
    val = os.environ.get("HERMES_RAG_CRAG", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _get_client():
    global _CLIENT
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


def _reset_client_for_tests() -> None:
    global _CLIENT
    _CLIENT = None


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def _excerpt_block(parents: list[ParentResult], max_chars: int = 500) -> str:
    if not parents:
        return "(no excerpts retrieved)"
    pieces = []
    for i, p in enumerate(parents):
        title = p.title or f"{p.kind} (parent {p.parent_id})"
        body = (p.text or "")[:max_chars]
        pieces.append(f"[{i + 1}] {title}\n{body}")
    return "\n\n".join(pieces)


def judge_retrieval(
    query: str,
    parents: list[ParentResult],
    *,
    client=None,
    model: str = ANTHROPIC_MODEL,
) -> dict:
    """Returns `{"sufficient": bool, "reason": str}`.

    On any failure path (no API key, no SDK, network error, malformed
    response) returns `{"sufficient": True, "reason": "skip"}` so the caller
    treats CRAG as a no-op rather than triggering a useless retry.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return {"sufficient": True, "reason": "anthropic unavailable"}
    cli = client if client is not None else _get_client()
    if cli is None:
        return {"sufficient": True, "reason": "anthropic unavailable"}

    excerpts = _excerpt_block(parents)
    try:
        msg = cli.messages.create(
            model=model,
            max_tokens=256,
            messages=[{"role": "user",
                       "content": _JUDGE_PROMPT.format(q=query, excerpts=excerpts)}],
        )
        text = "".join(getattr(part, "text", "") for part in msg.content)
        payload = json.loads(_strip_fences(text))
        return {
            "sufficient": bool(payload.get("sufficient", True)),
            "reason": str(payload.get("reason", "")),
        }
    except Exception as e:
        log.warning("CRAG judge failed: %s", e)
        return {"sufficient": True, "reason": f"judge failed: {e!r}"}


def reformulate_query(
    query: str,
    parents: list[ParentResult],
    judge_reason: str,
    *,
    client=None,
    model: str = ANTHROPIC_MODEL,
) -> str | None:
    """Returns the rewritten query, or None on any failure. None tells the
    caller to fall back to the original query (no retry)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    cli = client if client is not None else _get_client()
    if cli is None:
        return None

    try:
        msg = cli.messages.create(
            model=model,
            max_tokens=128,
            messages=[{"role": "user",
                       "content": _REFORMULATE_PROMPT.format(
                           q=query,
                           reason=judge_reason or "(no reason given)")}],
        )
        text = "".join(getattr(part, "text", "") for part in msg.content).strip()
        # Strip whatever the model wrapped the rewrite in: ASCII quotes,
        # backticks, smart quotes (U+2018-201D), and surrounding whitespace.
        # The prompt asks for the query alone, but Haiku occasionally wraps it.
        text = text.strip("\"' \n\t`‘’“”")
        # Collapse any internal whitespace (newlines, tabs) into single
        # spaces — the rewritten query is fed back into retrieval and BM25
        # tokenization, so multi-line garbage here just adds noise.
        text = _WHITESPACE_RE.sub(" ", text).strip()
        if not text or len(text) > _REFORMULATED_MAX_CHARS:
            return None
        return text
    except Exception as e:
        log.warning("CRAG reformulate failed: %s", e)
        return None
