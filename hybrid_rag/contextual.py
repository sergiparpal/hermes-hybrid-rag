"""Contextual Retrieval (Anthropic-style).

For each chunk, generate a 50–100 token prefix that locates the chunk within
its parent. Both the dense index and BM25 then index `prefix + chunk`,
materially improving retrieval recall when the chunk text alone is ambiguous.

The parent text is sent in a cached content block so that all chunks of the
same parent share one cache entry — the per-chunk marginal cost is just the
chunk's tokens plus the (small) prefix completion.

All failure modes degrade silently: any error during prefix generation
returns None and the indexer proceeds with the raw chunk text.
"""
from __future__ import annotations

import logging

from . import _anthropic
from ._anthropic import ANTHROPIC_MODEL
from .config import env_flag

# Output cap for the prefix-generation LLM call.
CONTEXTUAL_MAX_TOKENS = 150
# Per-parent thread pool for contextual prefix generation. Anthropic's prompt
# cache is parent-scoped, so concurrent requests for chunks of the SAME parent
# all hit the same cache entry — concurrency multiplies throughput without
# multiplying token cost. Conservative default keeps tier-1 API users under
# the rate limit; bump for tier 3+.
CONTEXTUAL_CONCURRENCY = 4

log = logging.getLogger(__name__)

_CHUNK_INSTRUCTION = (
    "Here is the chunk we want to situate within the whole document:\n"
    "<chunk>\n{chunk}\n</chunk>\n\n"
    "Please give a short succinct context (50-100 tokens) to situate this "
    "chunk within the overall document for the purposes of improving search "
    "retrieval of the chunk. Answer only with the succinct context and "
    "nothing else."
)


def is_contextual_enabled() -> bool:
    return env_flag("HERMES_RAG_CONTEXTUAL")


def generate_contextual_prefix(
    parent_text: str,
    chunk_text: str,
    *,
    client=None,
    model: str = ANTHROPIC_MODEL,
    max_tokens: int = CONTEXTUAL_MAX_TOKENS,
) -> str | None:
    """Generate the contextual prefix string, or None on any failure.

    The parent_text is delivered in a `cache_control: ephemeral` block, so
    successive calls for chunks of the same parent within ~5 minutes hit the
    Anthropic prompt cache. See:
    https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
    """
    if not parent_text or not chunk_text:
        return None
    cli = client if client is not None else _anthropic.get_client()
    if cli is None:
        return None

    try:
        msg = cli.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"<document>\n{parent_text}\n</document>",
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "text",
                            "text": _CHUNK_INSTRUCTION.format(chunk=chunk_text),
                        },
                    ],
                }
            ],
        )
        text = _anthropic.extract_text(msg).strip()
        return text or None
    except Exception as e:
        # Per spec: never abort an index run because contextual generation
        # failed. The caller logs once per parent in aggregate.
        log.warning("contextual prefix generation failed: %s", e)
        return None
