"""LLM-based query expansion. Returns the original query plus paraphrases and
a HyDE document. Always returns at least [q]; never raises out to the caller.
"""
from __future__ import annotations

import json
import logging

from . import _anthropic
from ._anthropic import ANTHROPIC_MODEL

log = logging.getLogger(__name__)

_PROMPT = """You are helping a retrieval system find relevant documents.

Original query: {q}

Output a single JSON object (no surrounding text, no code fences) with two keys:
  - "paraphrases": list of 3 distinct rewrites of the query, each preserving the
    user's intent but using different vocabulary.
  - "hyde": one short hypothetical answer paragraph (1-3 sentences) that, if it
    existed in the corpus, would likely be retrieved for this query.

Return only the JSON. Do not explain.
"""


def expand_query(q: str) -> list[str]:
    """Return [q] (fallback) or [q, p1, p2, p3, hyde] when expansion succeeds.

    Paraphrases are deduplicated against the original query AND against each
    other (case- and whitespace-insensitive), so a model that returns
    ``["foo", "FOO", "bar"]`` does not waste a hybrid_search round on the
    duplicate.

    Failure modes that fall back silently to [q]:
      - `anthropic` SDK missing or `ANTHROPIC_API_KEY` unset
      - Any exception raised by the SDK call
      - Response missing the expected JSON structure
    """
    # Whitespace-only and empty queries both collapse to "no expansion" —
    # returning ``[q]`` (a whitespace string) would waste a hybrid_search
    # variant on noise the moment a future caller skipped the tool layer's
    # validation. Production callers already reject empty queries upstream.
    if not q or not q.strip():
        return []
    client = _anthropic.get_client()
    if client is None:
        return [q]

    try:
        msg = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": _PROMPT.format(q=q)}],
        )
        payload = json.loads(_anthropic.strip_json_fences(_anthropic.extract_text(msg)))
        paraphrases = payload.get("paraphrases", []) or []
        hyde = payload.get("hyde", "") or ""
        out = [q]
        # Dedupe FIRST (canonical = stripped + lowercased), THEN cap at 3 — so
        # a model that returns ["foo", "foo", "bar", "baz"] still contributes
        # three useful paraphrases instead of being undermined by duplicates.
        seen = {q.strip().lower()}
        kept = 0
        for p in paraphrases:
            if kept >= 3:
                break
            if not isinstance(p, str):
                continue
            stripped = p.strip()
            if not stripped:
                continue
            key = stripped.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(stripped)
            kept += 1
        # Dedupe HyDE against the original query and the surviving paraphrases
        # too — otherwise a model that echoes the query into the `hyde` field
        # wastes a hybrid_search variant on an exact duplicate.
        if isinstance(hyde, str) and hyde.strip():
            stripped = hyde.strip()
            if stripped.lower() not in seen:
                out.append(stripped)
        return out or [q]
    except Exception as e:
        log.warning("query expansion failed: %s", e)
        return [q]
