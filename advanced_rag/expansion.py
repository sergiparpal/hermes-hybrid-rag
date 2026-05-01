"""LLM-based query expansion. Returns the original query plus paraphrases and
a HyDE document. Always returns at least [q]; never raises out to the caller.
"""
from __future__ import annotations

import json
import logging
import os
import re

from .config import ANTHROPIC_MODEL

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

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def expand_query(q: str) -> list[str]:
    """Return [q] (fallback) or [q, p1, p2, p3, hyde] when expansion succeeds.

    Failure modes that fall back silently to [q]:
      - `import anthropic` fails (package not installed)
      - ANTHROPIC_API_KEY env var unset
      - Any exception raised by the SDK call
      - Response missing the expected JSON structure
    """
    if not q or not q.strip():
        return [q] if q else []
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return [q]
    try:
        import anthropic
    except ImportError:
        return [q]

    try:
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": _PROMPT.format(q=q)}],
        )
        text = "".join(getattr(part, "text", "") for part in msg.content)
        payload = json.loads(_strip_fences(text))
        paraphrases = payload.get("paraphrases", []) or []
        hyde = payload.get("hyde", "") or ""
        out = [q]
        for p in paraphrases[:3]:
            if isinstance(p, str) and p.strip() and p.strip() != q.strip():
                out.append(p.strip())
        if isinstance(hyde, str) and hyde.strip():
            out.append(hyde.strip())
        return out or [q]
    except Exception as e:
        log.warning("query expansion failed: %s", e)
        return [q]
