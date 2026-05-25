"""Cross-cutting configuration: data-dir paths, env-flag parsing, and the
pipeline-wide tunables read by more than one module.

Feature-specific defaults live with their consumers:

- embed model id / dim registry / env reads → ``embeddings.py``
- reranker model ids → ``rerank.py``
- Anthropic default model → ``_anthropic.py``
- contextual-retrieval token caps + concurrency → ``contextual.py``
- ambient pipeline thresholds + pools → ``hooks.py``
- ambient convo-memory weights → ``convo.py``

The single data-dir rule: explicit ``Store(data_dir=...)`` arg
> ``HERMES_RAG_DATA_DIR`` env var > default
``~/.hermes/plugins/hybrid-rag/data/``.
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DATA_DIR = Path.home() / ".hermes" / "plugins" / "hybrid-rag" / "data"


def env_flag(name: str, default: bool = False) -> bool:
    """Truthy parse of an env var. Treats `1/true/yes/on` (case-insensitive)
    as True. Unset → ``default``."""
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def get_data_dir() -> Path:
    env = os.environ.get("HERMES_RAG_DATA_DIR")
    return Path(env) if env else DEFAULT_DATA_DIR


def db_path(data_dir: Path | None = None) -> Path:
    return (data_dir or get_data_dir()) / "rag.sqlite"


def npz_path(data_dir: Path | None = None) -> Path:
    return (data_dir or get_data_dir()) / "embeddings.npz"


def bm25_path(data_dir: Path | None = None) -> Path:
    return (data_dir or get_data_dir()) / "bm25.pkl"


def toggles_path(data_dir: Path | None = None) -> Path:
    return (data_dir or get_data_dir()) / "toggles.json"


# --- Pipeline tunables shared by more than one module ---

MAX_CHUNK = 300
CHUNK_OVERLAP = 50
MAX_PARENT_CHARS = 8000
# Per-file size cap at index time. The whole file is read into memory by the
# extractors (md/txt/pdf), so an unbounded read on a multi-GB file OOMs the
# Hermes process. 50 MB covers ~all human-written corpora; users with larger
# files should split them or raise this knowingly.
MAX_INDEX_FILE_BYTES = 50 * 1024 * 1024
# Per-page char cap for PDF extraction. A malformed / adversarial PDF can
# make pypdf return huge per-page strings; cap before they hit chunking /
# embedding.
MAX_PDF_PAGE_CHARS = 200_000
# Markdown text before the first `##` heading is captured as a synthetic
# "preamble" parent only when its body length (after stripping any leading
# `# H1` line) clears this threshold. Below it the prefix is dropped on the
# assumption that it's boilerplate (a stray title line, frontmatter, etc.).
PREAMBLE_MIN_CHARS = 200
RRF_K = 60

# `rag_search` funnel widths. The pool of chunks that survive the
# second-level RRF before parent rollup, and the pool of parents fed into
# the reranker.
RAG_SEARCH_CHUNK_POOL = 30
RAG_SEARCH_PARENT_POOL = 10
