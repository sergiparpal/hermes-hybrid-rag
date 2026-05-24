"""Configuration constants and data-dir resolution.

The single rule: explicit `Store(data_dir=...)` arg > `HERMES_RAG_DATA_DIR`
env var > default `~/.hermes/plugins/advanced-rag/data/`.
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DATA_DIR = Path.home() / ".hermes" / "plugins" / "advanced-rag" / "data"


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


# Tunables
MAX_CHUNK = 300
CHUNK_OVERLAP = 50
MAX_PARENT_CHARS = 8000
# Per-file size cap at index time. The whole file is read into memory by the
# extractors (md/txt/pdf), so an unbounded read on a multi-GB file OOMs the
# Hermes process. 50 MB covers ~all human-written corpora; users with larger
# files should split them or raise this knowingly.
MAX_INDEX_FILE_BYTES = 50 * 1024 * 1024
# Per-page char cap for PDF extraction. A malformed / adversarial PDF can make
# pypdf return huge per-page strings; cap before they hit chunking/embedding.
MAX_PDF_PAGE_CHARS = 200_000
# Markdown text before the first `##` heading is captured as a synthetic
# "preamble" parent only when its body length (after stripping any leading
# `# H1` line) clears this threshold. Below it the prefix is dropped on the
# assumption that it's boilerplate (a stray title line, frontmatter, etc.).
PREAMBLE_MIN_CHARS = 200
RRF_K = 60
AMBIENT_TOP_PARENTS = 3
AMBIENT_TOKEN_CAP = 1500
AMBIENT_SCORE_THRESHOLD = 0.25
# Modern multilingual default. Override at runtime with HERMES_RAG_EMBED_MODEL
# (any sentence-transformers-compatible id). Switching models requires a full
# `hermes rag index --force` — the .npz dim won't match otherwise.
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"
# Known dimensionalities. Used to pre-allocate the `(0, dim)` empty-result
# array in `Embedder.encode([])` without loading the model. Anything missing
# here is auto-detected on first load by querying the model itself.
EMBED_MODEL_DIMS: dict[str, int] = {
    "all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-m3": 1024,
}


def get_embed_model() -> str:
    """Configured embedding model id. HERMES_RAG_EMBED_MODEL overrides the
    default; unset → DEFAULT_EMBED_MODEL."""
    env = os.environ.get("HERMES_RAG_EMBED_MODEL")
    return env.strip() if env and env.strip() else DEFAULT_EMBED_MODEL


def get_embed_dim() -> int | None:
    """Optional manual dimension override via HERMES_RAG_EMBED_DIM. Returns
    None when unset/invalid so the Embedder auto-detects from the loaded
    model."""
    env = os.environ.get("HERMES_RAG_EMBED_DIM")
    if not env:
        return None
    try:
        n = int(env)
        return n if n > 0 else None
    except ValueError:
        return None


RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
COHERE_RERANK_MODEL = "rerank-english-v3.0"

# Contextual Retrieval (Phase 2) — opt-in via HERMES_RAG_CONTEXTUAL=1.
CONTEXTUAL_MAX_TOKENS = 150  # output cap for the prefix-generation LLM call
# Per-parent thread pool for contextual prefix generation. Anthropic's prompt
# cache is parent-scoped, so concurrent requests for chunks of the SAME parent
# all hit the same cache entry — concurrency multiplies throughput without
# multiplying token cost. Conservative default keeps tier-1 API users under
# the rate limit; bump for tier 3+.
CONTEXTUAL_CONCURRENCY = 4

# CRAG-lite (Phase 4) — opt-in via HERMES_RAG_CRAG=1.

# Ambient conversational memory (Phase 3) — opt-in via
# HERMES_RAG_AMBIENT_CONVO_MEMORY=1. Weights apply to current/previous/older
# user turn embeddings, normalized before mixing.
AMBIENT_CONVO_MEMORY_WEIGHTS = (1.0, 0.25, 0.1)
# Top-K pool that survives the lightweight ambient rerank.
AMBIENT_RERANK_POOL = 10
