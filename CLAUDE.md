# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A **Hermes Agent plugin** that combines Advanced RAG (smart recursive chunking, hybrid BM25 + dense + RRF, query expansion, reranking) with Hierarchical RAG (embed small chunks, return their parent units). The deployable artifact is the inner `hybrid_rag/` package — that whole directory is what gets `rsync`'d to `~/.hermes/plugins/hybrid-rag/` on the runtime machine.

`README.md` is user-facing. `REQUIREMENTS.md` is the authoritative spec — the single source of truth for module responsibilities, DDL, invariants, and acceptance criteria. `HERMES_API.md` is the verified Hermes plugin signatures — when something at the adapter layer doesn't behave as expected, check it against `HERMES_API.md` before assuming bugs.

## Dev machine ≠ runtime machine

This separation drives almost every design choice — read it before doing anything substantive:

- **Canonical project root is `/home/sergi/Documentos/hybrid-rag/`.** Hermes runs elsewhere; `~/.hermes/plugins/hybrid-rag/` does not exist on this machine and must not be created here.
- **Light deps only on dev.** `numpy`, `rank_bm25`, `pyyaml`, `pytest`. Do **not** install `sentence-transformers`, `anthropic`, `cohere`, `pypdf` here — tests stub them out via `sys.modules` patching (`tests/conftest.py` `mock_anthropic`, `mock_cohere`, `mock_cross_encoder`, `StubEmbedder`).
- **Runtime state never appears in the repo.** `data/` (SQLite, `.npz`, BM25 pickle) is gitignored and is created lazily on the runtime machine on first index/use. Tests must always route through `tmp_data_dir` (sets `HERMES_RAG_DATA_DIR=tmp_path`) or pass `data_dir=tmp_path` to `Store`. The data-dir precedence in `config.get_data_dir()` (explicit arg > `HERMES_RAG_DATA_DIR` > default) is the only thing keeping tests from polluting a real index — keep it strict.
- **No real Hermes integration test on dev.** The adapter layer is verified manually post-deploy. Dev verifies pure logic only.

## Architecture: pure core + thin Hermes adapter

The codebase is split into two layers; the boundary matters because adapters are the only place Hermes API drift can break things.

**Pure modules** (no Hermes import, unit-tested directly): `chunking.py`, `parents.py`, `storage.py`, `embeddings.py`, `indexing.py`, `retrieval.py`, `expansion.py`, `rerank.py`, `engine.py`, `state.py`, `hooks.py`, `tools.py`, `schemas.py`, `cli.py`, `slash.py`, `config.py`.

**Hermes-coupled surface** (the only files to edit if Hermes' API drifts):
- `hybrid_rag/__init__.py::register(ctx)` — wires everything into `ctx.register_*`.
- `hybrid_rag/adapters.py` — closures that reshape pure handlers to whatever signature Hermes wants. Lazy imports inside each closure so a missing pure module fails loud during dev rather than at registration.

When Hermes signatures shift, the fix lives in those two files. Don't push Hermes shapes (e.g. `**kwargs`, `dict | None` returns) into the pure modules.

### Pipeline shape (engine + retrieval)

`RAGEngine` (engine.py) is a process-wide singleton. It holds the lazily-loaded BM25, embeddings ndarray, `_chunk_ids` list (row index → chunk_id, in canonical SQLite order), embedder, and store. Use `get_engine()` to access it. After a re-index, call `engine.reset()` so the next query reloads.

`rag_search` pipeline (`tools.tool_rag_search`):
```
query → expansion.expand_query → [q, p1, p2, p3, hyde]
      → per variant: retrieval.hybrid_search (BM25+dense, RRF, top-30 chunks)
      → second-level RRF on CHUNK rankings (not parent rankings) → top-30
      → retrieval.chunks_to_parents (MAX rollup) → ~10 parents
      → rerank.rerank (Cohere → local cross-encoder → identity fallback) → top-k
      → JSON
```

The ambient `pre_llm_call` hook is a lighter pipeline: hybrid_search → top-3 parents → 1500-token cap → 0.25 score gate. Returns `{"context": str}` or `None`.

### Invariants you must not break

- **Retrieval target is always a parent, never a chunk.** Chunks are the search space; parents are what the agent receives.
- **`embed_row` invariant.** Chunk row N in canonical SQLite ordering (`SELECT … ORDER BY parent_id, ord`) ↔ row N of `embeddings.npz`. `Store.bulk_update_embed_rows` writes row indices back after a rebuild. Indexing rebuilds the whole `.npz` and `bm25.pkl` from this canonical ordering and renames atomically (`.tmp` → final).
- **Identical tokenizer at index time and query time.** `retrieval._tokenize` is the single source. BM25 build and BM25 query must both go through it.
- **Parent rollup uses MAX of children's RRF scores**, not SUM/MEAN — avoids penalizing parents whose other children are unrelated.
- **Second-level RRF fuses chunk rankings**, not parent rankings — fusion benefits from all matched evidence; the parent rollup happens once afterward.
- **Hooks must never raise.** `hooks.ambient_pre_llm_call` wraps the entire body in `try/except Exception: return None`. `state.is_ambient_enabled()` fails open (errors → True). Tools wrap their bodies in `try/except` and return JSON-encoded errors, never raise.
- **Data dir precedence (config.py).** Explicit `Store(data_dir=…)` arg > `HERMES_RAG_DATA_DIR` env > default `~/.hermes/plugins/hybrid-rag/data/`. Don't add a fourth path.

### Optional dependency degradation

Three external deps are optional and the plugin must never block on a missing one:

- `COHERE_API_KEY` unset or `cohere` import fails → `rerank` falls back to local cross-encoder; if that also fails → identity (parents returned unchanged).
- `ANTHROPIC_API_KEY` unset or `anthropic` import fails → `expand_query` returns `[q]` (just the original).
- `pypdf` missing → indexing a `.pdf` raises `IndexingError` for that file but doesn't abort the run.

Tests cover each fallback path with mocked modules — keep that coverage.

## Common commands

```bash
pytest -q                            # full suite, target ≥30 tests, all should pass
pytest -q tests/test_retrieval.py    # one file
pytest -q tests/test_retrieval.py::test_name   # one test
python -c "from hybrid_rag import register"  # smoke-imports the plugin
python -c "import yaml; yaml.safe_load(open('hybrid_rag/plugin.yaml'))"  # validates manifest
```

Runtime-only commands (do **not** run on dev — they would pollute `~/.hermes/...`):
```bash
hermes rag index <path> [--force]
hermes rag stats
hermes rag clear
# In a Hermes session: /rag, /rag on, /rag off, /rag stats
```

## Conventions

- `*Plan*.md` is gitignored — feel free to keep scratch plans in the repo root without polluting commits.
- Tests for new pure modules go in `tests/test_<module>.py` and must run without heavy deps (use the existing stub/mock fixtures in `conftest.py`).
- `hybrid_rag/requirements.txt` is a copy of the repo-root `requirements.txt`; the duplication is intentional so a single `rsync -av hybrid_rag/ …` carries deps to runtime in one shot. If you change one, change both.
- Slash handler is `(raw_args: str) -> str | None` with **no kwargs** (see `HERMES_API.md` §4) — per-session toggle is impossible in v0.1, only a process-global `_default` key.
- `register_skill` requires `pathlib.Path`, not `str` (`HERMES_API.md` §5) — `Path(__file__).parent / "skills" / "rag-usage" / "SKILL.md"`.
