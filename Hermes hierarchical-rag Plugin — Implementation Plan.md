 Hermes hierarchical-rag Plugin — Implementation Plan

 Context

 We're building a Hermes Agent plugin that combines Advanced RAG (smart chunking, hybrid BM25+dense search via RRF, query expansion, reranking) with
 Hierarchical RAG (embed small ~300-char chunks, return their parent units — sections/pages/paragraph groups). The combination delivers precise matching
 (small chunks) with rich context (large parents).

 The plugin exposes:
 - An ambient pre_llm_call hook that injects relevant parent context every turn (lightweight, <300ms warm).
 - Three tools: rag_search (full pipeline w/ expansion + rerank), rag_drill_down (chunks for a parent), rag_list_sources.
 - CLI: hermes rag {index,stats,clear}.
 - Slash commands: /rag, /rag on|off, /rag stats.
 - A bundled skill teaching the agent when to use each retrieval mode.

 Constraints (from this conversation)

 1. Dev machine ≠ runtime machine. /home/sergi/advanced-rag is the canonical project root. Hermes runs elsewhere; ~/.hermes/plugins/hierarchical-rag/ does
 not exist on this machine and must not be created here.
 2. Runtime state never appears in the repo. data/ (SQLite, .npz, BM25 pickle) is created lazily on first index/use on the runtime machine. Gitignored. Never
  written to during dev.
 3. Data dir is overridable. Code resolves the data dir via env var HERMES_RAG_DATA_DIR, falling back to the default
 ~/.hermes/plugins/hierarchical-rag/data/. Tests use pytest's tmp_path and set the env var (or pass an explicit data_dir to constructors) — never touch the
 default location.
 4. Pure functions tested directly; Hermes adapter is thin and unverified locally. All handler logic lives in plain Python functions with their dependencies
 passed in. The Hermes integration layer (register(ctx) and the wrapper closures) is small enough to inspect by eye and is verified manually post-deployment.
 5. No real Hermes runtime here. Cannot run hermes rag index, cannot fire a real pre_llm_call. End-to-end smoke happens after deploy.
 6. Doc gaps still open. register_cli_command(name, help, setup_fn, handler_fn), register_command(name, handler, description), and SKILL.md frontmatter are
 inferred. We use an adapter layer so the inferred shapes are isolated and easy to refactor in one pass after the user inspects the actual Hermes source.
 7. Light dep install only. Install pyyaml, numpy, rank_bm25, pytest (and stdlib sqlite3). Stub sentence-transformers, anthropic, cohere, pypdf in tests via
 mocks/fakes. No torch download on the dev machine.

 Architecture

                 ┌─────────────────────────────────┐
                 │      User documents (md/txt/pdf) │
                 └────────────────┬────────────────┘
                                  │ hermes rag index
                                  ▼
    ┌──────────────────────────────────────────────────────┐
    │  Indexing pipeline                                    │
    │  parents.extract_*  →  chunking.recursive_split       │
    │  storage (SQLite: files/parents/chunks)               │
    │  embeddings.encode → embeddings.npz                   │
    │  rank_bm25 → bm25.pkl                                 │
    └──────────────────────────────────────────────────────┘
                                  │
                                  ▼
    ┌──────────────────────────────────────────────────────┐
    │  RAGEngine (singleton, lazy-loaded)                   │
    │  in-memory: BM25Okapi, np.ndarray of embeddings,      │
    │             chunk_id↔embed_row map, sqlite handle     │
    └──────────────────────────────────────────────────────┘
         │                                  │
         │ pre_llm_call (ambient)           │ rag_search/rag_drill_down (explicit)
         │  hybrid only, top-3 parents,     │  expansion + hybrid + rerank
         │  1500-tok cap, threshold gate    │  top-k parents
         ▼                                  ▼
    {"context": …}  or  None          JSON result string

 The retrieval target is always a parent (not a chunk). Small chunks are the search space; parents are the unit returned to the caller.

 Project structure

 /home/sergi/advanced-rag/
 ├── .git/
 ├── .gitignore                       # data/, __pycache__/, .pytest_cache/, *.pyc, .venv/, .mypy_cache/, *.egg-info
 ├── README.md                        # install, usage, architecture, deployment
 ├── pyproject.toml                   # entry-point install option
 ├── requirements.txt                 # runtime + dev deps, optional deps marked
 ├── hierarchical_rag/                # deployable plugin payload (what gets rsync'd)
 │   ├── plugin.yaml                  # Hermes manifest
 │   ├── __init__.py                  # register(ctx) — Hermes adapter only
 │   ├── adapters.py                  # closures wrapping pure handlers for ctx.*
 │   ├── config.py                    # paths (with HERMES_RAG_DATA_DIR override), tunables
 │   ├── chunking.py                  # recursive_split
 │   ├── parents.py                   # extract_md/txt/pdf, _enforce_parent_cap
 │   ├── storage.py                   # Store class — sqlite + npz + pickle
 │   ├── embeddings.py                # Embedder (lazy MiniLM)
 │   ├── indexing.py                  # index_path, _index_file, manifest diff
 │   ├── retrieval.py                 # hybrid_search, rrf_fuse, chunks_to_parents
 │   ├── expansion.py                 # expand_query (Anthropic SDK + fallback)
 │   ├── rerank.py                    # rerank (Cohere API or local cross-encoder)
 │   ├── engine.py                    # RAGEngine singleton, get_engine()
 │   ├── state.py                     # file-backed ambient toggle
 │   ├── hooks.py                     # ambient_pre_llm_call(...)
 │   ├── tools.py                     # tool_rag_search/drill_down/list_sources (pure)
 │   ├── schemas.py                   # JSON Schema dicts for the three tools
 │   ├── cli.py                       # setup_rag_parser(parser), handle_rag(args)
 │   ├── slash.py                     # slash_rag(rest) dispatcher
 │   └── skills/
 │       └── rag-usage/
 │           └── SKILL.md
 └── tests/
     ├── conftest.py                  # adds project root to sys.path; tmp_data_dir fixture
     ├── fixtures/
     │   └── docs/
     │       ├── alpha.md             # 3 sections w/ ## headings
     │       ├── beta.md              # no ## headings (forces fallback)
     │       └── gamma.txt            # ~30 paragraphs
     ├── test_chunking.py
     ├── test_parents.py
     ├── test_storage.py
     ├── test_indexing.py             # uses stub Embedder
     ├── test_retrieval.py
     ├── test_rrf.py
     ├── test_expansion.py            # mocked anthropic
     ├── test_rerank.py               # mocked cohere + cross-encoder
     ├── test_state.py
     ├── test_hook.py                 # uses stub engine
     ├── test_tools.py                # asserts JSON shape, never raises
     ├── test_cli.py                  # parses argv, runs handler with monkey-patched store
     ├── test_slash.py
     └── test_adapters.py             # fake ctx records register_* calls

 Deploying to runtime: rsync -av hierarchical_rag/ user@host:~/.hermes/plugins/hierarchical-rag/ — the trailing slash on the source flattens contents
 (plugin.yaml + .py + skills/) into the plugin dir at the layout Hermes expects.

 Module specs

 config.py

 - DEFAULT_DATA_DIR = Path.home() / ".hermes" / "plugins" / "hierarchical-rag" / "data"
 - get_data_dir() -> Path — checks HERMES_RAG_DATA_DIR env var first, falls back to default. Creates the dir lazily inside Store.__init__.
 - Functions: db_path(), npz_path(), bm25_path(), toggles_path().
 - Constants: MAX_CHUNK = 300, CHUNK_OVERLAP = 50, MAX_PARENT_CHARS = 8000, RRF_K = 60, AMBIENT_TOP_PARENTS = 3, AMBIENT_TOKEN_CAP = 1500,
 AMBIENT_SCORE_THRESHOLD = 0.25, EMBED_MODEL = "all-MiniLM-L6-v2", RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2", ANTHROPIC_MODEL =
 "claude-haiku-4-5-20251001".

 chunking.py

 recursive_split(text, max_size=300, overlap=50, separators=("\n\n","\n",". "," ","")) — greedy pack of split parts; recurses on remaining separator list
 when a single part overflows; falls through to fixed-size hard split with overlap when no separator works. Returns [] for whitespace-only input. Edge cases:
  text shorter than max → [text]; word longer than max → hard-split fallback.

 parents.py

 - extract_md(text) -> list[Parent] — split on ##  (level-2) lines; if zero level-2 headings, defer to extract_txt. Each parent's title is the literal
 heading line; kind="section".
 - extract_txt(text) -> list[Parent] — paragraph groups by \n\s*\n regex; greedy pack to ~2000 chars; kind="paragraph_group"; title None.
 - extract_pdf(path) -> list[Parent] — guarded import pypdf; one parent per page with kind="page", title=f"Page {i+1}". Raises IndexingError (caught
 upstream) if pypdf missing.
 - _enforce_parent_cap(parents, MAX_PARENT_CHARS=8000) — splits oversized parents on paragraph/line boundaries.

 storage.py

 Class Store(data_dir: Path | None = None) — accepts override or reads from env. Lazily creates the dir. Public API:
 - Store.connect() — opens conn, sets PRAGMA foreign_keys=ON.
 - init_schema(conn) — DDL below.
 - manifest_diff(disk_files: dict[path, stat]) -> {unchanged, changed, new, deleted}.
 - delete_files(file_ids) — cascades to parents → chunks.
 - bulk_insert_files/parents/chunks(...).
 - iter_chunks_ordered() -> Iterator[ChunkRow] — for full embed re-emit.
 - bulk_update_embed_rows(pairs).
 - get_chunk(chunk_id), get_parent(parent_id), chunks_for_parent(parent_id), list_sources(), stats() -> dict.
 - save_embeddings(npz_path, embeddings, chunk_ids) — writes via .npz.tmp + atomic rename.
 - load_embeddings(npz_path), load_bm25(pickle_path), save_bm25(...) — same atomic write.

 SQLite DDL

 PRAGMA foreign_keys = ON;

 CREATE TABLE IF NOT EXISTS files (
   id           INTEGER PRIMARY KEY,
   path         TEXT    NOT NULL UNIQUE,
   mtime        REAL    NOT NULL,
   size         INTEGER NOT NULL,
   content_hash TEXT    NOT NULL,
   filetype     TEXT    NOT NULL,
   indexed_at   REAL    NOT NULL
 );
 CREATE INDEX IF NOT EXISTS idx_files_path ON files(path);

 CREATE TABLE IF NOT EXISTS parents (
   id        INTEGER PRIMARY KEY,
   file_id   INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
   ord       INTEGER NOT NULL,
   kind      TEXT    NOT NULL,                -- 'section'|'page'|'paragraph_group'
   title     TEXT,
   page_no   INTEGER,
   text      TEXT    NOT NULL,
   char_len  INTEGER NOT NULL
 );
 CREATE INDEX IF NOT EXISTS idx_parents_file ON parents(file_id);

 CREATE TABLE IF NOT EXISTS chunks (
   id         INTEGER PRIMARY KEY,
   parent_id  INTEGER NOT NULL REFERENCES parents(id) ON DELETE CASCADE,
   ord        INTEGER NOT NULL,
   text       TEXT    NOT NULL,
   embed_row  INTEGER NOT NULL
 );
 CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_id);
 CREATE INDEX IF NOT EXISTS idx_chunks_embed_row ON chunks(embed_row);

 CREATE TABLE IF NOT EXISTS meta (
   key   TEXT PRIMARY KEY,
   value TEXT NOT NULL
 );

 embeddings.py

 Embedder(model_name) — __init__ is cheap; .encode(texts, batch_size=64) is the lazy load point. Returns L2-normalized np.float32 array. Wraps from
 sentence_transformers import SentenceTransformer inside the first encode() call. Tests inject a stub Embedder with deterministic vectors.

 retrieval.py

 - _tokenize(text) — lowercase, strip punctuation, whitespace split. Used identically at index time and query time.
 - rrf_fuse(rankings, k=60) -> dict[int, float] — Σ 1/(k + rank+1) over each ranking; ranks are 1-indexed.
 - hybrid_search(engine, query, k_pool=30) -> list[Hit] — runs BM25 + dense, RRF-fuses chunk rankings.
 - chunks_to_parents(engine, hits, top) -> list[ParentResult] — rolls up to parents using MAX of children's RRF score; sorts; returns top.
 - format_context(parents, token_cap=1500) -> str — truncates by char count (~4 chars/token), packs as ## <title>\n<text>\n\n.

 expansion.py

 expand_query(q) -> list[str]:
 - If anthropic import fails OR ANTHROPIC_API_KEY unset → return [q].
 - Else: prompt Haiku 4.5 to return JSON {"paraphrases": [3], "hyde": "..."}. Parse defensively (strip code fences). Return [q] + paraphrases + [hyde].
 - Any exception → log + return [q].

 rerank.py

 rerank(query, parents, top_k) -> list[ParentResult]:
 - If COHERE_API_KEY set: call cohere.Client.rerank(model="rerank-english-v3.0", query, documents, top_n). On exception, fall back to local.
 - Local: from sentence_transformers import CrossEncoder (module-level cache _CROSS), score (query, parent.text[:2000]) pairs, sort.
 - Sets parent.rerank_score on each result.

 engine.py

 Singleton RAGEngine with lazy load (see Plan-agent design §4). get_engine() returns the process-wide instance. Carries: _store, _bm25, _embeddings,
 _chunk_ids, _embedder. _ensure_loaded() is locked. reset() clears state for re-index.

 state.py

 - is_ambient_enabled(session_id=None) -> bool — reads toggles_path() JSON { "_default": true, "<sid>": bool }. Cache 1s in-process.
 - set_ambient(on: bool, session_id=None) — writes via tmp + rename. Without session, sets _default.
 - All errors → return True (fail open) so a corrupted toggle file doesn't lock the user out.

 hooks.py

 ambient_pre_llm_call(session_id, user_message, conversation_history, is_first_turn, model, platform, **kwargs) -> dict | None:
 - Wrap the entire body in try/except Exception: return None.
 - Cheap rejects: not state.is_ambient_enabled(session_id) → None; len(user_message.strip()) < 8 → None.
 - engine = get_engine() (lazy load).
 - hits = retrieval.hybrid_search(engine, user_message, k_pool=30).
 - parents = retrieval.chunks_to_parents(engine, hits, top=3).
 - If empty or parents[0].score < AMBIENT_SCORE_THRESHOLD → None.
 - context = retrieval.format_context(parents, token_cap=1500) (target ~1200 to leave room for other plugins).
 - Return {"context": context}.

 tools.py

 Each tool is tool_x(args: dict, store=None, engine=None) -> str — returns JSON, never raises. Default store=None/engine=None resolve to module-level
 singletons; tests pass explicit instances.
 - tool_rag_search(args): q = args["query"], k = args.get("k", 5). Pipeline = expand → per-variant hybrid → second-level RRF → roll-up to ~10 parents →
 rerank → top-k. JSON: {"results": [{"parent_id", "title", "source_path", "score", "rerank_score", "text", "page_no"}], "expansions_used": int}.
 - tool_rag_drill_down(args): pid = args["parent_id"]. Returns parent + ordered chunks. JSON: {"parent": {...}, "chunks": [...]}.
 - tool_rag_list_sources(args): ignores args. Returns {"sources": [{"path", "filetype", "indexed_at", "parent_count", "chunk_count"}]}.
 - All wrap their body in try/except Exception as e: return json.dumps({"error": str(e), "type": type(e).__name__}).

 schemas.py

 Plain JSON Schema dicts:
 RAG_SEARCH = {
     "name": "rag_search",
     "description": "Deep search of indexed user documents...",
     "parameters": {
         "type": "object",
         "properties": {
             "query": {"type": "string", "description": "Natural-language query."},
             "k": {"type": "integer", "description": "Number of parents to return.", "default": 5},
         },
         "required": ["query"],
     },
 }
 # RAG_DRILL_DOWN, RAG_LIST_SOURCES analogous.

 cli.py (best-guess argparse contract)

 def setup_rag_parser(parser):
     sub = parser.add_subparsers(dest="rag_cmd", required=True)
     p = sub.add_parser("index"); p.add_argument("path"); p.add_argument("--force", action="store_true")
     sub.add_parser("stats"); sub.add_parser("clear")

 def handle_rag(args):
     if args.rag_cmd == "index": ...
     elif args.rag_cmd == "stats": ...
     elif args.rag_cmd == "clear": ...
 Both functions are pure (return int exit codes, take dependencies via injection in tests). The Hermes wiring sits in adapters.py.

 slash.py

 def slash_rag(rest: str, *, state_mod=state, store_factory=Store) -> str:
     # dispatch on rest.split()[0] in {"", "on", "off", "stats"}
 Pure function: inject state_mod and store_factory for tests. The Hermes wrapper omits these to use defaults.

 adapters.py

 Thin closures that wrap pure functions to whatever shape Hermes wants. Each is one short function.
 def make_cli_setup():
     def _setup(parser):
         from .cli import setup_rag_parser
         setup_rag_parser(parser)
     return _setup

 def make_cli_handler():
     def _handle(args):
         from .cli import handle_rag
         return handle_rag(args)
     return _handle

 def make_slash_handler():
     def _slash(rest):  # adjust signature once Hermes API is verified
         from .slash import slash_rag
         return slash_rag(rest)
     return _slash

 def make_tool_wrapper(fn):
     def _tool(args, **kwargs):  # Hermes-documented signature
         return fn(args)
     return _tool

 def make_hook_wrapper():
     def _hook(session_id, user_message, conversation_history,
               is_first_turn, model, platform, **kwargs):
         from .hooks import ambient_pre_llm_call
         return ambient_pre_llm_call(session_id=session_id,
                                     user_message=user_message,
                                     conversation_history=conversation_history,
                                     is_first_turn=is_first_turn,
                                     model=model, platform=platform, **kwargs)
     return _hook

 __init__.py

 import os
 from . import schemas, tools as _tools, adapters

 def register(ctx):
     ctx.register_tool(name="rag_search", toolset="rag",
                       schema=schemas.RAG_SEARCH,
                       handler=adapters.make_tool_wrapper(_tools.tool_rag_search))
     ctx.register_tool(name="rag_drill_down", toolset="rag",
                       schema=schemas.RAG_DRILL_DOWN,
                       handler=adapters.make_tool_wrapper(_tools.tool_rag_drill_down))
     ctx.register_tool(name="rag_list_sources", toolset="rag",
                       schema=schemas.RAG_LIST_SOURCES,
                       handler=adapters.make_tool_wrapper(_tools.tool_rag_list_sources))
     ctx.register_hook("pre_llm_call", adapters.make_hook_wrapper())
     ctx.register_cli_command("rag", "Hierarchical RAG operations",
                              adapters.make_cli_setup(),
                              adapters.make_cli_handler())
     ctx.register_command("rag", adapters.make_slash_handler(),
                          "Hierarchical RAG control: /rag, /rag on|off, /rag stats")
     ctx.register_skill("rag-usage",
                        os.path.join(os.path.dirname(__file__), "skills", "rag-usage", "SKILL.md"))
 This file is the only Hermes-coupled module. If any inferred API turns out wrong, the fix is here + adapters.py.

 plugin.yaml

 name: hierarchical-rag
 version: 0.1.0
 description: Advanced + Hierarchical RAG over local documents (md/txt/pdf) — hybrid BM25+dense search with query expansion, reranking, and parent-unit
 retrieval.
 author: Sergi Parpal
 provides_tools:
   - rag_search
   - rag_drill_down
   - rag_list_sources
 provides_hooks:
   - pre_llm_call
 requires_env:
   - name: COHERE_API_KEY
     description: Optional. Enables Cohere reranker (rerank-english-v3.0). Without it, falls back to a local cross-encoder (~80MB download on first use).
     url: https://dashboard.cohere.com/api-keys
     secret: true
   - name: ANTHROPIC_API_KEY
     description: Optional. Enables LLM-based query expansion (paraphrases + HyDE) via claude-haiku-4-5. Without it, expansion is skipped and the original
 query is used.
     url: https://console.anthropic.com/
     secret: true
   - name: HERMES_RAG_DATA_DIR
     description: Optional. Override the data directory (defaults to ~/.hermes/plugins/hierarchical-rag/data). Useful for tests and isolated runs.
     secret: false

 skills/rag-usage/SKILL.md

 Frontmatter (assumed standard name/description keys; verify post-deploy):
 ---
 name: rag-usage
 description: Choose between ambient context, rag_search, and rag_drill_down when answering questions grounded in indexed documents.
 ---
 Body teaches: prefer ambient when sufficient; call rag_search for research/cross-doc; rag_drill_down after a promising parent; rag_list_sources to confirm
 coverage; cite as (<basename>, <title-or-page>); stop after two empty searches.

 Data flow specifics

 Indexing (recap)

 index_path(path, force=False) walks files, computes (mtime, size) cheap diff, hashes only on miss/change, deletes obsolete file rows (cascade),
 parents.extract_*, chunking.recursive_split, bulk inserts, rebuilds whole .npz and bm25.pkl from the canonical SQLite ordering, writes them via tmp+rename,
 calls engine.reset().

 Hybrid search

 1. BM25 over tokenized chunks (top 2*k_pool).
 2. Dense cosine embeddings @ q_vec (top 2*k_pool).
 3. RRF fuse the two ranked lists.
 4. Top k_pool chunk hits.

 Chunks → parents

 MAX of children's RRF scores (avoids penalizing parents whose other children are unrelated). Returns top parents in MAX-score order, with text, title, kind,
  page_no, source_path, score.

 rag_search second-level RRF

 Each query variant yields a top-30 chunk list. RRF-fuse all variants' lists → top-30 → roll up to ~10 parents → rerank → top-k. We RRF chunk rankings (not
 parent rankings) so the second-level fusion benefits from all the matched evidence.

 Ambient hook performance budget (warm)

 ┌──────────────────────────────────────┬───────────┐
 │                 Step                 │  Budget   │
 ├──────────────────────────────────────┼───────────┤
 │ state.is_ambient_enabled (cached 1s) │ <1ms      │
 ├──────────────────────────────────────┼───────────┤
 │ Heuristic chitchat reject            │ <1ms      │
 ├──────────────────────────────────────┼───────────┤
 │ Bi-encoder query embed (CPU)         │ 30–80ms   │
 ├──────────────────────────────────────┼───────────┤
 │ BM25 scoring                         │ 5–30ms    │
 ├──────────────────────────────────────┼───────────┤
 │ Cosine over embeddings               │ 5–20ms    │
 ├──────────────────────────────────────┼───────────┤
 │ Top-k argpartition × 2               │ <5ms      │
 ├──────────────────────────────────────┼───────────┤
 │ RRF + parent rollup                  │ <5ms      │
 ├──────────────────────────────────────┼───────────┤
 │ SQLite parent fetch (3 rows)         │ <5ms      │
 ├──────────────────────────────────────┼───────────┤
 │ Format + truncate                    │ <5ms      │
 ├──────────────────────────────────────┼───────────┤
 │ Total warm                           │ ~60–150ms │
 └──────────────────────────────────────┴───────────┘

 Cold first call after process start: dominated by MiniLM weight load (~1–3s on CPU). Mitigation: also register on_session_start hook in v0.2 to warm the
 engine. v0.1 documents the cold-start.

 Test strategy

 tests/conftest.py:
 - Adds project root to sys.path so import hierarchical_rag.tools works.
 - tmp_data_dir fixture sets HERMES_RAG_DATA_DIR=tmp_path and yields the path.
 - stub_embedder fixture: deterministic vectors (np.array([hash(t) % N / N] * 384) style, then L2-normalized).
 - fake_ctx fixture: records register_tool/hook/cli_command/command/skill calls into a dict.
 - mock_anthropic, mock_cohere, mock_cross_encoder: monkey-patches.

 What runs without heavy deps:
 - test_chunking, test_parents (md/txt only — pdf test skipped if pypdf missing).
 - test_storage, test_state.
 - test_indexing with stub Embedder.
 - test_retrieval, test_rrf with stub Embedder + tiny synthetic corpus.
 - test_expansion, test_rerank with mocks.
 - test_hook, test_tools with stub engine.
 - test_cli, test_slash, test_adapters — pure logic + fake ctx.

 What does not run here:
 - Real MiniLM embedding (no sentence-transformers installed).
 - Real cross-encoder rerank (same).
 - Real Cohere/Anthropic calls.
 - Real Hermes integration — verified manually after deploy.

 The build sequence below ensures pytest -q is green at every step.

 Build sequence

 1. Repo init: git init, .gitignore, README.md skeleton, requirements.txt, pyproject.toml, hierarchical_rag/ package dir, tests/conftest.py.
 2. config.py (paths + override) → smoke test.
 3. chunking.py + test_chunking.py.
 4. parents.py + test_parents.py (md/txt; pdf branch shipped but skipped in tests).
 5. storage.py + test_storage.py (in-memory SQLite + tmp data dir).
 6. embeddings.py (lazy wrapper, untested directly).
 7. indexing.py + test_indexing.py with stub Embedder; covers add/skip/modify/delete + force.
 8. retrieval.py + test_rrf.py + test_retrieval.py (synthetic corpus, stub embedder).
 9. expansion.py + test_expansion.py (mocked anthropic; assert fallback path on missing key/SDK).
 10. rerank.py + test_rerank.py (mocked cohere; mocked cross-encoder load).
 11. engine.py (composes 5–10).
 12. state.py + test_state.py.
 13. hooks.py + test_hook.py (stub engine; assert None paths and threshold gate).
 14. schemas.py, tools.py + test_tools.py (assert JSON shape and no-raise on bad input).
 15. cli.py + test_cli.py (argparse parsing + handler dispatch).
 16. slash.py + test_slash.py.
 17. adapters.py + test_adapters.py (fake ctx records calls).
 18. __init__.py register() — keep small; leans entirely on adapters.
 19. plugin.yaml.
 20. skills/rag-usage/SKILL.md.
 21. README.md (full): architecture, install, usage, deployment, troubleshooting.
 22. Final pass: pytest -q green; manual sanity scan of __init__.py and plugin.yaml.

 Dependency strategy

 requirements.txt:
 # Required at runtime (must be installed in the Hermes Python env)
 sentence-transformers>=3.0
 rank_bm25>=0.2.2
 numpy>=1.26
 pyyaml>=6.0

 # Optional — gracefully degrade if missing
 pypdf>=4.0     # PDF support
 anthropic>=0.40  # query expansion
 cohere>=5.0    # remote reranker

 # Dev only
 pytest>=8.0

 On the dev machine we install only: numpy, rank_bm25, pyyaml, pytest (already partly satisfied: pyyaml present). Stub the rest in tests.

 On the runtime machine, the user installs the full set inside Hermes's Python env (e.g., python -m pip install -r requirements.txt from inside
 ~/.hermes/plugins/hierarchical-rag/). First explicit search triggers MiniLM (~80MB) and (if no Cohere key) cross-encoder (~80MB) downloads.

 Deployment (README excerpt)

 Three supported flows, in order of recommendation:

 1. Direct directory deploy via rsync
 rsync -av --delete \
   --exclude='__pycache__' --exclude='*.pyc' \
   /home/sergi/advanced-rag/hierarchical_rag/ \
   user@runtime:~/.hermes/plugins/hierarchical-rag/
 ssh user@runtime 'cd ~/.hermes/plugins/hierarchical-rag && python -m pip install -r requirements.txt'
 1. Note: requirements.txt must also be synced — adjust includes accordingly, or place a copy inside hierarchical_rag/.
 2. git clone
 git clone <repo-url> ~/.hermes/plugins/hierarchical-rag-source
 ln -s ~/.hermes/plugins/hierarchical-rag-source/hierarchical_rag ~/.hermes/plugins/hierarchical-rag
 3. pip entry-point install (cleanest for distribution)
 pyproject.toml declares:
 [project.entry-points."hermes_agent.plugins"]
 hierarchical-rag = "hierarchical_rag"
 3. On runtime: pip install /path/to/clone. Hermes auto-discovers via the entry point.

 The data/ directory is created lazily by Store(get_data_dir()) on first index/use. Override with HERMES_RAG_DATA_DIR=/some/path to relocate runtime state.

 Critical files

 - /home/sergi/advanced-rag/hierarchical_rag/__init__.py — Hermes adapter; touched if any register_* API differs from inferred shape.
 - /home/sergi/advanced-rag/hierarchical_rag/adapters.py — closures wrapping pure handlers; second touch point for API drift.
 - /home/sergi/advanced-rag/hierarchical_rag/engine.py — singleton lifecycle; correctness of lazy load + reset is critical for hook latency.
 - /home/sergi/advanced-rag/hierarchical_rag/storage.py — atomic writes to .npz and bm25.pkl; embed_row maintenance.
 - /home/sergi/advanced-rag/hierarchical_rag/retrieval.py — RRF + chunks-to-parents rollup logic.
 - /home/sergi/advanced-rag/hierarchical_rag/hooks.py — must never raise; threshold gate; token cap.
 - /home/sergi/advanced-rag/hierarchical_rag/config.py — HERMES_RAG_DATA_DIR override is the only thing keeping tests from polluting the user's real index.

 Verification

 On the dev machine (after build):
 - pytest -q is green; ≥30 tests passing.
 - python -c "from hierarchical_rag import register" succeeds.
 - python -c "import yaml; print(yaml.safe_load(open('hierarchical_rag/plugin.yaml')))" validates manifest.
 - Manual: read __init__.py and adapters.py end-to-end; confirm imports and signatures by eye.

 On the runtime machine (after deploy):
 - hermes plugin list shows hierarchical-rag enabled.
 - mkdir test-corpus && echo "# Foo\n## Hello\nWorld" > test-corpus/a.md
 - hermes rag index ./test-corpus → reports 1 file, 1 parent, 1+ chunks.
 - /rag stats (in a Hermes session) shows the stats.
 - Send a message mentioning "Hello" or "World" — verify ambient context injection in Hermes logs/transcripts.
 - Call rag_search from the agent — verify reranked results returned.
 - rag_drill_down(parent_id=1) — verify chunks list.
 - Modify a.md, re-run hermes rag index ./test-corpus — only that file re-processed.
 - hermes rag clear (with confirmation prompt) — wipes data/.

 Risks & open assumptions

 1. register_cli_command shape (inferred) — argparse-based. Wrong-shape risk isolated to adapters.make_cli_setup/make_cli_handler.
 2. register_command shape (inferred) — (rest: str) -> str. Wrong-shape risk isolated to adapters.make_slash_handler.
 3. SKILL.md frontmatter (inferred) — name/description keys. One-file change if Hermes wants different keys.
 4. Slash handler has no session_id — we ship a process-global toggle (_default key). Per-session toggle is a v0.2 if the real API exposes session info via
 **kwargs.
 5. Cold-start latency — first ambient call eats MiniLM load (~1–3s on CPU). Documented; v0.2 can warm via on_session_start.
 6. Threshold tuning — 0.25 cosine is a placeholder. Add a tuning helper in v0.2.
 7. Embed cache for re-index speedup — deferred to v0.2 (current rebuild-from-scratch is O(N) but fine for personal corpora).
 8. Multi-plugin context budget contention — we cap at 1500 tokens per call but several plugins injecting at once can balloon total context. Reduce target to
  1200 tokens to leave room.
 9. Race on re-index during query — engine.reset() plus atomic .npz.tmp rename handles this; mid-rebuild reads see the previous valid index.
 10. No live Hermes integration test on dev machine. Manual verification required post-deploy. Adapter layer is small enough (~50 LOC) to inspect by eye.