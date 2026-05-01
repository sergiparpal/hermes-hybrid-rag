# HERMES_API.md — verified Hermes plugin signatures

Verified against `hermes-agent` source (`hermes_cli/plugins.py`,
`run_agent.py`, `agent/skill_utils.py`, `cli.py`). Each section below cites the
exact source location.

The `PluginContext` class is defined in `hermes_cli/plugins.py:233`.

---

## 1. `ctx.register_tool(...)`  ✅ compatible

Source: `hermes_cli/plugins.py:242`

```python
def register_tool(
    self,
    name: str,
    toolset: str,
    schema: dict,
    handler: Callable,
    check_fn: Callable | None = None,
    requires_env: list | None = None,
    is_async: bool = False,
    description: str = "",
    emoji: str = "",
) -> None
```

`(name, toolset, schema, handler)` is the minimum. Extra optional kwargs:

- `check_fn` — callable to gate availability (e.g., "ffmpeg installed?").
- `requires_env` — list of env var names that must be set.
- `is_async` — set `True` if `handler` is async.
- `description` / `emoji` — UI hints.

**Handler invocation**: tools are dispatched via `tools.registry.registry`.
Handler signature: `handler(args: dict, **kwargs) -> str` (JSON string).

## 2. `ctx.register_hook(event, fn)`  ✅ matches

Source: `hermes_cli/plugins.py:528`. Valid event names: `hermes_cli/plugins.py:78`.

```python
def register_hook(self, hook_name: str, callback: Callable) -> None
```

Hooks are invoked as `cb(**kwargs)` (`hermes_cli/plugins.py:1079`).

### `pre_llm_call` contract

Source: `run_agent.py:10619`. Kwargs passed:

```python
session_id:           str
user_message:         str
conversation_history: list[dict]
is_first_turn:        bool
model:                str
platform:             str
sender_id:            str
```

Return contract (`hermes_cli/plugins.py:1063-1073`):

- `None` to inject nothing.
- `{"context": "<text>"}` or a plain `"<text>"` string — equivalent.
- Context is **always injected into the user message** (never the system
  prompt — preserves prompt-cache prefix). Ephemeral, not persisted to
  session DB.

### `on_session_start` contract  ✅ exists

Source: `run_agent.py:10519`. Fired only on **brand-new** sessions (not on
continuation). Kwargs:

```python
session_id: str
model:      str
platform:   str
```

Used by this plugin to warm the engine in v0.1 (background thread, never
blocks).

### Other valid hooks (full list)

`pre_tool_call`, `post_tool_call`, `transform_terminal_output`,
`transform_tool_result`, `pre_llm_call`, `post_llm_call`, `pre_api_request`,
`post_api_request`, `on_session_start`, `on_session_end`,
`on_session_finalize`, `on_session_reset`, `subagent_stop`,
`pre_gateway_dispatch`, `pre_approval_request`, `post_approval_response`.

## 3. `ctx.register_cli_command(...)`  ✅ compatible

Source: `hermes_cli/plugins.py:301`

```python
def register_cli_command(
    self,
    name: str,
    help: str,
    setup_fn: Callable,
    handler_fn: Callable | None = None,
    description: str = "",
) -> None
```

`setup_fn(parser: argparse.ArgumentParser) -> None` — adds args/sub-commands.
`handler_fn(args: argparse.Namespace) -> int` — exit code. Optional; if
provided, set as default dispatch via `set_defaults(func=handler_fn)`.

## 4. `ctx.register_command(...)` (slash)  ⚠️ no kwargs

Source: `hermes_cli/plugins.py:326`. Dispatch site: `cli.py:6599`.

```python
def register_command(
    self,
    name: str,
    handler: Callable,
    description: str = "",
    args_hint: str = "",
) -> None
```

**Slash handler signature**: `handler(raw_args: str) -> str | None`.
Single positional arg only — **no `**kwargs`** are passed by Hermes
(`cli.py:6599` calls `plugin_handler(user_args)`).

Consequence: per-session toggle is impossible (no `session_id` here);
process-global toggle is the only option in v0.1.

`args_hint` (e.g., `"on|off|stats"`) is shown as the parameter field in
gateway adapters like Discord's native slash command picker.

## 5. `ctx.register_skill(name, path, description="")`  🔴 path must be `Path`

Source: `hermes_cli/plugins.py:547`

```python
def register_skill(
    self,
    name: str,
    path: Path,                          # ← pathlib.Path, NOT str
    description: str = "",
) -> None
```

The implementation calls `path.exists()`, which fails on a string with
`AttributeError`. Always pass `Path(__file__).parent / "skills" / "rag-usage" / "SKILL.md"`.

Skill name constraints: `[a-zA-Z0-9_-]+`, no `:`. Hermes auto-prefixes the
plugin name to make a qualified id like `advanced-rag:rag-usage`.

`SKILL.md` frontmatter keys actually read by Hermes (`agent/skill_utils.py`):

- `name` (line 351) — falls back to directory name.
- `description` (line 428).
- `platforms` (line 104) — optional, e.g., `[linux, macos]`.
- `metadata` (lines 251, 285) — optional nested config.

## 6. `plugin.yaml` schema

Source: `hermes_cli/plugins.py:892` (the parser).

Keys actually parsed:

```yaml
name: advanced-rag           # required
version: "0.1.0"
description: "..."
author: "Sergi Parpal"
kind: standalone                 # default; alternatives: backend, exclusive, platform
requires_env:                    # convention: list of strings
  - COHERE_API_KEY
  - ANTHROPIC_API_KEY
  - HERMES_RAG_DATA_DIR
provides_tools:                  # informational only
  - rag_search
  - rag_drill_down
  - rag_list_sources
provides_hooks:                  # informational only
  - pre_llm_call
  - on_session_start
```

`provides_tools` and `provides_hooks` are advisory (used by `hermes plugin
list`). Real registration happens via `ctx.register_*` calls in
`register(ctx)`.

`requires_env` accepts dicts in addition to strings (the dataclass type is
`List[Union[str, Dict[str, Any]]]`) but no codebase consumer reads the dict
fields. Convention is strings; document each env var in the README.

## 7. Discovery

- User plugins: `~/.hermes/plugins/<dirname>/` (via `get_hermes_home() / "plugins"`).
  Manifest must be `<dirname>/plugin.yaml` and entry must be
  `<dirname>/__init__.py::register(ctx)`.
- Entry-point group: `hermes_agent.plugins` (matches `pyproject.toml`).
- Discovery order (later overrides earlier on key collision): bundled → user → project → entry-point.

---

## Summary table

| API | Verified status |
|---|---|
| `register_tool` | ✅ Compatible. Extra optional kwargs: `check_fn`, `requires_env`, `is_async`, `description`, `emoji`. |
| `register_hook("pre_llm_call", fn)` | ✅ Match. Real kwargs include extra `sender_id` (absorbed by `**kwargs`). Return `{"context": str}` or plain string. |
| `register_hook("on_session_start", fn)` | ✅ **Available** — kwargs `session_id, model, platform`. Fires only on new sessions. Used in v0.1 for engine warming. |
| `register_cli_command` | ✅ Compatible. `handler_fn` is optional; new `description` kwarg. |
| `register_command` (slash) | ⚠️ Handler is `(raw_args: str) -> str \| None` — **no kwargs**. Per-session toggle impossible. |
| `register_skill` | 🔴 `path` must be `pathlib.Path`, not `str`. New `description` kwarg available. |
