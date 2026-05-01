"""hierarchical-rag — Hermes Agent plugin entry point.

Public surface is `register(ctx)`. All adapter shapes are isolated in
`adapters.py`; if Hermes' API drifts, the fix lives here + `adapters.py` only.
Signatures verified against the Hermes source — see HERMES_API.md.
"""
from __future__ import annotations

from pathlib import Path

from . import adapters, schemas, tools as _tools

__all__ = ["register"]


def register(ctx) -> None:
    ctx.register_tool(
        name="rag_search",
        toolset="rag",
        schema=schemas.RAG_SEARCH,
        handler=adapters.make_tool_wrapper(_tools.tool_rag_search),
    )
    ctx.register_tool(
        name="rag_drill_down",
        toolset="rag",
        schema=schemas.RAG_DRILL_DOWN,
        handler=adapters.make_tool_wrapper(_tools.tool_rag_drill_down),
    )
    ctx.register_tool(
        name="rag_list_sources",
        toolset="rag",
        schema=schemas.RAG_LIST_SOURCES,
        handler=adapters.make_tool_wrapper(_tools.tool_rag_list_sources),
    )

    ctx.register_hook("pre_llm_call", adapters.make_hook_wrapper())
    ctx.register_hook("on_session_start", adapters.make_session_warm_hook())

    ctx.register_cli_command(
        "rag",
        "Hierarchical RAG operations",
        adapters.make_cli_setup(),
        adapters.make_cli_handler(),
    )

    ctx.register_command(
        "rag",
        handler=adapters.make_slash_handler(),
        description="Hierarchical RAG control: /rag, /rag on|off, /rag stats",
        args_hint="on|off|stats",
    )

    ctx.register_skill(
        "rag-usage",
        Path(__file__).parent / "skills" / "rag-usage" / "SKILL.md",
        description="When to use ambient context, rag_search, and rag_drill_down for indexed documents.",
    )
