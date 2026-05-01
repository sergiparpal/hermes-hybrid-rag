"""JSON Schema dicts for the three exposed tools.

Hermes wires these into the LLM's tool-use context. Keep the descriptions
agent-friendly — they're what the model reads when deciding whether to call.
"""
from __future__ import annotations

RAG_SEARCH = {
    "name": "rag_search",
    "description": (
        "Deep search of indexed user documents. Runs query expansion "
        "(paraphrases + HyDE), hybrid BM25+dense retrieval with second-level "
        "RRF fusion, parent rollup, and reranking. Returns ranked parent "
        "units with text and metadata."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language query.",
            },
            "k": {
                "type": "integer",
                "description": "Number of parents to return.",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}

RAG_DRILL_DOWN = {
    "name": "rag_drill_down",
    "description": (
        "Fetch the full ordered chunk list for a specific parent unit. Use "
        "after rag_search returned a promising parent and you need finer-"
        "grained text."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "parent_id": {
                "type": "integer",
                "description": "Parent ID returned by a previous rag_search call.",
            },
        },
        "required": ["parent_id"],
    },
}

RAG_LIST_SOURCES = {
    "name": "rag_list_sources",
    "description": (
        "List all indexed source documents with their parent and chunk "
        "counts. Useful to confirm coverage before deciding whether the "
        "corpus contains an answer."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
    },
}

ALL_SCHEMAS = (RAG_SEARCH, RAG_DRILL_DOWN, RAG_LIST_SOURCES)
