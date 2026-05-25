"""Shared data classes. Living here (not inside retrieval/storage/parents)
breaks the fan-in that would otherwise force every consumer through one of
those modules. Nothing in this file imports any other ``hybrid_rag``
module — keep it that way.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Hit:
    """A single chunk-level retrieval result. `score` is the fused RRF score."""
    chunk_id: int
    score: float
    parent_id: int


@dataclass
class ParentResult:
    """A parent unit returned to the caller. Carries the source path and the
    optional post-rerank score; consumers read `effective_score` to gate on a
    threshold without branching on whether the reranker ran."""
    parent_id: int
    title: str | None
    kind: str
    page_no: int | None
    text: str
    source_path: str
    score: float
    rerank_score: float | None = None

    @property
    def effective_score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.score


@dataclass
class ChunkRow:
    """One row from the chunks table in canonical order. The two `text_for_*`
    fields are populated only when contextual retrieval ran at index time."""
    id: int
    parent_id: int
    ord: int
    text: str
    embed_row: int
    contextual_prefix: str | None = None
    text_for_embedding: str | None = None
    text_for_bm25: str | None = None

    @property
    def effective_embedding_text(self) -> str:
        """`prefix + chunk` when contextual retrieval is active, else the raw
        chunk. Single source of truth so embeddings rebuild and any future
        inspectors agree."""
        return self.text_for_embedding or self.text

    @property
    def effective_bm25_text(self) -> str:
        return self.text_for_bm25 or self.text


@dataclass
class Parent:
    """A parent unit extracted from a source file. The four `kind` values
    ('section', 'page', 'paragraph_group', 'preamble') reflect the four
    extractors we ship; new extractors are free to add their own."""
    kind: str
    title: str | None
    text: str
    page_no: int | None = None
