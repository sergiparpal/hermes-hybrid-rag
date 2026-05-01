from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from hierarchical_rag.retrieval import (
    _tokenize,
    chunks_to_parents,
    format_context,
    hybrid_search,
)


# --- shared fixtures ---

@dataclass
class _Parent:
    id: int
    title: str
    kind: str
    text: str
    page_no: int | None = None
    source_path: str = "/x.md"


class FakeStore:
    def __init__(self, parents, chunk_to_parent):
        self.parents = {p.id: p for p in parents}
        self.chunk_to_parent = chunk_to_parent

    def parent_id_for_chunk(self, chunk_id):
        return self.chunk_to_parent.get(chunk_id)

    def get_parent(self, parent_id):
        p = self.parents.get(parent_id)
        if p is None:
            return None
        return {
            "id": p.id, "title": p.title, "kind": p.kind, "text": p.text,
            "page_no": p.page_no, "source_path": p.source_path,
        }


class FakeEngine:
    def __init__(self, chunk_texts, chunk_ids, parent_ids, embedder, parents):
        self._chunk_ids = chunk_ids
        if chunk_texts:
            tokenized = [_tokenize(t) for t in chunk_texts]
            self._bm25 = BM25Okapi(tokenized)
            self._embeddings = embedder.encode(chunk_texts)
        else:
            self._bm25 = None
            self._embeddings = np.zeros((0, 32), dtype=np.float32)
        self._embedder = embedder
        self._store = FakeStore(parents,
                                {cid: pid for cid, pid in zip(chunk_ids, parent_ids)})


# --- tokenizer parity ---

def test_tokenizer_index_query_parity():
    assert _tokenize("Cosmic Rays!") == _tokenize("cosmic rays")
    assert _tokenize("Hello, world.") == _tokenize("hello world")


def test_tokenizer_strips_punctuation():
    assert _tokenize("foo-bar baz_qux 42") == ["foo", "bar", "baz", "qux", "42"]


# --- hybrid search ---

def _build_engine(stub_embedder):
    parents = [
        _Parent(id=1, title="Cosmic rays", kind="section",
                text="Cosmic rays are high energy particles from space."),
        _Parent(id=2, title="Pasta recipes", kind="section",
                text="Cacio e pepe needs pasta pecorino and black pepper."),
        _Parent(id=3, title="Quick brown fox", kind="section",
                text="The quick brown fox jumps over the lazy dog."),
    ]
    chunk_texts = [
        "Cosmic rays are high energy particles from outer space.",
        "Cosmic ray showers strike the upper atmosphere.",
        "Cacio e pepe needs pasta pecorino and black pepper.",
        "Pasta recipes include carbonara and amatriciana.",
        "The quick brown fox jumps over the lazy dog.",
        "Brown foxes are fast and lazy dogs are slow.",
    ]
    chunk_ids = [101, 102, 201, 202, 301, 302]
    chunk_parents = [1, 1, 2, 2, 3, 3]
    eng = FakeEngine(chunk_texts, chunk_ids, chunk_parents, stub_embedder, parents)
    return eng


def test_hybrid_search_returns_relevant_top_hits(stub_embedder):
    engine = _build_engine(stub_embedder)
    hits = hybrid_search(engine, "cosmic ray particles", k_pool=5)
    assert hits, "expected at least one hit"
    # the top hit should belong to parent 1 (cosmic rays)
    assert hits[0].parent_id == 1


def test_hybrid_search_returns_empty_when_corpus_empty(stub_embedder):
    eng = FakeEngine([], [], [], stub_embedder, [])
    assert hybrid_search(eng, "anything", k_pool=5) == []


# --- parent rollup MAX vs SUM ---

def test_max_rollup_picks_strongest_chunk(stub_embedder):
    engine = _build_engine(stub_embedder)
    hits = hybrid_search(engine, "cosmic ray", k_pool=10)
    parents = chunks_to_parents(engine, hits, top=3)
    assert parents
    assert parents[0].parent_id == 1
    # MAX rollup: parent score must equal the highest hit score among its chunks
    cosmic_hits = [h.score for h in hits if h.parent_id == 1]
    assert abs(parents[0].score - max(cosmic_hits)) < 1e-9


# --- context formatting ---

def test_format_context_packs_within_cap():
    from hierarchical_rag.retrieval import ParentResult

    parents = [
        ParentResult(parent_id=1, title="A", kind="section", page_no=None,
                     text="alpha " * 100, source_path="/x.md", score=1.0),
        ParentResult(parent_id=2, title="B", kind="section", page_no=None,
                     text="beta " * 100, source_path="/y.md", score=0.9),
    ]
    out = format_context(parents, token_cap=200)
    # 200 tok cap → ~800 char budget; two 500-char blocks won't both fit fully
    assert len(out) <= 800 + 50  # small slack for header lines / truncation marker


def test_format_context_empty_input_returns_empty():
    assert format_context([], token_cap=1500) == ""
