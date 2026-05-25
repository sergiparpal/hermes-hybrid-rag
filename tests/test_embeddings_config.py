"""Configurable embeddings.

Covers HERMES_RAG_EMBED_MODEL / HERMES_RAG_EMBED_DIM, dim mismatch detection,
and the meta row written at index time.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from hybrid_rag import embeddings as embeddings_mod
from hybrid_rag.embeddings import Embedder
from hybrid_rag.engine import EngineLoadError, RAGEngine
from hybrid_rag.indexing import index_path
from hybrid_rag.storage import Store


# --- env-driven model selection ---

def test_default_model_is_bge_m3(monkeypatch):
    monkeypatch.delenv("HERMES_RAG_EMBED_MODEL", raising=False)
    assert embeddings_mod.get_embed_model() == "BAAI/bge-m3"
    e = Embedder()
    assert e.model_name == "BAAI/bge-m3"


def test_env_overrides_model(monkeypatch):
    monkeypatch.setenv("HERMES_RAG_EMBED_MODEL",
                       "sentence-transformers/all-MiniLM-L6-v2")
    assert (embeddings_mod.get_embed_model()
            == "sentence-transformers/all-MiniLM-L6-v2")
    e = Embedder()
    assert e.model_name == "sentence-transformers/all-MiniLM-L6-v2"


def test_dim_env_override(monkeypatch):
    monkeypatch.setenv("HERMES_RAG_EMBED_DIM", "768")
    assert embeddings_mod.get_embed_dim() == 768


def test_dim_env_invalid_returns_none(monkeypatch):
    monkeypatch.setenv("HERMES_RAG_EMBED_DIM", "not-a-number")
    assert embeddings_mod.get_embed_dim() is None


def test_known_models_have_known_dims():
    # The pre-registered table is what `Embedder.encode([])` relies on to
    # answer dim without loading the model — guard it explicitly.
    for known in ("BAAI/bge-m3", "all-MiniLM-L6-v2",
                  "sentence-transformers/all-MiniLM-L6-v2"):
        assert known in embeddings_mod.EMBED_MODEL_DIMS


# --- auto-detect dim on unknown model ---

def test_unknown_model_auto_detects_dim(monkeypatch):
    """A model id not in EMBED_MODEL_DIMS should auto-detect by loading."""
    fake_st = types.ModuleType("sentence_transformers")

    class _M:
        def __init__(self, _name):
            self.name = _name

        def get_sentence_embedding_dimension(self):
            return 512

        def encode(self, texts, **kw):
            return np.zeros((len(texts), 512), dtype=np.float32)

    fake_st.SentenceTransformer = _M
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)

    e = Embedder(model_name="some/private-model")
    assert e.dim is None  # not loaded yet
    out = e.encode(["hello"])
    assert out.shape == (1, 512)
    assert e.dim == 512


def test_empty_input_uses_resolved_dim(monkeypatch):
    """encode([]) must not silently return a (0, 384) array when the
    configured model has a different dim."""
    fake_st = types.ModuleType("sentence_transformers")

    class _M:
        def __init__(self, _name):
            pass

        def get_sentence_embedding_dimension(self):
            return 1024

        def encode(self, texts, **kw):
            return np.zeros((len(texts), 1024), dtype=np.float32)

    fake_st.SentenceTransformer = _M
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)

    e = Embedder(model_name="anything")
    out = e.encode([])
    assert out.shape == (0, 1024)


def test_encode_empty_raises_when_dim_cannot_be_resolved(monkeypatch):
    """If a model fails to expose its dim (returns None / 0) and the user
    didn't pin HERMES_RAG_EMBED_DIM, encode([]) must raise rather than
    silently hand back a (0, 0) array that breaks downstream dense search."""
    fake_st = types.ModuleType("sentence_transformers")

    class _NoDim:
        def __init__(self, _name):
            pass

        def get_sentence_embedding_dimension(self):
            return None

        def encode(self, texts, **kw):
            return np.zeros((len(texts), 0), dtype=np.float32)

    fake_st.SentenceTransformer = _NoDim
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)
    monkeypatch.delenv("HERMES_RAG_EMBED_DIM", raising=False)

    e = Embedder(model_name="mystery-model")
    with pytest.raises(RuntimeError, match="could not determine embedding dim"):
        e.encode([])


# --- meta row written at index time ---

def test_indexing_writes_embed_model_meta(tmp_data_dir, tmp_path, stub_embedder):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "x.md").write_text("# T\n\n## S1\nhello world here is content")

    store = Store()
    index_path(docs, store=store, embedder=stub_embedder)
    # Stub embedder is named "stub" — see conftest.StubEmbedder
    assert store.get_meta("embed_model") == "stub"
    assert int(store.get_meta("embed_dim")) == 32


# --- engine refuses on dim mismatch ---

class _DimMismatchEmbedder:
    """Stub that claims dim=64 so the engine compares 64 (configured) vs the
    actual .npz dim (=32 from StubEmbedder)."""
    DIM = 64

    def __init__(self):
        self.model_name = "stub-dim64"
        self.dim = 64

    def encode(self, texts, batch_size=64):
        return np.zeros((len(texts), self.DIM), dtype=np.float32)


def test_engine_refuses_dim_mismatch(tmp_data_dir, tmp_path, stub_embedder):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "x.md").write_text("# T\n\n## S1\nbody text long enough to chunk.")
    store = Store()
    index_path(docs, store=store, embedder=stub_embedder)

    # Now try to load with an embedder whose dim disagrees with the .npz.
    bad = _DimMismatchEmbedder()
    eng = RAGEngine(store=Store(), embedder=bad)
    with pytest.raises(EngineLoadError, match="dim"):
        eng._ensure_loaded()


def test_engine_refuses_meta_dim_disagreement(
    tmp_data_dir, tmp_path, stub_embedder,
):
    """A stale meta row that says dim=999 but .npz is shape[1]=32 must
    surface as EngineLoadError — points the user at --force."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "x.md").write_text("# T\n\n## S1\nbody text long enough to chunk.")
    store = Store()
    index_path(docs, store=store, embedder=stub_embedder)
    # Corrupt the meta row to simulate divergence.
    store.set_meta("embed_dim", "999")

    eng = RAGEngine(store=Store(), embedder=stub_embedder)
    with pytest.raises(EngineLoadError, match="dim"):
        eng._ensure_loaded()


def test_engine_warns_on_model_drift_same_dim(
    tmp_data_dir, tmp_path, stub_embedder, caplog,
):
    """Same dim but different model id → loud warning, but still loads."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "x.md").write_text("# T\n\n## S1\nbody text long enough to chunk.")
    store = Store()
    index_path(docs, store=store, embedder=stub_embedder)

    # Build a fresh stub embedder claiming a different model name but the
    # same dim. The engine must warn, not raise.
    other = type(stub_embedder)("other-stub")
    eng = RAGEngine(store=Store(), embedder=other)
    import logging
    with caplog.at_level(logging.WARNING):
        eng._ensure_loaded()
    msgs = [r.message for r in caplog.records]
    assert any("embedding-model drift" in m for m in msgs)
