"""Contextual Retrieval.

Covers:
- HERMES_RAG_CONTEXTUAL default-off (contextual columns NULL).
- Prompt caching: the parent block carries cache_control: ephemeral.
- Anthropic error during indexing → contextual_prefix NULL for that chunk;
  the run completes.
- Lazy migration of pre-existing DBs to the contextual columns.
"""
from __future__ import annotations

import sqlite3
import sys
import types
from pathlib import Path

import pytest

from hybrid_rag import _anthropic, contextual
from hybrid_rag.indexing import index_path
from hybrid_rag.storage import Store

FIXTURES = Path(__file__).parent / "fixtures" / "docs"


@pytest.fixture(autouse=True)
def _reset_contextual_client():
    _anthropic.reset_for_tests()
    yield
    _anthropic.reset_for_tests()


# --- env-driven toggle ---

def test_contextual_off_by_default(monkeypatch):
    monkeypatch.delenv("HERMES_RAG_CONTEXTUAL", raising=False)
    assert contextual.is_contextual_enabled() is False


def test_contextual_on_via_env(monkeypatch):
    monkeypatch.setenv("HERMES_RAG_CONTEXTUAL", "1")
    assert contextual.is_contextual_enabled() is True


def test_contextual_truthy_values(monkeypatch):
    for val in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("HERMES_RAG_CONTEXTUAL", val)
        assert contextual.is_contextual_enabled() is True
    monkeypatch.setenv("HERMES_RAG_CONTEXTUAL", "0")
    assert contextual.is_contextual_enabled() is False


# --- graceful fallback when no Anthropic ---

def test_returns_none_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert contextual.generate_contextual_prefix("parent", "chunk") is None


def test_returns_none_when_sdk_missing(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setitem(sys.modules, "anthropic", None)
    assert contextual.generate_contextual_prefix("parent", "chunk") is None


def test_returns_none_for_empty_inputs(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert contextual.generate_contextual_prefix("", "chunk") is None
    assert contextual.generate_contextual_prefix("parent", "") is None


# --- prompt caching is requested ---

def _install_recording_anthropic(monkeypatch, text="this is the context"):
    """Install a fake `anthropic` module that records the request kwargs."""
    mod = types.ModuleType("anthropic")
    recorder = {"calls": []}

    class _Msg:
        def __init__(self, t):
            self.content = [types.SimpleNamespace(text=t)]

    class _Messages:
        def create(self, **kwargs):
            recorder["calls"].append(kwargs)
            return _Msg(text)

    class _Client:
        def __init__(self, *a, **kw):
            self.messages = _Messages()

    mod.Anthropic = _Client
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _anthropic.reset_for_tests()
    return recorder


def test_prompt_cache_control_on_parent_block(monkeypatch):
    """The Anthropic-style cache layer is what keeps indexing cost sane —
    if the parent block stops carrying cache_control, regressions go
    silent and tokens triple. Pin this explicitly."""
    recorder = _install_recording_anthropic(monkeypatch, text="prefix here")

    out = contextual.generate_contextual_prefix(
        "this is the whole parent document text",
        "and this is one chunk inside it",
    )
    assert out == "prefix here"
    assert len(recorder["calls"]) == 1
    msg = recorder["calls"][0]["messages"][0]
    content = msg["content"]
    # First block carries the parent text and cache_control:
    assert content[0]["type"] == "text"
    assert "parent document" in content[0]["text"]
    assert content[0].get("cache_control") == {"type": "ephemeral"}
    # Second block carries the chunk and does NOT cache:
    assert "chunk" in content[1]["text"].lower()
    assert "cache_control" not in content[1]


def test_silent_failure_on_sdk_exception(monkeypatch):
    mod = types.ModuleType("anthropic")

    class _Messages:
        def create(self, **kw):
            raise RuntimeError("api down")

    class _Client:
        def __init__(self, *a, **kw):
            self.messages = _Messages()

    mod.Anthropic = _Client
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    _anthropic.reset_for_tests()
    assert contextual.generate_contextual_prefix("p", "c") is None


# --- indexing integration: default-off matches v0.1 ---

def _stage(tmp_path: Path) -> Path:
    out = tmp_path / "docs"
    out.mkdir()
    for n in ("alpha.md", "beta.md", "gamma.txt"):
        (out / n).write_text((FIXTURES / n).read_text())
    return out


def _chunk_rows(store: Store) -> list[sqlite3.Row]:
    conn = store.connect()
    return conn.execute(
        "SELECT text, contextual_prefix, text_for_embedding, text_for_bm25 "
        "FROM chunks"
    ).fetchall()


def test_contextual_off_leaves_new_columns_null(
    tmp_data_dir, tmp_path, stub_embedder, monkeypatch,
):
    monkeypatch.delenv("HERMES_RAG_CONTEXTUAL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    docs = _stage(tmp_path)
    store = Store()
    index_path(docs, store=store, embedder=stub_embedder)
    rows = _chunk_rows(store)
    assert rows
    for r in rows:
        assert r["contextual_prefix"] is None
        assert r["text_for_embedding"] is None
        assert r["text_for_bm25"] is None


def test_contextual_on_writes_prefix_and_composed_text(
    tmp_data_dir, tmp_path, stub_embedder, monkeypatch,
):
    monkeypatch.setenv("HERMES_RAG_CONTEXTUAL", "1")
    _install_recording_anthropic(monkeypatch, text="situated context")

    docs = _stage(tmp_path)
    store = Store()
    index_path(docs, store=store, embedder=stub_embedder)
    rows = _chunk_rows(store)
    assert rows
    populated = [r for r in rows if r["contextual_prefix"]]
    assert populated, "expected at least one chunk to receive a contextual prefix"
    for r in populated:
        assert r["contextual_prefix"] == "situated context"
        # Composition rule: prefix + "\n\n" + chunk.
        assert r["text_for_embedding"] == "situated context\n\n" + r["text"]
        assert r["text_for_bm25"] == r["text_for_embedding"]


def test_anthropic_failure_during_indexing_does_not_abort(
    tmp_data_dir, tmp_path, stub_embedder, monkeypatch,
):
    """Per spec: a flaky Anthropic call must not abort the run. The chunk
    still gets indexed with contextual_prefix = NULL."""
    monkeypatch.setenv("HERMES_RAG_CONTEXTUAL", "1")
    mod = types.ModuleType("anthropic")

    class _Messages:
        def create(self, **kw):
            raise RuntimeError("synthetic rate limit")

    class _Client:
        def __init__(self, *a, **kw):
            self.messages = _Messages()

    mod.Anthropic = _Client
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    _anthropic.reset_for_tests()

    docs = _stage(tmp_path)
    store = Store()
    summary = index_path(docs, store=store, embedder=stub_embedder)
    # Run completes:
    assert summary["files_added_or_updated"] == 3
    # Every chunk has NULL prefix because every call failed:
    rows = _chunk_rows(store)
    assert rows
    for r in rows:
        assert r["contextual_prefix"] is None


# --- lazy migration of existing DBs ---

def test_lazy_schema_migration_for_pre_phase2_db(tmp_data_dir):
    """Open a Store, drop the Phase-2 columns, reopen, and verify the
    migration restores them. Simulates upgrading an existing on-disk DB."""
    store = Store()
    store.connect()  # initial init creates new schema
    conn = store._conn
    # Build a pre-Phase-2 table shape by re-creating chunks without the new
    # columns.
    conn.executescript(
        """
        DROP TABLE chunks;
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            parent_id INTEGER NOT NULL,
            ord INTEGER NOT NULL,
            text TEXT NOT NULL,
            embed_row INTEGER NOT NULL
        );
        """
    )
    conn.commit()
    cols_before = {r[1] for r in conn.execute("PRAGMA table_info(chunks)")}
    assert "contextual_prefix" not in cols_before
    store.close()

    # Reopen — the lazy migration should add the missing columns.
    store2 = Store()
    store2.connect()
    cols_after = {
        r[1] for r in store2._conn.execute("PRAGMA table_info(chunks)")
    }
    assert {"contextual_prefix", "text_for_embedding", "text_for_bm25"} <= cols_after


# --- retrieval honors text_for_embedding when set ---

def test_rebuild_uses_text_for_embedding(
    tmp_data_dir, tmp_path, stub_embedder, monkeypatch,
):
    """When the .npz is rebuilt, contextual rows feed `text_for_embedding`
    to the embedder, not raw `text`. Verifiable by snooping the embedder."""
    monkeypatch.setenv("HERMES_RAG_CONTEXTUAL", "1")
    _install_recording_anthropic(monkeypatch, text="CTXSENTINEL")

    seen_texts: list[str] = []

    class _SpyEmbedder:
        DIM = 32
        model_name = "spy"
        dim = 32

        def encode(self, texts, batch_size: int = 64):
            seen_texts.extend(texts)
            return stub_embedder.encode(texts)

    docs = _stage(tmp_path)
    store = Store()
    index_path(docs, store=store, embedder=_SpyEmbedder())
    # Every chunk's input to the embedder must start with the contextual
    # sentinel because the recording-anthropic always returns "CTXSENTINEL".
    assert seen_texts
    assert all(t.startswith("CTXSENTINEL\n\n") for t in seen_texts)


def test_contextual_generation_uses_thread_pool(
    tmp_data_dir, tmp_path, stub_embedder, monkeypatch,
):
    """Contextual prefixes for chunks of one parent are generated concurrently
    so a large parent doesn't pay N × per-call latency. Verifiable by
    counting concurrent in-flight calls."""
    import threading
    import time
    monkeypatch.setenv("HERMES_RAG_CONTEXTUAL", "1")

    in_flight = {"current": 0, "peak": 0}
    lock = threading.Lock()

    mod = types.ModuleType("anthropic")

    class _Msg:
        def __init__(self, t):
            self.content = [types.SimpleNamespace(text=t)]

    class _Messages:
        def create(self, **kw):
            with lock:
                in_flight["current"] += 1
                in_flight["peak"] = max(in_flight["peak"], in_flight["current"])
            try:
                # Hold the call long enough for siblings to pile up if the
                # caller is running serially.
                time.sleep(0.05)
                return _Msg("prefix")
            finally:
                with lock:
                    in_flight["current"] -= 1

    class _Client:
        def __init__(self, *a, **kw):
            self.messages = _Messages()

    mod.Anthropic = _Client
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _anthropic.reset_for_tests()

    # Build a single doc whose parent will split into many chunks so the
    # thread pool has something to parallelize.
    docs = tmp_path / "docs"
    docs.mkdir()
    long_section = " ".join(f"word{i}" for i in range(2000))
    (docs / "big.md").write_text(f"# T\n\n## Section\n{long_section}\n")

    store = Store()
    index_path(docs, store=store, embedder=stub_embedder)
    # Concurrency cap is 4 in config; we only need to confirm we ran more
    # than 1 in parallel to prove the serial loop is gone.
    assert in_flight["peak"] >= 2, (
        f"contextual generation appears serial; peak in-flight = "
        f"{in_flight['peak']}"
    )


def test_contextual_skips_whitespace_only_pieces(
    tmp_data_dir, tmp_path, stub_embedder, monkeypatch,
):
    """A parent whose recursive_split happens to yield a whitespace-only
    piece must not produce an empty chunk row."""
    monkeypatch.delenv("HERMES_RAG_CONTEXTUAL", raising=False)

    docs = tmp_path / "docs"
    docs.mkdir()
    # A section header with no body — recursive_split returns [], falling
    # back to [parent.text] which itself may be whitespace.
    (docs / "empty.md").write_text("# T\n\n## Section\n\n   \n\t\n")

    store = Store()
    summary = index_path(docs, store=store, embedder=stub_embedder)
    # File is indexed, but the empty section produces no chunk rows.
    assert summary["files_added_or_updated"] == 1
    # Every chunk that *did* land must have non-empty text.
    rows = store.connect().execute(
        "SELECT text FROM chunks"
    ).fetchall()
    for r in rows:
        assert r["text"].strip(), "indexer leaked an empty/whitespace chunk"
