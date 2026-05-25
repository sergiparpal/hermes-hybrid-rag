import numpy as np
import pytest

from hybrid_rag.artifacts import ArtifactStore
from hybrid_rag.engine import (
    EngineLoadError,
    RAGEngine,
    get_engine,
    reset_for_tests,
    set_engine_for_tests,
)
from hybrid_rag.storage import Store


def _seed_chunks(store: Store, ids: list[int]) -> None:
    """Write `len(ids)` chunks into SQLite under one file, with embed_row set
    so iter_chunks_ordered yields rows in canonical order."""
    fid = store.bulk_insert_files([("/x.md", 0.0, 0, "h", "md", 0.0)])["/x.md"]
    pid = store.bulk_insert_parents([(fid, 0, "section", "T", None, "body", 4)])[0]
    rows = [(pid, i, f"chunk-{i}", 0, None, None, None) for i in range(len(ids))]
    actual_ids = store.bulk_insert_chunks(rows)
    # Force chunk ids to match what the caller wanted (bulk_insert assigns
    # autoincrement, so we patch via SQL).
    conn = store.connect()
    for new_id, old_id in zip(ids, actual_ids):
        conn.execute("UPDATE chunks SET id = ? WHERE id = ?", (new_id, old_id))
    conn.commit()
    # embed_row matches insertion order (canonical).
    store.bulk_update_embed_rows([(cid, row) for row, cid in enumerate(ids)])


def test_get_engine_is_singleton(tmp_data_dir):
    reset_for_tests()
    a = get_engine()
    b = get_engine()
    assert a is b
    reset_for_tests()


def test_set_engine_for_tests_replaces_singleton(tmp_data_dir, stub_embedder):
    eng = RAGEngine(store=Store(), embedder=stub_embedder)
    set_engine_for_tests(eng)
    assert get_engine() is eng
    reset_for_tests()


def test_ensure_loaded_reads_artifacts_once(tmp_data_dir, stub_embedder):
    """First _ensure_loaded() loads the .npz and rebuilds BM25 from SQLite;
    reset() drops cached state; the next call rebuilds again."""
    store = Store()
    arr = np.eye(3, 32, dtype=np.float32)
    chunk_ids = [10, 20, 30]
    _seed_chunks(store, chunk_ids)
    ArtifactStore(store.data_dir).save(arr)

    eng = RAGEngine(store=store, embedder=stub_embedder)
    eng._ensure_loaded()
    assert eng._chunk_ids == chunk_ids
    assert eng._embeddings.shape == (3, 32)
    # BM25 came from SQLite, not a pickle on disk — corpus_size must reflect
    # the live chunks count.
    assert eng._bm25 is not None
    assert getattr(eng._bm25, "corpus_size", None) == 3

    bm25_before = eng._bm25
    eng._ensure_loaded()
    assert eng._bm25 is bm25_before

    eng.reset()
    assert eng._bm25 is None
    assert eng._embeddings is None
    assert eng._chunk_ids == []
    assert eng._loaded is False

    eng._ensure_loaded()
    assert eng._chunk_ids == chunk_ids
    assert eng._bm25 is not None


def test_ensure_loaded_does_not_open_bm25_pickle(tmp_data_dir, stub_embedder):
    """Defense-in-depth: a leftover bm25.pkl from an old install (or a hostile
    plant) must never be deserialized. We rebuild from SQLite regardless."""
    store = Store()
    chunk_ids = [1, 2, 3]
    _seed_chunks(store, chunk_ids)
    arr = np.eye(3, 32, dtype=np.float32)
    ArtifactStore(store.data_dir).save(arr)
    # Plant a poisoned pickle: invalid bytes that would crash pickle.load if
    # anyone tried to deserialize it. If the engine reaches into it we'll
    # see an exception here.
    ArtifactStore(store.data_dir).legacy_bm25_path.write_bytes(
        b"\x80\x04this-is-not-a-pickle"
    )

    eng = RAGEngine(store=store, embedder=stub_embedder)
    eng._ensure_loaded()  # must not raise
    assert eng._bm25 is not None
    assert getattr(eng._bm25, "corpus_size", None) == 3


def test_ensure_loaded_with_missing_artifacts(tmp_data_dir, stub_embedder):
    eng = RAGEngine(store=Store(), embedder=stub_embedder)
    eng._ensure_loaded()
    assert eng._bm25 is None
    assert eng._embeddings is None
    assert eng._chunk_ids == []


def test_consistency_check_rejects_embedding_chunk_mismatch(tmp_data_dir, stub_embedder):
    """Embeddings array with N rows but SQLite has M chunks (N != M) must
    refuse to load with a clear EngineLoadError, not crash later in retrieval."""
    store = Store()
    _seed_chunks(store, [1, 2, 3])  # SQLite has 3 chunks
    arr = np.eye(5, 4, dtype=np.float32)  # but .npz has 5 rows
    ArtifactStore(store.data_dir).save(arr)

    eng = RAGEngine(store=store, embedder=stub_embedder)
    with pytest.raises(EngineLoadError, match="3 chunks"):
        eng._ensure_loaded()
    # State was scrubbed so a later call doesn't return half-loaded data.
    assert eng._embeddings is None
    assert eng._chunk_ids == []


