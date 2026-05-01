import numpy as np

from advanced_rag.engine import RAGEngine, get_engine, set_engine_for_tests
from advanced_rag.storage import Store


def test_get_engine_is_singleton(tmp_data_dir):
    set_engine_for_tests(None)
    a = get_engine()
    b = get_engine()
    assert a is b
    set_engine_for_tests(None)


def test_set_engine_for_tests_replaces_singleton(tmp_data_dir, stub_embedder):
    eng = RAGEngine(store=Store(), embedder=stub_embedder)
    set_engine_for_tests(eng)
    assert get_engine() is eng
    set_engine_for_tests(None)


def test_ensure_loaded_reads_artifacts_once(tmp_data_dir, stub_embedder):
    """First _ensure_loaded() loads artifacts; reset() drops them; the next
    call reloads."""
    store = Store()
    arr = np.eye(3, 4, dtype=np.float32)
    chunk_ids = [10, 20, 30]
    store.save_embeddings(store.npz_path, arr, chunk_ids)
    # Save a tiny BM25-like sentinel — engine just unpickles whatever's there.
    store.save_bm25(store.bm25_path, {"sentinel": True})

    eng = RAGEngine(store=store, embedder=stub_embedder)
    eng._ensure_loaded()
    assert eng._chunk_ids == chunk_ids
    assert eng._embeddings.shape == (3, 4)
    assert eng._bm25 == {"sentinel": True}

    # second call is a no-op (loaded flag set)
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


def test_ensure_loaded_with_missing_artifacts(tmp_data_dir, stub_embedder):
    eng = RAGEngine(store=Store(), embedder=stub_embedder)
    eng._ensure_loaded()
    assert eng._bm25 is None
    assert eng._embeddings is None
    assert eng._chunk_ids == []
