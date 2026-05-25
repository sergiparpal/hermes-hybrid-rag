import numpy as np

from hybrid_rag.artifacts import ArtifactStore


def test_save_and_load_atomic(tmp_data_dir):
    arts = ArtifactStore(tmp_data_dir)
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    arts.save(arr)
    loaded = arts.load()
    assert np.array_equal(loaded, arr)
    # The .tmp must be cleaned up after a successful write.
    tmp = arts.npz_path.with_suffix(arts.npz_path.suffix + ".tmp")
    assert not tmp.exists()


def test_load_ignores_legacy_chunk_ids_array(tmp_data_dir):
    """`.npz` files written by older versions still carried a `chunk_ids`
    array. The new loader silently ignores that key and reads only
    ``embeddings``."""
    arts = ArtifactStore(tmp_data_dir)
    arr = np.arange(8, dtype=np.float32).reshape(2, 4)
    # Manually write an .npz in the old shape so we don't accidentally drift
    # the compatibility surface.
    np.savez(arts.npz_path, embeddings=arr,
             chunk_ids=np.asarray([99, 100], dtype=np.int64))
    loaded = arts.load()
    assert np.array_equal(loaded, arr)


def test_delete_removes_npz(tmp_data_dir):
    arts = ArtifactStore(tmp_data_dir)
    arts.save(np.zeros((1, 4), dtype=np.float32))
    assert arts.exists()
    arts.delete()
    assert not arts.exists()


def test_delete_is_noop_when_missing(tmp_data_dir):
    arts = ArtifactStore(tmp_data_dir)
    assert not arts.exists()
    arts.delete()  # must not raise


def test_unlink_legacy_bm25_removes_planted_file(tmp_data_dir):
    """The old pickle path is gone; if a leftover bm25.pkl exists from a
    previous install (or was planted by a hostile cohabiter), the rebuild
    path must unlink it."""
    arts = ArtifactStore(tmp_data_dir)
    arts.legacy_bm25_path.write_bytes(b"junk")
    assert arts.legacy_bm25_path.exists()
    arts.unlink_legacy_bm25()
    assert not arts.legacy_bm25_path.exists()


def test_unlink_legacy_bm25_is_noop_when_missing(tmp_data_dir):
    arts = ArtifactStore(tmp_data_dir)
    assert not arts.legacy_bm25_path.exists()
    arts.unlink_legacy_bm25()  # must not raise
