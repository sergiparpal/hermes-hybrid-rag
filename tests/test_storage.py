import os
from pathlib import Path

import numpy as np

from advanced_rag.storage import Store


def _stat(path: Path) -> os.stat_result:
    return path.stat()


def _seed_file(store, path, mtime=1000.0, size=42, h="hash", filetype="md"):
    return store.bulk_insert_files([(str(path), mtime, size, h, filetype, 0.0)])


def test_init_creates_data_dir_and_schema(tmp_data_dir):
    store = Store()
    assert store.data_dir == tmp_data_dir
    conn = store.connect()
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"files", "parents", "chunks", "meta"}.issubset(tables)


def test_explicit_data_dir_overrides_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_RAG_DATA_DIR", str(tmp_path / "env"))
    other = tmp_path / "explicit"
    store = Store(data_dir=other)
    assert store.data_dir == other
    assert other.exists()


def test_manifest_diff_buckets(tmp_data_dir, tmp_path):
    store = Store()
    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    c = tmp_path / "c.md"
    for f in (a, b, c):
        f.write_text("x")
    _seed_file(store, a, mtime=1.0, size=1)  # unchanged
    _seed_file(store, b, mtime=2.0, size=2)  # changed
    _seed_file(store, Path("/gone.md"), mtime=3.0, size=3)  # deleted

    disk = {
        a: _Stat(mtime=1.0, size=1),
        b: _Stat(mtime=99.0, size=99),  # mtime changed
        c: _Stat(mtime=4.0, size=4),    # new
    }
    diff = store.manifest_diff(disk)
    assert a in diff["unchanged"]
    assert any(p == b for p, _ in diff["changed"])
    assert c in diff["new"]
    assert len(diff["deleted"]) == 1


def test_manifest_diff_hash_tiebreaker_detects_in_place_edit(tmp_data_dir, tmp_path):
    """When (mtime, size) match, the hash callback decides changed vs unchanged."""
    store = Store()
    a = tmp_path / "a.md"
    a.write_text("x")
    _seed_file(store, a, mtime=1.0, size=1, h="hash-original")

    disk = {a: _Stat(mtime=1.0, size=1)}

    # Same hash → unchanged.
    diff = store.manifest_diff(disk, hash_fn=lambda _p: "hash-original")
    assert a in diff["unchanged"]
    assert diff["changed"] == []

    # Different hash → changed (the in-place-edit case).
    diff = store.manifest_diff(disk, hash_fn=lambda _p: "hash-different")
    assert diff["unchanged"] == []
    assert any(p == a for p, _ in diff["changed"])


def test_manifest_diff_hash_fn_skipped_when_stat_already_differs(tmp_data_dir, tmp_path):
    """The hash callback must not run for files whose (mtime, size) already
    differ — the diff has enough signal to mark them changed without I/O."""
    store = Store()
    a = tmp_path / "a.md"
    a.write_text("x")
    _seed_file(store, a, mtime=1.0, size=1, h="hash-original")

    calls: list[Path] = []

    def hash_fn(p: Path) -> str:
        calls.append(p)
        return "anything"

    disk = {a: _Stat(mtime=99.0, size=99)}
    store.manifest_diff(disk, hash_fn=hash_fn)
    assert calls == []


def test_cascade_delete_kills_parents_and_chunks(tmp_data_dir):
    store = Store()
    file_ids = store.bulk_insert_files([("/x.md", 0.0, 0, "h", "md", 0.0)])
    fid = file_ids["/x.md"]
    pids = store.bulk_insert_parents(
        [(fid, 0, "section", "T1", None, "body1", 5),
         (fid, 1, "section", "T2", None, "body2", 5)]
    )
    store.bulk_insert_chunks([(pids[0], 0, "c1", 0), (pids[0], 1, "c2", 1),
                              (pids[1], 0, "c3", 2)])
    assert store.stats()["chunks"] == 3
    store.delete_files([fid])
    assert store.stats() == {**store.stats(), "files": 0, "parents": 0, "chunks": 0}


def test_save_and_load_embeddings_atomic(tmp_data_dir):
    store = Store()
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    store.save_embeddings(store.npz_path, arr, [10, 20, 30])
    loaded = store.load_embeddings(store.npz_path)
    assert np.array_equal(loaded, arr)
    # the .tmp must be cleaned up after a successful write
    assert not store.npz_path.with_suffix(store.npz_path.suffix + ".tmp").exists()


def test_load_embeddings_ignores_legacy_chunk_ids_array(tmp_data_dir):
    """`.npz` files written by older versions still carried `chunk_ids`.
    The new loader silently ignores that key and reads only `embeddings`."""
    store = Store()
    arr = np.arange(8, dtype=np.float32).reshape(2, 4)
    # Manually write an .npz in the old shape so we don't accidentally drift
    # the old compatibility surface.
    np.savez(store.npz_path, embeddings=arr,
             chunk_ids=np.asarray([99, 100], dtype=np.int64))
    loaded = store.load_embeddings(store.npz_path)
    assert np.array_equal(loaded, arr)


def test_save_embeddings_accepts_chunk_ids_for_compat(tmp_data_dir):
    """Old callers passing `chunk_ids=` should still work; the array is
    intentionally not persisted to the .npz."""
    store = Store()
    arr = np.arange(8, dtype=np.float32).reshape(2, 4)
    store.save_embeddings(store.npz_path, arr, chunk_ids=[1, 2])
    with np.load(store.npz_path) as data:
        assert "embeddings" in data.files
        assert "chunk_ids" not in data.files


def test_iter_bm25_texts_prefers_contextual(tmp_data_dir):
    """Phase 2: when text_for_bm25 is present it shadows the raw chunk text;
    otherwise the raw text is yielded. The engine builds BM25 from this
    stream — pickle is gone, the only persistent BM25 source is SQLite."""
    store = Store()
    fid = store.bulk_insert_files([("/x.md", 0.0, 0, "h", "md", 0.0)])["/x.md"]
    pid = store.bulk_insert_parents([(fid, 0, "section", "T", None, "x", 1)])[0]
    store.bulk_insert_chunks([
        (pid, 0, "raw-zero", 0),  # no text_for_bm25 → raw text
        (pid, 1, "raw-one", 0, "prefix", "raw-one-embed", "composed-one"),
    ])
    out = list(store.iter_bm25_texts_ordered())
    assert out == ["raw-zero", "composed-one"]


def test_parent_ids_for_chunks_batched(tmp_data_dir):
    store = Store()
    fid = store.bulk_insert_files([("/x.md", 0.0, 0, "h", "md", 0.0)])["/x.md"]
    p1, p2 = store.bulk_insert_parents([
        (fid, 0, "section", "T1", None, "x", 1),
        (fid, 1, "section", "T2", None, "y", 1),
    ])
    cid_a, cid_b, cid_c = store.bulk_insert_chunks([
        (p1, 0, "a", 0), (p1, 1, "b", 0), (p2, 0, "c", 0),
    ])
    out = store.parent_ids_for_chunks([cid_a, cid_b, cid_c, 999_999])
    assert out == {cid_a: p1, cid_b: p1, cid_c: p2}
    # missing id is silently dropped, not raised
    assert 999_999 not in out
    # empty input returns empty dict, no SQL roundtrip
    assert store.parent_ids_for_chunks([]) == {}


def test_get_parents_batched(tmp_data_dir):
    store = Store()
    fid = store.bulk_insert_files([("/x.md", 0.0, 0, "h", "md", 0.0)])["/x.md"]
    p1, p2 = store.bulk_insert_parents([
        (fid, 0, "section", "T1", None, "x-body", 6),
        (fid, 1, "section", "T2", None, "y-body", 6),
    ])
    rows = store.get_parents([p1, p2, 999_999])
    assert set(rows.keys()) == {p1, p2}
    assert rows[p1]["title"] == "T1"
    assert rows[p2]["title"] == "T2"
    # source_path / filetype are joined in
    assert rows[p1]["source_path"] == "/x.md"
    assert rows[p1]["filetype"] == "md"


def test_iter_chunks_ordered_canonical_order(tmp_data_dir):
    store = Store()
    fid = store.bulk_insert_files([("/x.md", 0.0, 0, "h", "md", 0.0)])["/x.md"]
    p1, p2 = store.bulk_insert_parents([
        (fid, 0, "section", "T1", None, "x", 1),
        (fid, 1, "section", "T2", None, "y", 1),
    ])
    # insert chunks out of order to confirm iter_chunks_ordered re-sorts
    store.bulk_insert_chunks([
        (p2, 0, "from p2 ord 0", 0),
        (p1, 1, "from p1 ord 1", 0),
        (p1, 0, "from p1 ord 0", 0),
        (p2, 1, "from p2 ord 1", 0),
    ])
    rows = list(store.iter_chunks_ordered())
    assert [(r.parent_id, r.ord) for r in rows] == [
        (p1, 0), (p1, 1), (p2, 0), (p2, 1)
    ]


class _Stat:
    """Tiny stand-in for os.stat_result."""

    def __init__(self, mtime: float, size: int):
        self.st_mtime = mtime
        self.st_size = size
