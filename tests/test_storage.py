import os
from pathlib import Path

import numpy as np
import pytest

from hierarchical_rag.storage import Store


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
    loaded, ids = store.load_embeddings(store.npz_path)
    assert np.array_equal(loaded, arr)
    assert ids == [10, 20, 30]
    # the .tmp must be cleaned up after a successful write
    assert not store.npz_path.with_suffix(store.npz_path.suffix + ".tmp").exists()


def test_save_bm25_atomic(tmp_data_dir):
    store = Store()
    obj = {"k": [1, 2, 3]}
    store.save_bm25(store.bm25_path, obj)
    loaded = store.load_bm25(store.bm25_path)
    assert loaded == obj


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
