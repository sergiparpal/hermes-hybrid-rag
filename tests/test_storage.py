from hybrid_rag.storage import Store


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


def test_cascade_delete_kills_parents_and_chunks(tmp_data_dir):
    store = Store()
    file_ids = store.bulk_insert_files([("/x.md", 0.0, 0, "h", "md", 0.0)])
    fid = file_ids["/x.md"]
    pids = store.bulk_insert_parents(
        [(fid, 0, "section", "T1", None, "body1", 5),
         (fid, 1, "section", "T2", None, "body2", 5)]
    )
    store.bulk_insert_chunks([
        (pids[0], 0, "c1", 0, None, None, None),
        (pids[0], 1, "c2", 1, None, None, None),
        (pids[1], 0, "c3", 2, None, None, None),
    ])
    assert store.stats()["chunks"] == 3
    store.delete_files([fid])
    assert store.stats() == {**store.stats(), "files": 0, "parents": 0, "chunks": 0}


def test_iter_bm25_texts_prefers_contextual(tmp_data_dir):
    """When text_for_bm25 is present it shadows the raw chunk text;
    otherwise the raw text is yielded. The engine builds BM25 from this
    stream — pickle is gone, the only persistent BM25 source is SQLite."""
    store = Store()
    fid = store.bulk_insert_files([("/x.md", 0.0, 0, "h", "md", 0.0)])["/x.md"]
    pid = store.bulk_insert_parents([(fid, 0, "section", "T", None, "x", 1)])[0]
    store.bulk_insert_chunks([
        (pid, 0, "raw-zero", 0, None, None, None),  # no text_for_bm25 → raw text
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
        (p1, 0, "a", 0, None, None, None),
        (p1, 1, "b", 0, None, None, None),
        (p2, 0, "c", 0, None, None, None),
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
        (p2, 0, "from p2 ord 0", 0, None, None, None),
        (p1, 1, "from p1 ord 1", 0, None, None, None),
        (p1, 0, "from p1 ord 0", 0, None, None, None),
        (p2, 1, "from p2 ord 1", 0, None, None, None),
    ])
    rows = list(store.iter_chunks_ordered())
    assert [(r.parent_id, r.ord) for r in rows] == [
        (p1, 0), (p1, 1), (p2, 0), (p2, 1)
    ]
