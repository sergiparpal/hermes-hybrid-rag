from pathlib import Path

import numpy as np

from advanced_rag.indexing import index_path
from advanced_rag.storage import Store

FIXTURES = Path(__file__).parent / "fixtures" / "docs"


def _stage(tmp_path: Path) -> Path:
    """Copy the three fixture docs into a fresh dir so tests can mutate them."""
    out = tmp_path / "docs"
    out.mkdir()
    for name in ("alpha.md", "beta.md", "gamma.txt"):
        (out / name).write_text((FIXTURES / name).read_text())
    return out


def test_index_path_creates_artifacts(tmp_data_dir, tmp_path, stub_embedder):
    docs = _stage(tmp_path)
    store = Store()
    summary = index_path(docs, store=store, embedder=stub_embedder)
    assert summary["files_added_or_updated"] == 3
    assert summary["parents"] >= 3
    assert summary["chunks"] >= 3
    assert store.npz_path.exists()
    assert store.bm25_path.exists()


def test_index_skips_unchanged(tmp_data_dir, tmp_path, stub_embedder):
    docs = _stage(tmp_path)
    store = Store()
    index_path(docs, store=store, embedder=stub_embedder)
    again = index_path(docs, store=store, embedder=stub_embedder)
    assert again["files_added_or_updated"] == 0
    assert again["files_unchanged"] == 3


def test_index_picks_up_modified_file(tmp_data_dir, tmp_path, stub_embedder):
    docs = _stage(tmp_path)
    store = Store()
    index_path(docs, store=store, embedder=stub_embedder)
    target = docs / "alpha.md"
    # bump mtime + content to ensure the diff trips
    new_text = target.read_text() + "\n\n## New section\nFresh content here.\n"
    target.write_text(new_text)
    import os, time
    later = time.time() + 5
    os.utime(target, (later, later))

    again = index_path(docs, store=store, embedder=stub_embedder)
    assert again["files_added_or_updated"] == 1


def test_index_handles_deleted_file(tmp_data_dir, tmp_path, stub_embedder):
    docs = _stage(tmp_path)
    store = Store()
    index_path(docs, store=store, embedder=stub_embedder)
    (docs / "alpha.md").unlink()
    again = index_path(docs, store=store, embedder=stub_embedder)
    assert again["files_deleted"] == 1
    assert again["totals"]["files"] == 2


def test_index_force_reprocesses_everything(tmp_data_dir, tmp_path, stub_embedder):
    docs = _stage(tmp_path)
    store = Store()
    index_path(docs, store=store, embedder=stub_embedder)
    again = index_path(docs, store=store, embedder=stub_embedder, force=True)
    assert again["files_added_or_updated"] == 3


def test_embed_row_invariant(tmp_data_dir, tmp_path, stub_embedder):
    """Chunk row N in canonical SQLite order ↔ row N of embeddings.npz."""
    docs = _stage(tmp_path)
    store = Store()
    index_path(docs, store=store, embedder=stub_embedder)

    embeddings, chunk_ids_in_npz = store.load_embeddings(store.npz_path)
    canonical = [c.id for c in store.iter_chunks_ordered()]
    assert chunk_ids_in_npz == canonical
    assert embeddings.shape[0] == len(canonical)
    # embed_row column in SQLite must match the row index it occupies
    rows = list(store.iter_chunks_ordered())
    for row_idx, row in enumerate(rows):
        assert row.embed_row == row_idx
