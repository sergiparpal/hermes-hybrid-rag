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
    # BM25 is no longer persisted — it's rebuilt from SQLite on engine load.
    # Any bm25.pkl on disk would be either a stale legacy file or a planted
    # one; either way, indexing must leave none behind.
    assert not store.bm25_path.exists()


def test_index_unlinks_legacy_bm25_pickle(tmp_data_dir, tmp_path, stub_embedder):
    """A bm25.pkl left behind by a pre-fix install — or planted by an
    attacker into HERMES_RAG_DATA_DIR — must be removed during the next
    index so it can never be loaded by a future code path."""
    docs = _stage(tmp_path)
    store = Store()
    # Plant a pickle file before indexing.
    store.bm25_path.parent.mkdir(parents=True, exist_ok=True)
    store.bm25_path.write_bytes(b"\x80\x04this-should-be-removed")
    assert store.bm25_path.exists()

    index_path(docs, store=store, embedder=stub_embedder)
    assert not store.bm25_path.exists()


def test_walk_skips_symlinks(tmp_data_dir, tmp_path, stub_embedder):
    """Symlinks discovered during a recursive walk must not be followed —
    indexing them would land the symlink target's content (e.g. ~/.ssh/...)
    in the catalog where every later query can surface it."""
    from advanced_rag.indexing import _walk

    docs = tmp_path / "docs"
    docs.mkdir()
    real = tmp_path / "outside_secret.md"
    real.write_text("# Secret\n\n## Section\nsensitive content\n")

    real_doc = docs / "ok.md"
    real_doc.write_text("# Hi\n\n## Topic\nbody")

    # Plant a symlink inside the indexed dir that points outside.
    link = docs / "innocent_looking.md"
    link.symlink_to(real)

    out = _walk(docs)
    assert real_doc in out
    assert link not in out
    assert real not in out


def test_walk_skips_symlinked_directories(tmp_data_dir, tmp_path, stub_embedder):
    """os.walk(followlinks=False) prevents descent into symlinked dirs —
    closes the variant where the attacker symlinks a whole directory."""
    from advanced_rag.indexing import _walk

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "ok.md").write_text("# Hi\n\n## Topic\nbody")

    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "secret.md").write_text("# Secret\n\n## Bad\nsensitive")

    (docs / "shortcut").symlink_to(outside)

    out = _walk(docs)
    paths = {str(p) for p in out}
    assert str(docs / "ok.md") in paths
    assert not any("secret.md" in p for p in paths)


def test_walk_skips_oversized_file(tmp_data_dir, tmp_path, monkeypatch, capsys):
    """Files above MAX_INDEX_FILE_BYTES are skipped with a stderr warning —
    OOM protection: extractors load the whole file into memory."""
    from advanced_rag import indexing as indexing_mod

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "small.md").write_text("# T\n\n## S\nbody")
    (docs / "big.md").write_text("x" * 1024)  # 1 KiB

    monkeypatch.setattr(indexing_mod, "MAX_INDEX_FILE_BYTES", 512)
    out = indexing_mod._walk(docs)
    names = {p.name for p in out}
    assert "small.md" in names
    assert "big.md" not in names
    assert "exceeds MAX_INDEX_FILE_BYTES" in capsys.readouterr().err


def test_walk_honors_explicit_symlink_root(tmp_data_dir, tmp_path):
    """A symlink chosen explicitly by the user (single-file index target) is
    honored — only recursive walks reject symlinks."""
    from advanced_rag.indexing import _walk

    real = tmp_path / "real.md"
    real.write_text("# T\n\n## S\nbody")
    link = tmp_path / "link.md"
    link.symlink_to(real)

    out = _walk(link)
    assert out == [link]


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


def test_index_picks_up_in_place_edit_with_preserved_mtime_size(
    tmp_data_dir, tmp_path, stub_embedder
):
    """An in-place edit that preserves both mtime and size must still trigger
    a reindex — the SHA-256 tiebreaker is what catches this."""
    docs = _stage(tmp_path)
    store = Store()
    index_path(docs, store=store, embedder=stub_embedder)

    target = docs / "alpha.md"
    original = target.read_text()
    # rewrite with same byte length and force the original mtime back on
    replacement = "Z" * len(original)
    assert len(replacement) == len(original)
    pre_stat = target.stat()
    target.write_text(replacement)
    import os as _os
    _os.utime(target, (pre_stat.st_mtime, pre_stat.st_mtime))

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
    """Chunk row N in canonical SQLite order ↔ row N of embeddings.npz, and
    `chunks.embed_row` is the on-disk source of truth for that mapping.
    `.npz` no longer carries `chunk_ids`; the engine derives the list from
    SQLite via `iter_chunks_ordered()`."""
    docs = _stage(tmp_path)
    store = Store()
    index_path(docs, store=store, embedder=stub_embedder)

    embeddings = store.load_embeddings(store.npz_path)
    rows = list(store.iter_chunks_ordered())
    assert embeddings.shape[0] == len(rows)
    # embed_row column in SQLite must match the row index it occupies — this
    # is now what the query path uses to map dense-search hits back to chunks.
    for row_idx, row in enumerate(rows):
        assert row.embed_row == row_idx


def test_engine_chunk_ids_match_canonical_order(tmp_data_dir, tmp_path, stub_embedder):
    """After indexing, the engine's `_chunk_ids` (loaded from SQLite) must
    align with the .npz row index order."""
    from advanced_rag.engine import RAGEngine
    docs = _stage(tmp_path)
    store = Store()
    index_path(docs, store=store, embedder=stub_embedder)

    eng = RAGEngine(store=store, embedder=stub_embedder)
    eng._ensure_loaded()
    canonical = [c.id for c in store.iter_chunks_ordered()]
    assert eng._chunk_ids == canonical
    assert eng._embeddings.shape[0] == len(canonical)


def test_index_with_explicit_store_does_not_reset_singleton(
    tmp_data_dir, tmp_path, stub_embedder, monkeypatch,
):
    """When the caller supplies an explicit store=, the process-wide engine
    singleton is bound to a different data_dir; resetting it would force
    an unrelated cold reload on the next ambient call."""
    from advanced_rag import engine as engine_mod

    reset_called = {"n": 0}

    class _Spy:
        def reset(self):
            reset_called["n"] += 1

    monkeypatch.setattr(engine_mod, "get_engine", lambda: _Spy())

    docs = _stage(tmp_path)
    store = Store()
    index_path(docs, store=store, embedder=stub_embedder)
    assert reset_called["n"] == 0


def test_index_without_explicit_store_resets_singleton(
    tmp_data_dir, tmp_path, stub_embedder, monkeypatch,
):
    """When the caller omits store=, index_path owns the Store and is
    expected to flush the singleton's cached artifacts."""
    from advanced_rag import engine as engine_mod
    # Make sure index_path won't actually try to load a real Embedder.
    import advanced_rag.indexing as indexing_mod
    monkeypatch.setattr(indexing_mod, "rebuild_artifacts", lambda *a, **kw: None)

    reset_called = {"n": 0}

    class _Spy:
        def reset(self):
            reset_called["n"] += 1

    monkeypatch.setattr(engine_mod, "get_engine", lambda: _Spy())

    docs = _stage(tmp_path)
    # Bypass Embedder construction by passing stub explicitly.
    index_path(docs, embedder=stub_embedder)
    assert reset_called["n"] == 1


def test_indexing_failure_warning_goes_to_stderr(
    tmp_data_dir, tmp_path, stub_embedder, monkeypatch, capsys
):
    """Per-file failures must not pollute stdout — the CLI emits JSON there."""
    docs = _stage(tmp_path)
    # Force every file to raise during extraction so the failure path runs.
    import advanced_rag.indexing as indexing_mod

    def boom(_path):
        raise RuntimeError("synthetic extraction error")

    monkeypatch.setattr(indexing_mod, "_extract_parents", boom)
    store = Store()
    index_path(docs, store=store, embedder=stub_embedder)
    captured = capsys.readouterr()
    assert "[advanced-rag] failed to index" in captured.err
    assert "[advanced-rag] failed to index" not in captured.out
