from pathlib import Path

from hybrid_rag import manifest
from hybrid_rag.storage import Store


class _Stat:
    """Tiny stand-in for os.stat_result."""

    def __init__(self, mtime: float, size: int):
        self.st_mtime = mtime
        self.st_size = size


def _seed_file(store, path, mtime=1000.0, size=42, h="hash", filetype="md"):
    return store.bulk_insert_files([(str(path), mtime, size, h, filetype, 0.0)])


def test_diff_buckets(tmp_data_dir, tmp_path):
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
    diff = manifest.diff(store, disk)
    assert a in diff["unchanged"]
    assert any(p == b for p, _ in diff["changed"])
    assert c in diff["new"]
    assert len(diff["deleted"]) == 1


def test_diff_hash_tiebreaker_detects_in_place_edit(tmp_data_dir, tmp_path):
    """When (mtime, size) match, the hash callback decides changed vs
    unchanged."""
    store = Store()
    a = tmp_path / "a.md"
    a.write_text("x")
    _seed_file(store, a, mtime=1.0, size=1, h="hash-original")

    disk = {a: _Stat(mtime=1.0, size=1)}

    # Same hash → unchanged.
    diff = manifest.diff(store, disk, hash_fn=lambda _p: "hash-original")
    assert a in diff["unchanged"]
    assert diff["changed"] == []

    # Different hash → changed (the in-place-edit case).
    diff = manifest.diff(store, disk, hash_fn=lambda _p: "hash-different")
    assert diff["unchanged"] == []
    assert any(p == a for p, _ in diff["changed"])


def test_diff_hash_fn_skipped_when_stat_already_differs(tmp_data_dir, tmp_path):
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
    manifest.diff(store, disk, hash_fn=hash_fn)
    assert calls == []
