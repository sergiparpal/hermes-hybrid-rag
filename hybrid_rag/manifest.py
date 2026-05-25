"""Filesystem-vs-catalog reconciliation. Pure function — takes the catalog
plus a dict of paths-on-disk and returns the four-bucket changeset.

Lives outside ``storage.py`` because it's reconciliation logic, not SQL: the
SQL is exactly one SELECT, and everything else is set arithmetic over the
result. Splitting it out lets the storage layer focus on the row-level
operations and makes the diff testable in isolation.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable


def diff(
    catalog,
    disk_files: dict[Path, os.stat_result],
    hash_fn: Callable[[Path], str] | None = None,
) -> dict:
    """Returns ``{unchanged, changed, new, deleted}``, each a list keyed by path.

    - ``unchanged``: same ``mtime`` AND ``size`` (and same ``content_hash``,
      when ``hash_fn`` is supplied) as the row in ``files``.
    - ``changed``: row exists but ``mtime``, ``size``, or (when checked)
      ``content_hash`` differ. List of ``(path, file_id)`` tuples.
    - ``new``: path not in the ``files`` table.
    - ``deleted``: row exists but path not in ``disk_files``. List of
      ``file_id`` ints.

    ``hash_fn`` is only invoked on the ``(mtime, size)``-match branch, so
    unchanged files dominate the cost: each pays exactly one hash. Files
    with stale ``(mtime, size)`` shortcut to "changed" without re-hashing —
    the hash will be recomputed when the file is reindexed anyway.
    """
    conn = catalog.connect()
    rows = {
        Path(r["path"]): {
            "id": r["id"], "mtime": r["mtime"],
            "size": r["size"], "content_hash": r["content_hash"],
        }
        for r in conn.execute(
            "SELECT id, path, mtime, size, content_hash FROM files"
        )
    }

    unchanged: list[Path] = []
    changed: list[tuple[Path, int]] = []
    new: list[Path] = []
    deleted: list[int] = []

    for path, st in disk_files.items():
        row = rows.get(path)
        if row is None:
            new.append(path)
        elif row["mtime"] == st.st_mtime and row["size"] == st.st_size:
            if hash_fn is None:
                unchanged.append(path)
            else:
                disk_hash = hash_fn(path)
                if disk_hash == row["content_hash"]:
                    unchanged.append(path)
                else:
                    # In-place edit that preserved (mtime, size) — rare but
                    # real (e.g. `os.utime` after a same-size rewrite).
                    changed.append((path, row["id"]))
        else:
            changed.append((path, row["id"]))

    for path, row in rows.items():
        if path not in disk_files:
            deleted.append(row["id"])

    return {"unchanged": unchanged, "changed": changed,
            "new": new, "deleted": deleted}
