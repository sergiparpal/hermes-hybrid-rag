"""Atomic ``.npz`` I/O for the embeddings array.

Splitting this from ``storage.py`` separates two reasons-to-change: SQL
schema evolution vs. numpy format / serialization choices. The atomic-write
pattern (``.tmp`` then ``os.replace``) lives here because nothing else in
the package writes binary artifacts.

Legacy ``bm25.pkl`` cleanup also lives here — see ``storage.py`` module
docstring for the security reasoning (pickle is gone, but a previously-
installed file must still be unlinked on the first rebuild).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .config import bm25_path, npz_path


class ArtifactStore:
    """Owns the embeddings ``.npz`` file. One instance per ``data_dir``."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    @property
    def npz_path(self) -> Path:
        return npz_path(self.data_dir)

    @property
    def legacy_bm25_path(self) -> Path:
        return bm25_path(self.data_dir)

    def exists(self) -> bool:
        return self.npz_path.exists()

    def save(self, embeddings: np.ndarray) -> None:
        """Atomically write the embeddings array. Writes to ``.tmp`` first,
        then ``os.replace`` — so a partial write never corrupts the live
        file."""
        target = self.npz_path
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        try:
            # Pass a file handle so numpy doesn't auto-append `.npz` and
            # break our atomic-rename scheme.
            with open(tmp, "wb") as fh:
                np.savez(fh, embeddings=embeddings)
            os.replace(tmp, target)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    def load(self) -> np.ndarray:
        """Return the embeddings array. Old ``.npz`` files that still carry
        a ``chunk_ids`` array load fine — we just ignore that key."""
        with np.load(self.npz_path) as data:
            return data["embeddings"]

    def delete(self) -> None:
        """Remove the ``.npz`` if present. Called when an empty rebuild
        produces no chunks at all."""
        if self.npz_path.exists():
            self.npz_path.unlink()

    def unlink_legacy_bm25(self) -> None:
        """Delete any stale ``bm25.pkl`` left behind by an old install.

        Leaving it on disk would be a stale, unreferenced pickle file — and
        a cohabiting attacker's planted file (the exact threat we removed
        pickle to address) would otherwise survive a re-index unnoticed.
        """
        legacy = self.legacy_bm25_path
        if legacy.exists():
            try:
                legacy.unlink()
            except OSError:
                pass
