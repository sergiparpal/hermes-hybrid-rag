"""Atomic on-disk artifacts: the embeddings ``.npz`` and the BM25 state
sidecar (``bm25_state.json``).

Splitting this from ``storage.py`` separates two reasons-to-change: SQL
schema evolution vs. numpy format / serialization choices. The atomic-write
pattern (``.tmp`` then ``os.replace``) lives here because nothing else in
the package writes binary or sidecar artifacts.

The BM25 sidecar is JSON, not pickle. Persisting BM25 at all is what
turns engine load from "tokenize-the-whole-corpus" (1-3 s on 100K chunks)
into "decode a JSON blob" (~100 ms) — see ``RAGEngine._ensure_loaded`` for
the consumer. Pickle would deliver an even faster load, but pickle is RCE
(CWE-502) if the data dir is ever writable by an attacker; the JSON path
is safe to deserialize unconditionally.

Legacy ``bm25.pkl`` cleanup also lives here — see ``storage.py`` module
docstring for the security reasoning (pickle is gone, but a previously-
installed file must still be unlinked on the first rebuild).
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from .config import bm25_path, get_data_dir, npz_path


def bm25_state_path(data_dir: Path | None = None) -> Path:
    return (data_dir or get_data_dir()) / "bm25_state.json"


def _atomic_write_bytes(target: Path, payload_writer) -> None:
    """Common atomic-replace wrapper. ``payload_writer(file_handle)`` does
    the actual write into the temp file; we handle the rename/cleanup.

    ``tempfile.mkstemp`` gives each writer a unique path (``<name>.<rand>``)
    on the same filesystem as ``target`` — so two writers to the same
    target can't clobber each other's tempfiles before the rename. Today
    the only producer of these artifacts is the indexing CLI (single
    process), but we don't want the safety of the atomic rename to
    silently depend on that.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=str(target.parent),
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            payload_writer(fh)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


class ArtifactStore:
    """Owns the embeddings ``.npz`` file and the BM25 state sidecar.
    One instance per ``data_dir``."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    @property
    def npz_path(self) -> Path:
        return npz_path(self.data_dir)

    @property
    def bm25_state_path(self) -> Path:
        return bm25_state_path(self.data_dir)

    @property
    def legacy_bm25_path(self) -> Path:
        return bm25_path(self.data_dir)

    def exists(self) -> bool:
        return self.npz_path.exists()

    def save(self, embeddings: np.ndarray) -> None:
        """Atomically write the embeddings array. Writes to ``.tmp`` first,
        then ``os.replace`` — so a partial write never corrupts the live
        file."""
        # Pass a file handle so numpy doesn't auto-append `.npz` and
        # break our atomic-rename scheme.
        _atomic_write_bytes(
            self.npz_path,
            lambda fh: np.savez(fh, embeddings=embeddings),
        )

    def load(self) -> np.ndarray:
        """Return the embeddings array. Old ``.npz`` files that still carry
        a ``chunk_ids`` array load fine — we just ignore that key."""
        with np.load(self.npz_path) as data:
            return data["embeddings"]

    def delete(self) -> None:
        """Remove the ``.npz`` and BM25 sidecar if present. Called when an
        empty rebuild produces no chunks at all."""
        if self.npz_path.exists():
            self.npz_path.unlink()
        self.delete_bm25_state()

    # --- BM25 sidecar ---

    def bm25_state_exists(self) -> bool:
        return self.bm25_state_path.exists()

    def save_bm25_state(self, state: dict) -> None:
        """Atomically write the BM25 state dict as JSON. See module
        docstring for why JSON instead of pickle."""
        _atomic_write_bytes(
            self.bm25_state_path,
            lambda fh: fh.write(json.dumps(state).encode("utf-8")),
        )

    def load_bm25_state(self) -> dict:
        """Decode the BM25 sidecar. Returns the raw dict — reconstruction
        into a `BM25Okapi` instance lives in `engine.py` so this module
        doesn't import rank_bm25."""
        with open(self.bm25_state_path, "rb") as fh:
            return json.loads(fh.read())

    def delete_bm25_state(self) -> None:
        if self.bm25_state_path.exists():
            try:
                self.bm25_state_path.unlink()
            except OSError:
                pass

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
