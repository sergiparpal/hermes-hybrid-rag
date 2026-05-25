"""`hermes rag {index,stats,clear}` — pure dispatcher returning an exit code.

Adapter wraps these for whatever shape Hermes wants in `ctx.register_cli_command`.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import indexing
from .artifacts import bm25_state_path
from .config import bm25_path, db_path, get_data_dir, npz_path, toggles_path
from .storage import Store

log = logging.getLogger(__name__)


def _owned_artifacts(data_dir: Path) -> list[Path]:
    """The files this plugin actually owns inside the data dir.

    `clear` only unlinks these; anything else the user (or another tool)
    has placed in HERMES_RAG_DATA_DIR is left alone. This replaces the
    previous denylist-guarded `rmtree`, which would have happily wiped any
    sibling content under a deeply-nested data dir.
    """
    return [
        db_path(data_dir),
        npz_path(data_dir),
        bm25_state_path(data_dir),
        bm25_path(data_dir),  # legacy file from the old pickle path
        toggles_path(data_dir),
    ]


def setup_rag_parser(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="rag_cmd", required=True)
    p_idx = sub.add_parser("index", help="Walk a directory and (re)index supported documents.")
    p_idx.add_argument("path", help="Directory or file to index (md/txt/pdf).")
    p_idx.add_argument("--force", action="store_true",
                       help="Reindex every matched file even if unchanged.")
    sub.add_parser("stats", help="Show counts of indexed files / parents / chunks.")
    p_clear = sub.add_parser("clear", help="Delete the entire data directory.")
    p_clear.add_argument("--yes", action="store_true",
                         help="Skip the interactive confirmation prompt.")


def handle_rag(args: argparse.Namespace, *, _indexer=indexing,
               _store_factory=Store, _input=input) -> int:
    """Returns exit code (0 success, 1 declined, 2 error)."""
    try:
        cmd = getattr(args, "rag_cmd", None)
        if cmd == "index":
            summary = _indexer.index_path(Path(args.path), force=bool(getattr(args, "force", False)))
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        if cmd == "stats":
            store = _store_factory()
            print(json.dumps(store.stats(), indent=2, sort_keys=True))
            return 0
        if cmd == "clear":
            data_dir = get_data_dir()
            artifacts = _owned_artifacts(data_dir)
            present = [p for p in artifacts if p.exists()]
            if not present:
                print(f"nothing to remove at {data_dir}")
                return 0
            if not bool(getattr(args, "yes", False)):
                names = ", ".join(p.name for p in present)
                resp = _input(
                    f"Delete RAG artifacts ({names}) in {data_dir}? [y/N] "
                ).strip().lower()
                if resp not in ("y", "yes"):
                    print("aborted")
                    return 1
            for p in present:
                try:
                    p.unlink()
                except OSError as e:
                    log.warning("failed to unlink %s: %s", p, e)
            print(f"removed {len(present)} file(s) from {data_dir}")
            # Note for ops: a running Hermes process keeps the SQLite file
            # open. POSIX preserves the inode for the open fd, so that
            # process keeps reading the *deleted* database (and never
            # notices the clear) until it restarts. Restart Hermes after
            # `hermes rag clear` if you want the change visible immediately.
            return 0
        print(f"unknown rag subcommand: {cmd!r}", file=sys.stderr)
        return 2
    except Exception as e:
        log.exception("rag command failed")
        print(f"error: {e}", file=sys.stderr)
        return 2
