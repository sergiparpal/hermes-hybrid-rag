"""`hermes rag {index,stats,clear}` — pure dispatcher returning an exit code.

Adapter wraps these for whatever shape Hermes wants in `ctx.register_cli_command`.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

from . import indexing
from .config import get_data_dir
from .storage import Store

log = logging.getLogger(__name__)


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
            if not bool(getattr(args, "yes", False)):
                resp = _input(f"Delete RAG data at {data_dir}? [y/N] ").strip().lower()
                if resp not in ("y", "yes"):
                    print("aborted")
                    return 1
            if data_dir.exists():
                shutil.rmtree(data_dir)
                print(f"removed {data_dir}")
            else:
                print(f"nothing to remove at {data_dir}")
            return 0
        print(f"unknown rag subcommand: {cmd!r}", file=sys.stderr)
        return 2
    except Exception as e:
        log.exception("rag command failed")
        print(f"error: {e}", file=sys.stderr)
        return 2
