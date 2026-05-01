import argparse
import json
from pathlib import Path

import pytest

from advanced_rag import cli, indexing
from advanced_rag.storage import Store

FIXTURES = Path(__file__).parent / "fixtures" / "docs"


def _parse(argv):
    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="cmd")
    rag_p = sub.add_parser("rag")
    cli.setup_rag_parser(rag_p)
    return parser.parse_args(argv)


def test_setup_parser_index_subcommand():
    args = _parse(["rag", "index", "/tmp/x"])
    assert args.rag_cmd == "index"
    assert args.path == "/tmp/x"
    assert args.force is False


def test_setup_parser_force_flag():
    args = _parse(["rag", "index", "/tmp/x", "--force"])
    assert args.force is True


def test_setup_parser_stats():
    args = _parse(["rag", "stats"])
    assert args.rag_cmd == "stats"


def test_setup_parser_clear_yes():
    args = _parse(["rag", "clear", "--yes"])
    assert args.rag_cmd == "clear"
    assert args.yes is True


def test_handle_index(tmp_data_dir, tmp_path, stub_embedder, monkeypatch, capsys):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "x.md").write_text("# Hi\n\n## A\nbody")

    # patch indexing.index_path to use the stub embedder so this test stays
    # offline; the real signature is exercised by test_indexing.
    real_index = indexing.index_path

    def _idx(path, force=False):
        return real_index(path, force=force, store=Store(), embedder=stub_embedder)

    monkeypatch.setattr(indexing, "index_path", _idx)

    args = argparse.Namespace(rag_cmd="index", path=str(docs), force=False)
    code = cli.handle_rag(args)
    assert code == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["files_added_or_updated"] == 1


def test_handle_stats(tmp_data_dir, capsys):
    args = argparse.Namespace(rag_cmd="stats")
    code = cli.handle_rag(args)
    assert code == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["files"] == 0


def test_handle_clear_declined(tmp_data_dir, capsys):
    args = argparse.Namespace(rag_cmd="clear", yes=False)
    code = cli.handle_rag(args, _input=lambda _: "n")
    assert code == 1
    assert "aborted" in capsys.readouterr().out


def test_handle_clear_confirmed(tmp_data_dir, capsys):
    # write a sentinel into the data dir so we can prove rmtree happened
    (tmp_data_dir / "sentinel.txt").write_text("x")
    args = argparse.Namespace(rag_cmd="clear", yes=True)
    code = cli.handle_rag(args)
    assert code == 0
    assert not (tmp_data_dir / "sentinel.txt").exists()


def test_handle_unknown_subcommand_returns_two(capsys):
    args = argparse.Namespace(rag_cmd="bogus")
    code = cli.handle_rag(args)
    assert code == 2


def test_handle_index_returns_two_on_failure(tmp_data_dir, monkeypatch, capsys):
    def boom(path, force=False):
        raise RuntimeError("simulated failure")
    monkeypatch.setattr(indexing, "index_path", boom)
    args = argparse.Namespace(rag_cmd="index", path="/tmp/whatever", force=False)
    code = cli.handle_rag(args)
    assert code == 2
