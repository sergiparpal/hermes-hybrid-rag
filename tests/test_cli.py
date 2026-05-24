import argparse
import json
from pathlib import Path

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


def _seed_artifacts(data_dir: Path):
    """Drop the four files clear is supposed to own into data_dir."""
    from advanced_rag.config import bm25_path, db_path, npz_path, toggles_path

    for p in (db_path(data_dir), npz_path(data_dir),
              bm25_path(data_dir), toggles_path(data_dir)):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")


def test_handle_clear_declined(tmp_data_dir, capsys):
    _seed_artifacts(tmp_data_dir)
    args = argparse.Namespace(rag_cmd="clear", yes=False)
    code = cli.handle_rag(args, _input=lambda _: "n")
    assert code == 1
    assert "aborted" in capsys.readouterr().out


def test_handle_clear_confirmed_unlinks_owned_files_only(tmp_data_dir, capsys):
    """clear unlinks only the artifacts the plugin owns. Anything else the
    user (or another tool) parked under HERMES_RAG_DATA_DIR is left alone —
    the previous rmtree behavior was a foot-gun anywhere the data dir got
    misconfigured."""
    _seed_artifacts(tmp_data_dir)
    foreign = tmp_data_dir / "user_notes.md"
    foreign.write_text("important user data")

    args = argparse.Namespace(rag_cmd="clear", yes=True)
    code = cli.handle_rag(args)
    assert code == 0

    from advanced_rag.config import bm25_path, db_path, npz_path, toggles_path
    for p in (db_path(tmp_data_dir), npz_path(tmp_data_dir),
              bm25_path(tmp_data_dir), toggles_path(tmp_data_dir)):
        assert not p.exists()
    assert foreign.exists()
    assert foreign.read_text() == "important user data"


def test_handle_clear_nothing_to_do(tmp_data_dir, capsys):
    """Empty data dir → exit 0 with a friendly message; never prompts."""
    args = argparse.Namespace(rag_cmd="clear", yes=False)
    sentinel_prompted = {"called": False}

    def _input(_):
        sentinel_prompted["called"] = True
        return "n"

    code = cli.handle_rag(args, _input=_input)
    assert code == 0
    assert sentinel_prompted["called"] is False
    assert "nothing to remove" in capsys.readouterr().out


def test_handle_clear_works_outside_safe_path(monkeypatch, tmp_path, capsys):
    """The denylist heuristic is gone. With artifact-only unlink, even a
    `--yes` clear on an unusual HERMES_RAG_DATA_DIR is bounded to the four
    files we own — non-artifacts stay put. Exercises a non-default data dir."""
    odd_dir = tmp_path / "weird"
    odd_dir.mkdir()
    monkeypatch.setenv("HERMES_RAG_DATA_DIR", str(odd_dir))
    _seed_artifacts(odd_dir)
    foreign = odd_dir / "keep_me.txt"
    foreign.write_text("kept")

    args = argparse.Namespace(rag_cmd="clear", yes=True)
    code = cli.handle_rag(args)
    assert code == 0
    assert foreign.exists()


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
