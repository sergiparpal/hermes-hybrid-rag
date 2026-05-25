import json

import hybrid_rag.state as state
from hybrid_rag.config import toggles_path


def setup_function(_):
    state.reset_for_tests()


def test_default_true_when_file_missing(tmp_data_dir):
    assert state.is_ambient_enabled() is True


def test_set_get_round_trip(tmp_data_dir):
    state.set_ambient(False)
    state.reset_for_tests()
    assert state.is_ambient_enabled() is False
    state.set_ambient(True)
    state.reset_for_tests()
    assert state.is_ambient_enabled() is True


def test_session_scoped_overrides_default(tmp_data_dir):
    state.set_ambient(False)               # default off
    state.reset_for_tests()
    state.set_ambient(True, session_id="abc")  # session on
    state.reset_for_tests()
    assert state.is_ambient_enabled() is False
    assert state.is_ambient_enabled(session_id="abc") is True
    assert state.is_ambient_enabled(session_id="other") is False


def test_corrupted_file_fails_open(tmp_data_dir):
    p = toggles_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not json")
    state.reset_for_tests()
    assert state.is_ambient_enabled() is True


def test_atomic_write_no_tmp_left_behind(tmp_data_dir):
    state.set_ambient(True)
    p = toggles_path()
    assert p.exists()
    assert not p.with_suffix(p.suffix + ".tmp").exists()
    assert json.loads(p.read_text())["_default"] is True
