"""File-backed ambient toggle. Default ON; per-session overrides supported but
v0.1 only writes the `_default` slot from slash commands.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .config import toggles_path

_DEFAULT_KEY = "_default"
# Module-level cache. Deliberately unsynchronized: races between concurrent
# _load/_store can leave a stale dict in `_CACHE`, but the worst case is
# exactly one wrong toggle read within the 1-second TTL. `is_ambient_enabled`
# fails open (errors → True), so a torn read at most disables/enables ambient
# context for a single LLM turn — never raises, never corrupts the file.
# Adding a lock here would defeat the purpose of the cache (every read would
# serialize). Keep this contract in mind before refactoring.
_CACHE: dict | None = None
_CACHE_TS: float = 0.0
_CACHE_PATH: Path | None = None
_CACHE_TTL = 1.0  # seconds


def _load(path: Path) -> dict:
    global _CACHE, _CACHE_TS, _CACHE_PATH
    now = time.time()
    if (_CACHE is not None and _CACHE_PATH == path
            and (now - _CACHE_TS) < _CACHE_TTL):
        return _CACHE
    if not path.exists():
        data = {_DEFAULT_KEY: True}
    else:
        try:
            data = json.loads(path.read_text())
            if not isinstance(data, dict):
                data = {_DEFAULT_KEY: True}
        except Exception:
            # corrupted file: fail open
            data = {_DEFAULT_KEY: True}
    _CACHE = data
    _CACHE_TS = now
    _CACHE_PATH = path
    return data


def _store(path: Path, data: dict) -> None:
    global _CACHE, _CACHE_TS, _CACHE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(data))
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    _CACHE = data
    _CACHE_TS = time.time()
    _CACHE_PATH = path


def is_ambient_enabled(session_id: str | None = None) -> bool:
    try:
        data = _load(toggles_path())
        if session_id and session_id in data:
            return bool(data[session_id])
        return bool(data.get(_DEFAULT_KEY, True))
    except Exception:
        return True  # fail open


def set_ambient(on: bool, session_id: str | None = None) -> None:
    path = toggles_path()
    data = _load(path).copy()
    key = session_id or _DEFAULT_KEY
    data[key] = bool(on)
    _store(path, data)


def reset_for_tests() -> None:
    global _CACHE, _CACHE_TS, _CACHE_PATH
    _CACHE = None
    _CACHE_TS = 0.0
    _CACHE_PATH = None
