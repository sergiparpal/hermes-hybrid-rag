"""Shared pytest fixtures and sys.path setup for the test suite."""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

# Add project root so `import hybrid_rag.tools` works during tests.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    """Point HERMES_RAG_DATA_DIR at a tmp_path. Yields the path."""
    monkeypatch.setenv("HERMES_RAG_DATA_DIR", str(tmp_path))
    yield tmp_path


class StubEmbedder:
    """Deterministic, dependency-free embedder used in tests.

    Each token contributes a positional component; the result is L2-normalized.
    Two texts sharing tokens get higher cosine similarity than disjoint pairs,
    which is enough to exercise the retrieval pipeline. Conforms structurally
    to ``EmbedderProtocol``.
    """

    DIM = 32

    def __init__(self, model_name: str = "stub"):
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self.DIM

    def encode(self, texts, batch_size: int = 64) -> np.ndarray:
        out = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in str(t).lower().split():
                slot = (hash(tok) & 0xFFFFFFFF) % self.DIM
                out[i, slot] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (out / norms).astype(np.float32)


@pytest.fixture
def stub_embedder():
    return StubEmbedder()


@pytest.fixture
def fake_ctx():
    """Records every register_* call. Signatures mirror the verified Hermes
    PluginContext (see HERMES_API.md)."""
    class _Ctx:
        def __init__(self):
            self.tools = []
            self.hooks = []
            self.cli_commands = []
            self.commands = []
            self.skills = []

        def register_tool(self, name, toolset, schema, handler,
                          check_fn=None, requires_env=None, is_async=False,
                          description="", emoji=""):
            self.tools.append({
                "name": name, "toolset": toolset, "schema": schema,
                "handler": handler, "check_fn": check_fn,
                "requires_env": requires_env, "is_async": is_async,
                "description": description, "emoji": emoji,
            })

        def register_hook(self, hook_name, callback):
            self.hooks.append({"event": hook_name, "fn": callback})

        def register_cli_command(self, name, help, setup_fn,
                                 handler_fn=None, description=""):
            self.cli_commands.append({
                "name": name, "help": help, "setup": setup_fn,
                "handler": handler_fn, "description": description,
            })

        def register_command(self, name, handler, description="", args_hint=""):
            self.commands.append({
                "name": name, "handler": handler,
                "description": description, "args_hint": args_hint,
            })

        def register_skill(self, name, path, description=""):
            self.skills.append({
                "name": name, "path": path, "description": description,
            })

    return _Ctx()


@pytest.fixture
def mock_anthropic(monkeypatch):
    """Install a fake anthropic module into sys.modules. Returns a controllable
    response object the test sets ahead of expand_query()."""
    mod = types.ModuleType("anthropic")

    class _Msg:
        def __init__(self, text):
            self.content = [types.SimpleNamespace(text=text)]

    class _Messages:
        def __init__(self):
            self.next_text = '{"paraphrases": ["a", "b", "c"], "hyde": "h"}'
            self.raise_with = None
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.raise_with is not None:
                raise self.raise_with
            return _Msg(self.next_text)

    class _Client:
        def __init__(self, *a, **kw):
            self.messages = _Messages()

    mod.Anthropic = _Client
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    # The shared `_anthropic` module caches the client — reset it so each test
    # sees a fresh constructor (tests rebind `mod.Anthropic` mid-run to swap
    # response shapes).
    try:
        from hybrid_rag import _anthropic
        _anthropic.reset_for_tests()
    except ImportError:
        pass
    return mod


@pytest.fixture
def mock_cohere(monkeypatch):
    """Install a fake cohere module. Test sets `mod._scores` to control the
    rerank result; if `mod._raise` is set, it raises instead."""
    mod = types.ModuleType("cohere")

    class _RerankResult:
        def __init__(self, idx, score):
            self.index = idx
            self.relevance_score = score

    class _RerankResponse:
        def __init__(self, results):
            self.results = results

    class _Client:
        def __init__(self, *a, **kw):
            pass

        def rerank(self, **kwargs):
            if mod._raise is not None:
                raise mod._raise
            documents = kwargs.get("documents", [])
            top_n = kwargs.get("top_n", len(documents))
            scores = mod._scores or [1.0 - i * 0.1 for i in range(len(documents))]
            ordered = sorted(enumerate(scores), key=lambda kv: -kv[1])[:top_n]
            return _RerankResponse([_RerankResult(i, s) for i, s in ordered])

    mod.Client = _Client
    mod.ClientV2 = _Client
    mod._scores = None
    mod._raise = None
    monkeypatch.setitem(sys.modules, "cohere", mod)
    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    return mod


@pytest.fixture
def mock_cross_encoder(monkeypatch):
    """Install a fake sentence_transformers module exposing CrossEncoder.
    Test sets `mod._scores` (list aligned with the input pairs); `mod._raise`
    forces an error path."""
    st = sys.modules.get("sentence_transformers")
    if st is None:
        st = types.ModuleType("sentence_transformers")
        monkeypatch.setitem(sys.modules, "sentence_transformers", st)

    class _CrossEncoder:
        def __init__(self, model_name):
            self.model_name = model_name

        def predict(self, pairs):
            if getattr(st, "_raise", None) is not None:
                raise st._raise
            scores = getattr(st, "_scores", None)
            if scores is not None:
                return list(scores)
            return [-float(i) for i in range(len(pairs))]

    st.CrossEncoder = _CrossEncoder
    st._scores = None
    st._raise = None
    # Drop the rerank module's cached CrossEncoder so each test gets a fresh
    # one bound to this fixture's scores.
    try:
        import hybrid_rag.rerank as _rr
        _rr._CROSS = None
    except ImportError:
        pass
    return st
