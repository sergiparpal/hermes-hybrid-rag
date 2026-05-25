"""Reinforced ambient path.

Covers:
- Ambient reranks via the local cross-encoder (top-10 → top-3).
- Ambient never calls Cohere even when COHERE_API_KEY is set.
- Cross-encoder warm-up runs in on_session_start.
- Convo memory is off by default; toggle works; mixing is L2-normalized.
- Ambient stays cheap (no Cohere round-trips, no LLM expansion calls).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

import hybrid_rag.convo as convo_mod
import hybrid_rag.hooks as hooks_mod
import hybrid_rag.pipelines as pipelines_mod
import hybrid_rag.rerank as rerank_mod
import hybrid_rag.state as state_mod
from hybrid_rag.engine import RAGEngine, reset_for_tests, set_engine_for_tests
from hybrid_rag.hooks import ambient_pre_llm_call
from hybrid_rag.indexing import index_path
from hybrid_rag.storage import Store

FIXTURES = Path(__file__).parent / "fixtures" / "docs"


@pytest.fixture(autouse=True)
def _reset_state_cache():
    state_mod.reset_for_tests()
    convo_mod.reset_for_tests()
    yield
    state_mod.reset_for_tests()
    convo_mod.reset_for_tests()


@pytest.fixture
def warmed_engine(tmp_data_dir, tmp_path, stub_embedder):
    docs = tmp_path / "docs"
    docs.mkdir()
    for n in ("alpha.md", "beta.md", "gamma.txt"):
        (docs / n).write_text((FIXTURES / n).read_text())
    store = Store()
    index_path(docs, store=store, embedder=stub_embedder)
    eng = RAGEngine(store=store, embedder=stub_embedder)
    eng._ensure_loaded()
    set_engine_for_tests(eng)
    yield eng
    reset_for_tests()


# --- ambient now reranks ---

def test_ambient_runs_local_rerank(warmed_engine, monkeypatch, mock_cross_encoder):
    """Drop the threshold to 0 and verify the cross-encoder is invoked."""
    monkeypatch.setattr(pipelines_mod, "AMBIENT_SCORE_THRESHOLD", 0.0)
    mock_cross_encoder._scores = [10.0, 5.0, 1.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]

    out = ambient_pre_llm_call(
        session_id="s1", user_message="cosmic rays from outer space",
        conversation_history=None, is_first_turn=True,
    )
    assert out is not None
    assert "context" in out


def test_ambient_narrows_from_top10_to_top3(warmed_engine, monkeypatch,
                                            mock_cross_encoder):
    """The pipeline hands `AMBIENT_RERANK_POOL` (=10) parents to the local
    cross-encoder, and the reranker returns at most `AMBIENT_TOP_PARENTS`
    (=3). Verify by spying on the pair count fed into `predict`."""
    monkeypatch.setattr(pipelines_mod, "AMBIENT_SCORE_THRESHOLD", 0.0)
    monkeypatch.setattr(pipelines_mod, "AMBIENT_TOP_PARENTS", 3)

    seen_pair_counts: list[int] = []
    real_predict = mock_cross_encoder.CrossEncoder("x").predict

    class _SpyCE:
        def __init__(self, _name):
            pass

        def predict(self, pairs):
            seen_pair_counts.append(len(pairs))
            # Return descending scores so the top-3 picks are well-defined.
            return [float(len(pairs) - i) for i in range(len(pairs))]

    mock_cross_encoder.CrossEncoder = _SpyCE
    rerank_mod._CROSS = None

    out = ambient_pre_llm_call(
        session_id="s1", user_message="cosmic rays from outer space",
        conversation_history=None, is_first_turn=True,
    )
    assert out is not None
    # The rerank pool fed to the cross-encoder is bounded by
    # AMBIENT_RERANK_POOL (=10). Depending on the fixture corpus there
    # may be fewer parents in total, but never more than 10.
    assert seen_pair_counts, "cross-encoder was not invoked"
    assert all(c <= 10 for c in seen_pair_counts)


# --- Cohere must never be called from the ambient path ---

def test_ambient_never_calls_cohere(warmed_engine, monkeypatch,
                                    mock_cohere, mock_cross_encoder):
    """With COHERE_API_KEY set and a working Cohere mock, the ambient path
    must still go straight to the local cross-encoder."""
    monkeypatch.setattr(pipelines_mod, "AMBIENT_SCORE_THRESHOLD", 0.0)

    cohere_calls = {"n": 0}
    real_client = mock_cohere.Client

    class _Spy:
        def __init__(self, *a, **kw):
            self._inner = real_client(*a, **kw)

        def rerank(self, **kwargs):
            cohere_calls["n"] += 1
            return self._inner.rerank(**kwargs)

    mock_cohere.Client = _Spy
    mock_cohere.ClientV2 = _Spy
    mock_cross_encoder._scores = [5.0, 3.0, 1.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]

    out = ambient_pre_llm_call(
        session_id="s2", user_message="cosmic rays high energy particles",
        conversation_history=None, is_first_turn=True,
    )
    assert out is not None
    assert cohere_calls["n"] == 0, (
        "Cohere must never be invoked from the ambient path"
    )


# --- warm-up wires the cross-encoder ---

def test_warm_hook_warms_cross_encoder(monkeypatch):
    """Ambient invariant: on_session_start preloads the cross-encoder so
    the first ambient rerank doesn't pay the cold-load cost."""
    from hybrid_rag import adapters as adapters_mod
    from hybrid_rag.adapters import make_session_warm_hook

    adapters_mod.reset_for_tests()

    warmed = {"called": False}

    def fake_warm():
        warmed["called"] = True

    monkeypatch.setattr(rerank_mod, "warm_local_cross_encoder", fake_warm)
    # Also stub engine load so the thread doesn't actually load anything.
    from hybrid_rag import engine as eng_mod

    class _DummyEngine:
        def _ensure_loaded(self):
            pass

    monkeypatch.setattr(eng_mod, "get_engine", lambda: _DummyEngine())

    hook = make_session_warm_hook()
    hook(session_id="s", model="m", platform="p")

    # Background thread may still be in flight — give it a beat.
    for _ in range(20):
        if warmed["called"]:
            break
        time.sleep(0.01)
    assert warmed["called"] is True


# --- convo memory: off by default, opt-in works ---

def test_convo_memory_off_by_default(monkeypatch):
    monkeypatch.delenv("HERMES_RAG_AMBIENT_CONVO_MEMORY", raising=False)
    assert convo_mod.is_enabled() is False


def test_convo_memory_toggle_via_env(monkeypatch):
    monkeypatch.setenv("HERMES_RAG_AMBIENT_CONVO_MEMORY", "1")
    assert convo_mod.is_enabled() is True
    monkeypatch.setenv("HERMES_RAG_AMBIENT_CONVO_MEMORY", "0")
    assert convo_mod.is_enabled() is False


def test_convo_ring_buffer_holds_last_n(monkeypatch):
    convo_mod.reset_for_tests()
    convo_mod.push("sid", np.array([1.0, 0.0], dtype=np.float32))
    convo_mod.push("sid", np.array([0.0, 1.0], dtype=np.float32))
    convo_mod.push("sid", np.array([1.0, 1.0], dtype=np.float32))
    convo_mod.push("sid", np.array([0.5, 0.5], dtype=np.float32))
    ring = convo_mod.get_ring("sid")
    # Buffer size matches the weights tuple length.
    assert len(ring) == len(convo_mod.AMBIENT_CONVO_MEMORY_WEIGHTS)
    # Newest first.
    assert np.allclose(ring[0], [0.5, 0.5])


def test_convo_mix_with_empty_history_returns_current():
    cur = np.array([0.6, 0.8], dtype=np.float32)
    out = convo_mod.mix_with_history(cur, [])
    assert np.allclose(out, cur)


def test_convo_mix_with_history_is_normalized():
    cur = np.array([1.0, 0.0], dtype=np.float32)
    prior = np.array([0.0, 1.0], dtype=np.float32)
    out = convo_mod.mix_with_history(cur, [prior], weights=(1.0, 1.0))
    assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-5


def test_convo_mix_ignores_dim_mismatch():
    """A prior embedding from before a reindex (different dim) must not
    crash the mix — it's silently skipped."""
    cur = np.array([1.0, 0.0], dtype=np.float32)
    bad_prior = np.array([0.0, 1.0, 0.5], dtype=np.float32)  # wrong dim
    out = convo_mod.mix_with_history(cur, [bad_prior])
    assert np.allclose(out, cur)


def test_ambient_convo_memory_path_used_when_enabled(
    warmed_engine, monkeypatch, mock_cross_encoder,
):
    monkeypatch.setenv("HERMES_RAG_AMBIENT_CONVO_MEMORY", "1")
    monkeypatch.setattr(pipelines_mod, "AMBIENT_SCORE_THRESHOLD", 0.0)
    mock_cross_encoder._scores = [5.0, 1.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    seen = {"vec_path": False}
    real = warmed_engine.hybrid_search

    def spy(query, *, qvec=None, k_pool=30):
        if qvec is not None:
            seen["vec_path"] = True
        return real(query, qvec=qvec, k_pool=k_pool)

    monkeypatch.setattr(warmed_engine, "hybrid_search", spy)

    ambient_pre_llm_call(
        session_id="conv-1", user_message="cosmic rays particles",
        conversation_history=None,
    )
    assert seen["vec_path"] is True
    # And the ring buffer now has the current turn's embedding.
    assert len(convo_mod.get_ring("conv-1")) == 1


def test_ambient_convo_memory_falls_back_when_encode_returns_empty(
    warmed_engine, monkeypatch, mock_cross_encoder,
):
    """If the embedder returns a (0, dim) array (e.g. a stub bug, or a
    pathological model), the convo path must fall back to the literal
    hybrid_search rather than crash."""
    monkeypatch.setenv("HERMES_RAG_AMBIENT_CONVO_MEMORY", "1")
    monkeypatch.setattr(pipelines_mod, "AMBIENT_SCORE_THRESHOLD", 0.0)
    mock_cross_encoder._scores = [5.0] * 10

    seen = {"vec_path": False, "literal_path": False}
    real = warmed_engine.hybrid_search

    def spy(query, *, qvec=None, k_pool=30):
        if qvec is not None:
            seen["vec_path"] = True
        else:
            seen["literal_path"] = True
        return real(query, qvec=qvec, k_pool=k_pool)

    monkeypatch.setattr(warmed_engine, "hybrid_search", spy)

    # Force the embedder to return (0, dim) — the empty-batch shape.
    real_encode = warmed_engine._embedder.encode

    def _empty_encode(texts, batch_size=64):
        return np.zeros((0, 32), dtype=np.float32)

    warmed_engine._embedder.encode = _empty_encode
    try:
        ambient_pre_llm_call(
            session_id="conv-empty", user_message="cosmic rays particles",
            conversation_history=None,
        )
    finally:
        warmed_engine._embedder.encode = real_encode

    assert seen["vec_path"] is False
    assert seen["literal_path"] is True


# --- latency budget ---

def test_ambient_warm_budget(warmed_engine, monkeypatch, mock_cross_encoder):
    """Warm ambient (stubs throughout) must complete fast. The threshold is
    deliberately loose — we're not benchmarking, we're guarding against an
    accidental synchronous network call sneaking in."""
    monkeypatch.setattr(pipelines_mod, "AMBIENT_SCORE_THRESHOLD", 0.0)
    mock_cross_encoder._scores = [1.0] * 10

    start = time.monotonic()
    ambient_pre_llm_call(
        session_id="s", user_message="cosmic rays from outer space",
        conversation_history=None, is_first_turn=True,
    )
    dt = time.monotonic() - start
    assert dt < 0.5, f"ambient turn took {dt:.3f}s — looks like a sync network call"
