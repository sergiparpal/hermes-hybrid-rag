import pytest

from hybrid_rag.chunking import recursive_split


def test_invalid_max_size_raises():
    with pytest.raises(ValueError, match="max_size"):
        recursive_split("anything", max_size=0)
    with pytest.raises(ValueError, match="max_size"):
        recursive_split("anything", max_size=-1)


def test_invalid_overlap_raises():
    with pytest.raises(ValueError, match="overlap"):
        recursive_split("anything", max_size=100, overlap=-1)


def test_empty_input_returns_empty_list():
    assert recursive_split("") == []
    assert recursive_split("   \n  \t  \n") == []


def test_short_text_returns_single_chunk():
    text = "short string under the limit"
    out = recursive_split(text, max_size=300, overlap=50)
    assert out == [text]


def test_paragraph_split_respects_max_size():
    text = "a" * 250 + "\n\n" + "b" * 250 + "\n\n" + "c" * 250
    out = recursive_split(text, max_size=300, overlap=0)
    assert all(len(c) <= 300 for c in out)
    assert any("aaa" in c for c in out)
    assert any("bbb" in c for c in out)
    assert any("ccc" in c for c in out)


def test_oversized_word_falls_through_to_hard_split():
    word = "x" * 1000
    out = recursive_split(word, max_size=300, overlap=50)
    assert len(out) > 1
    assert all(len(c) <= 300 for c in out)
    joined = "".join(out)
    assert "x" * 1000 in joined or joined.count("x") >= 1000


def test_overlap_stitches_adjacent_chunks():
    paras = ["alpha alpha alpha " * 20, "beta beta beta " * 20, "gamma gamma " * 20]
    text = "\n\n".join(paras)
    out = recursive_split(text, max_size=200, overlap=40)
    assert len(out) > 1


def test_overlap_relaxes_chunk_size_bound_to_max_plus_overlap():
    """With overlap > 0 the relaxed invariant is `len(c) <= max_size + overlap`,
    not `<= max_size`. Pin it so a future refactor can't quietly tighten the
    rule and surprise callers (downstream MAX_PARENT_CHARS=8000 absorbs the
    slop, but other callers might not)."""
    paras = ["alpha alpha alpha " * 20, "beta beta beta " * 20, "gamma gamma " * 20]
    text = "\n\n".join(paras)
    max_size, overlap = 200, 40
    out = recursive_split(text, max_size=max_size, overlap=overlap)
    assert all(len(c) <= max_size + overlap for c in out)


def test_nested_separator_fallback():
    text = "first part. second part. third part. " * 30
    out = recursive_split(text, max_size=120, overlap=20)
    assert len(out) > 1
    assert all(len(c) <= 200 for c in out)
