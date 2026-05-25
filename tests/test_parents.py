from pathlib import Path

import pytest

from hybrid_rag import parents
from hybrid_rag.models import Parent
from hybrid_rag.parents import (
    _enforce_parent_cap,
    extract_md,
    extract_pdf,
    extract_txt,
)

FIXTURES = Path(__file__).parent / "fixtures" / "docs"


def test_extract_md_splits_on_h2_headings():
    text = (FIXTURES / "alpha.md").read_text()
    out = extract_md(text)
    # alpha.md's preamble body is well under PREAMBLE_MIN_CHARS, so it is
    # dropped as boilerplate — only the three `##` sections remain.
    assert len(out) == 3
    assert all(p.kind == "section" for p in out)
    titles = [p.title for p in out]
    assert any("Section one" in t for t in titles)
    assert any("Section two" in t for t in titles)
    assert any("Section three" in t for t in titles)
    assert "cosmic ray" in out[1].text.lower()


def test_extract_md_preamble_emits_when_body_long_enough():
    intro = (
        "This document is a long-enough TL;DR that absolutely should not be "
        "silently discarded. " * 5
    )
    text = f"# Title\n\n{intro}\n\n## Section A\nbody A\n"
    out = extract_md(text)
    assert len(out) == 2
    pre = out[0]
    assert pre.kind == "preamble"
    assert pre.title == "Title"
    assert "TL;DR" in pre.text
    # the H1 line itself becomes the title; the body is what follows
    assert pre.text.startswith("Title\n")
    assert out[1].kind == "section"
    assert "Section A" in out[1].title


def test_extract_md_preamble_dropped_when_body_too_short():
    text = "# Title\n\nshort intro.\n\n## Section A\nbody A\n"
    out = extract_md(text)
    assert len(out) == 1
    assert out[0].kind == "section"


def test_extract_md_preamble_without_h1_has_no_title():
    body = "Plain prefix paragraph with substantial content. " * 6
    text = f"{body}\n\n## Section A\nbody A\n"
    out = extract_md(text)
    assert len(out) == 2
    assert out[0].kind == "preamble"
    assert out[0].title is None
    assert "Plain prefix paragraph" in out[0].text


def test_extract_md_preamble_threshold_is_overridable():
    text = "# T\n\nshort body.\n\n## S\nbody\n"
    # forced low threshold lets the short preamble through
    out = extract_md(text, preamble_min_chars=5)
    assert len(out) == 2
    assert out[0].kind == "preamble"
    assert out[0].title == "T"


def test_extract_md_splits_inside_fenced_code_block_known_limitation():
    """v0.1 limitation: the line-oriented `##` regex doesn't understand fenced
    code blocks, so an unindented `##` inside ``` ... ``` is treated as a
    section break. Pin the current behavior so a future refactor can't change
    it silently — the user-facing fix lands in v0.2."""
    text = (
        "# Title\n\n"
        "Intro is too short to become a preamble parent.\n\n"
        "## Real section\n"
        "Body of the real section.\n\n"
        "```python\n"
        "## comment that looks like a heading\n"
        "print('hello')\n"
        "```\n"
    )
    out = extract_md(text)
    titles = [p.title for p in out]
    # Two `##` lines were detected (the real one and the one inside the fence).
    assert any("Real section" in t for t in titles)
    assert any("comment that looks like a heading" in t for t in titles)


def test_extract_md_falls_back_when_no_h2():
    text = (FIXTURES / "beta.md").read_text()
    out = extract_md(text)
    assert all(p.kind == "paragraph_group" for p in out)
    assert len(out) >= 1


def test_extract_md_empty_text():
    assert extract_md("") == []
    assert extract_md("   \n  ") == []


def test_extract_txt_packs_paragraph_groups():
    text = (FIXTURES / "gamma.txt").read_text()
    out = extract_txt(text, target_chars=2000)
    assert len(out) >= 1
    assert all(p.kind == "paragraph_group" for p in out)
    assert all(p.title is None for p in out)
    assert all(len(p.text) <= 8000 for p in out)


def test_enforce_parent_cap_splits_long_parent():
    big = Parent(kind="section", title="Big", text="\n\n".join(["P" * 1000] * 12))
    out = _enforce_parent_cap([big], max_chars=2000)
    assert len(out) > 1
    assert all(len(p.text) <= 2000 for p in out)
    assert all(p.title and "Big" in p.title for p in out)


def test_extract_pdf_mocked(monkeypatch, tmp_path):
    class FakePage:
        def __init__(self, text):
            self._text = text

        def extract_text(self):
            return self._text

    class FakePdfReader:
        def __init__(self, path):
            assert Path(path).exists() or True
            self.pages = [FakePage("Page one body."),
                          FakePage("Page two body."),
                          FakePage("")]  # empty page should be skipped

    fake_path = tmp_path / "doc.pdf"
    fake_path.write_bytes(b"%PDF-fake")
    monkeypatch.setattr(parents, "PdfReader", FakePdfReader)

    out = extract_pdf(fake_path)
    assert len(out) == 2
    assert out[0].kind == "page"
    assert out[0].page_no == 1
    assert out[1].page_no == 2
    assert "Page one" in out[0].text
    assert all(p.title and p.title.startswith("Page ") for p in out)


def test_extract_pdf_missing_pypdf_raises(monkeypatch):
    import sys

    monkeypatch.setattr(parents, "PdfReader", None)
    monkeypatch.setitem(sys.modules, "pypdf", None)
    with pytest.raises(parents.IndexingError):
        extract_pdf(Path("/nonexistent.pdf"))
