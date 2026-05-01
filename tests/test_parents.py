from pathlib import Path

import pytest

from hierarchical_rag import parents
from hierarchical_rag.parents import (
    Parent,
    _enforce_parent_cap,
    extract_md,
    extract_pdf,
    extract_txt,
)

FIXTURES = Path(__file__).parent / "fixtures" / "docs"


def test_extract_md_splits_on_h2_headings():
    text = (FIXTURES / "alpha.md").read_text()
    out = extract_md(text)
    assert len(out) == 3
    assert all(p.kind == "section" for p in out)
    titles = [p.title for p in out]
    assert any("Section one" in t for t in titles)
    assert any("Section two" in t for t in titles)
    assert any("Section three" in t for t in titles)
    assert "cosmic ray" in out[1].text.lower()


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
