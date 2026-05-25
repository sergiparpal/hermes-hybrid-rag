from pathlib import Path

import pytest

from hybrid_rag.extractors import (
    DEFAULT_REGISTRY,
    ExtractorRegistry,
    MarkdownExtractor,
    PdfExtractor,
    TextExtractor,
)
from hybrid_rag.models import Parent


def test_default_registry_covers_md_txt_pdf():
    assert DEFAULT_REGISTRY.supported_suffixes == frozenset({".md", ".txt", ".pdf"})


def test_default_registry_dispatches_by_suffix(tmp_path):
    md = tmp_path / "x.md"
    md.write_text("# Title\n\n## Section\nbody text long enough to chunk.")
    out = DEFAULT_REGISTRY.extract(md)
    assert out
    assert all(isinstance(p, Parent) for p in out)


def test_unknown_suffix_returns_empty(tmp_path):
    weird = tmp_path / "x.unknown"
    weird.write_text("anything")
    assert DEFAULT_REGISTRY.extract(weird) == []


def test_register_extends_supported_suffixes():
    """A user-supplied extractor for a new suffix registers cleanly without
    editing indexing.py — the OCP win of the registry."""

    class JsonExtractor:
        supported_suffixes = (".json",)

        def extract(self, path):
            return [Parent(kind="paragraph_group", title=None,
                           text=path.read_text())]

    reg = ExtractorRegistry()
    reg.register(MarkdownExtractor())
    reg.register(TextExtractor())
    reg.register(PdfExtractor())
    assert ".json" not in reg.supported_suffixes

    reg.register(JsonExtractor())
    assert ".json" in reg.supported_suffixes


def test_register_replaces_for_same_suffix():
    """The explicit ask wins: re-registering for a suffix replaces the old
    extractor. Lets a deployment override a built-in (e.g. swap MarkdownExtractor
    for a fence-aware variant)."""

    class _AltMd:
        supported_suffixes = (".md",)
        extracted = False

        def extract(self, path):
            type(self).extracted = True
            return []

    reg = ExtractorRegistry()
    reg.register(MarkdownExtractor())
    reg.register(_AltMd())
    assert reg.get(Path("a.md")).__class__ is _AltMd


def test_suffix_lookup_is_case_insensitive(tmp_path):
    upper = tmp_path / "X.MD"
    upper.write_text("# Hello\n\n## S\nbody.")
    out = DEFAULT_REGISTRY.extract(upper)
    assert out, ".MD must dispatch the same as .md"
