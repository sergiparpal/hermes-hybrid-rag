"""Concrete ``ParentExtractor`` implementations and a registry.

The free functions in ``parents.py`` are the pure extraction algorithms —
they take strings (or paths, for PDF) and return ``list[Parent]``. The
classes here wrap them with the "open the file" plumbing and expose the
suffix → extractor mapping that ``indexing.py`` walks.

To add a new extractor (e.g. ``.docx``), implement the ``ParentExtractor``
protocol and register an instance with the default registry — no edit to
``indexing.py`` required.
"""
from __future__ import annotations

from pathlib import Path

from .models import Parent
from .parents import extract_md, extract_pdf, extract_txt


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


class MarkdownExtractor:
    supported_suffixes: tuple[str, ...] = (".md",)

    def extract(self, path: Path) -> list[Parent]:
        return extract_md(_read_text(path))


class TextExtractor:
    supported_suffixes: tuple[str, ...] = (".txt",)

    def extract(self, path: Path) -> list[Parent]:
        return extract_txt(_read_text(path))


class PdfExtractor:
    supported_suffixes: tuple[str, ...] = (".pdf",)

    def extract(self, path: Path) -> list[Parent]:
        return extract_pdf(path)


class ExtractorRegistry:
    """Suffix-keyed registry of ``ParentExtractor`` instances.

    ``register`` is idempotent for the same instance; a later registration
    with the same suffix replaces the earlier extractor (the explicit ask
    wins).
    """

    def __init__(self):
        self._by_suffix: dict[str, "ParentExtractor"] = {}  # noqa: F821

    def register(self, extractor) -> None:
        for suffix in extractor.supported_suffixes:
            self._by_suffix[suffix.lower()] = extractor

    @property
    def supported_suffixes(self) -> frozenset[str]:
        return frozenset(self._by_suffix)

    def get(self, path: Path):
        return self._by_suffix.get(path.suffix.lower())

    def extract(self, path: Path) -> list[Parent]:
        extractor = self.get(path)
        return extractor.extract(path) if extractor is not None else []


def _build_default_registry() -> ExtractorRegistry:
    reg = ExtractorRegistry()
    reg.register(MarkdownExtractor())
    reg.register(TextExtractor())
    reg.register(PdfExtractor())
    return reg


DEFAULT_REGISTRY = _build_default_registry()
