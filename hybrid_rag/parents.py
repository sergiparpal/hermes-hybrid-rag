"""Parent unit extraction.

A *parent* is the unit returned to the caller (markdown section, PDF page, or
paragraph group). Chunks are the search space; parents are what we hand back
once a chunk hits.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from .config import MAX_PARENT_CHARS, MAX_PDF_PAGE_CHARS, PREAMBLE_MIN_CHARS
from .models import Parent

PdfReader = None  # patched by tests; lazily imported in extract_pdf


class IndexingError(RuntimeError):
    """Raised when an indexing precondition is missing (e.g. pypdf for PDFs)."""


_H2_RE = re.compile(r"^##\s+", re.MULTILINE)
_H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_PARA_SPLIT_RE = re.compile(r"\n\s*\n")


def _build_preamble(prefix: str, min_body_chars: int) -> Parent | None:
    """Capture text before the first `##` heading as a synthetic parent.

    Returns None if the prefix is empty/whitespace, or if the body (after
    lifting a leading `# H1` line out as the title) is shorter than
    ``min_body_chars`` — that case is almost always boilerplate (a lone title
    line, a one-line status note) and not worth a parent of its own.

    H1 handling: ``_H1_RE.search`` returns the *first* H1 in the preamble. If
    the document opens with one H1 followed by paragraphs that's the common
    case and works fine. Multiple H1s before the first `##` are unusual; only
    the first becomes the preamble title, and any later H1 lines stay in the
    body verbatim. Newlines around the spliced H1 are preserved, so the body
    reads naturally even when the H1 sits mid-prefix.
    """
    if not prefix or not prefix.strip():
        return None
    title: str | None = None
    h1 = _H1_RE.search(prefix)
    if h1 is not None:
        title = h1.group(1).strip()
        # body is everything else in the prefix, with the H1 line removed
        body = (prefix[: h1.start()] + prefix[h1.end():]).strip()
    else:
        body = prefix.strip()
    if len(body) < min_body_chars:
        return None
    full = (title + "\n" + body) if title else body
    return Parent(kind="preamble", title=title, text=full.strip())


def extract_md(text: str, preamble_min_chars: int = PREAMBLE_MIN_CHARS) -> list[Parent]:
    """Split on `## ` (level-2) lines. Each parent's title is the literal
    heading line; kind="section". If zero level-2 headings, defer to extract_txt.

    Text preceding the first `##` heading (TL;DRs, abstracts, content under a
    lone `# H1`) is captured as a synthetic "preamble" parent when its body
    clears ``preamble_min_chars`` — otherwise it is dropped to avoid indexing
    boilerplate.

    **Known v0.1 limitation:** the regex is line-oriented and not aware of
    fenced code blocks (``` ... ``` / ~~~), so an unindented ``##`` *inside* a
    fenced block is treated as a section break. Real-world Python/shell/MDX
    samples can trip this. A code-fence-aware splitter is planned for v0.2.
    """
    if text is None or not text.strip():
        return []
    matches = list(_H2_RE.finditer(text))
    if not matches:
        return extract_txt(text)

    parents: list[Parent] = []
    starts = [m.start() for m in matches]
    preamble = _build_preamble(text[: starts[0]], preamble_min_chars)
    if preamble is not None:
        parents.append(preamble)
    starts.append(len(text))
    for i, start in enumerate(starts[:-1]):
        end = starts[i + 1]
        section = text[start:end]
        first_nl = section.find("\n")
        if first_nl == -1:
            title_line = section.strip()
            body = ""
        else:
            title_line = section[:first_nl].strip()
            body = section[first_nl + 1 :].rstrip()
        full = (title_line + ("\n" + body if body else "")).strip()
        parents.append(Parent(kind="section", title=title_line, text=full))
    return _enforce_parent_cap(parents)


def extract_txt(text: str, target_chars: int = 2000) -> list[Parent]:
    """Paragraph groups by `\\n\\s*\\n` regex; greedy pack to ~target_chars;
    kind="paragraph_group"; title=None."""
    if text is None or not text.strip():
        return []
    paras = [p for p in _PARA_SPLIT_RE.split(text) if p.strip()]

    parents: list[Parent] = []
    buf: list[str] = []
    buf_len = 0
    for p in paras:
        p_len = len(p)
        if buf and buf_len + p_len + 2 > target_chars:
            parents.append(Parent(kind="paragraph_group", title=None,
                                  text="\n\n".join(buf).strip()))
            buf = [p]
            buf_len = p_len
        else:
            buf.append(p)
            buf_len += p_len + 2
    if buf:
        parents.append(Parent(kind="paragraph_group", title=None,
                              text="\n\n".join(buf).strip()))
    return _enforce_parent_cap(parents)


def extract_pdf(path: Path) -> list[Parent]:
    """One parent per page with kind='page'. Test-friendly: if the module-level
    `PdfReader` was monkey-patched, that's used; otherwise we try to import
    pypdf and raise IndexingError if it's missing."""
    reader_cls = PdfReader
    if reader_cls is None:
        try:
            from pypdf import PdfReader as _Reader
        except ImportError as e:
            raise IndexingError(
                "PDF support requires `pypdf`. Install with `pip install pypdf`."
            ) from e
        reader_cls = _Reader

    reader = reader_cls(str(path))
    parents: list[Parent] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if not text.strip():
            continue
        # Cap per-page text. Malformed PDFs can return amplified strings
        # that blow up downstream chunking / embedding cost.
        if len(text) > MAX_PDF_PAGE_CHARS:
            text = text[:MAX_PDF_PAGE_CHARS]
        parents.append(Parent(kind="page", title=f"Page {i + 1}", text=text,
                              page_no=i + 1))
    return _enforce_parent_cap(parents)


def _enforce_parent_cap(parents: list[Parent], max_chars: int = MAX_PARENT_CHARS) -> list[Parent]:
    """Splits oversized parents on paragraph/line boundaries."""
    out: list[Parent] = []
    for p in parents:
        if len(p.text) <= max_chars:
            out.append(p)
            continue
        out.extend(_split_oversized(p, max_chars))
    return out


# --- oversized-parent split: a three-tier cascade ------------------------


def _greedy_pack(parts: Iterable[str], max_chars: int, joiner: str) -> list[str]:
    """Greedy pack `parts` into pieces of ≤``max_chars``, joining adjacent
    parts with ``joiner``. Single parts that already exceed ``max_chars``
    are yielded as-is — the caller cascades to a finer boundary."""
    pieces: list[str] = []
    buf = ""
    for part in parts:
        candidate = buf + (joiner if buf else "") + part
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            if buf:
                pieces.append(buf)
            buf = part
    if buf:
        pieces.append(buf)
    return pieces


def _hard_chunks(text: str, max_chars: int) -> list[str]:
    """Final fallback: slice into fixed-size pieces. Used only when a single
    line exceeds ``max_chars`` and there's no natural boundary left."""
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def _flatten(chunks_of_chunks: Iterable[list[str]]) -> list[str]:
    return [c for group in chunks_of_chunks for c in group]


def _split_oversized(parent: Parent, max_chars: int) -> list[Parent]:
    """Tier 1: paragraph boundaries (``\\n\\s*\\n``).
    Tier 2: if a packed piece still exceeds ``max_chars``, re-pack on lines.
    Tier 3: if a single line still exceeds ``max_chars``, hard-chunk it.
    """
    by_para = _greedy_pack(_PARA_SPLIT_RE.split(parent.text), max_chars, "\n\n")
    by_line = _flatten(
        _greedy_pack(p.split("\n"), max_chars, "\n") if len(p) > max_chars else [p]
        for p in by_para
    )
    pieces = _flatten(
        _hard_chunks(p, max_chars) if len(p) > max_chars else [p]
        for p in by_line
    )
    return _wrap_as_parts(parent, pieces)


def _wrap_as_parts(parent: Parent, pieces: list[str]) -> list[Parent]:
    """Wrap each piece as a Parent inheriting the original's metadata.
    When the original carries a title, suffix each part with `(part N)`."""
    suffixed = len(pieces) > 1
    return [
        Parent(
            kind=parent.kind,
            title=(parent.title + f" (part {idx + 1})") if (parent.title and suffixed) else parent.title,
            text=piece,
            page_no=parent.page_no,
        )
        for idx, piece in enumerate(pieces)
    ]
