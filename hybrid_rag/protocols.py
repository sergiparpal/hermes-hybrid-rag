"""Runtime-checkable protocols for the optional/swappable subsystems.

Concrete classes in this package (``embeddings.Embedder``,
``extractors.MarkdownExtractor`` …) and test stubs both satisfy these
protocols structurally — no inheritance required.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .models import Parent


@runtime_checkable
class EmbedderProtocol(Protocol):
    """A text → dense-vector encoder.

    ``dim`` may be ``None`` when the model hasn't been loaded yet AND wasn't
    pre-registered in ``config.EMBED_MODEL_DIMS``. Once any ``encode`` call
    has run it must be populated.
    """

    @property
    def model_name(self) -> str: ...

    @property
    def dim(self) -> int | None: ...

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray: ...


@runtime_checkable
class ParentExtractor(Protocol):
    """Extracts ``Parent`` units from a file.

    Implementers declare the file suffixes they handle and provide an
    ``extract(path)`` that returns the list of parents (already cap-enforced
    via ``parents._enforce_parent_cap``). Returning ``[]`` for an
    empty/whitespace-only file is fine; raising ``IndexingError`` is the
    contract for "I should handle this but a precondition is missing"
    (e.g. ``pypdf`` not installed).
    """

    @property
    def supported_suffixes(self) -> tuple[str, ...]: ...

    def extract(self, path: Path) -> list[Parent]: ...
