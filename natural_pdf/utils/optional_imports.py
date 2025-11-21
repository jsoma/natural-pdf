"""Helpers for managing optional dependency imports consistently."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module, util
from typing import Any, Callable, Dict, Mapping, Optional, Sequence


@dataclass
class OptionalDependency:
    """Represents a lazily imported optional dependency."""

    module_name: str
    install_hints: Sequence[str]
    description: Optional[str] = None
    import_fn: Optional[Callable[[], Any]] = None
    _module: Optional[Any] = field(default=None, init=False)
    _available: Optional[bool] = field(default=None, init=False)

    def is_available(self) -> bool:
        if self._available is None:
            try:
                self._available = util.find_spec(self.module_name) is not None
            except (ImportError, ValueError):
                self._available = False
        return bool(self._available)

    def load(self) -> Any:
        if self._module is None:
            try:
                if self.import_fn is not None:
                    self._module = self.import_fn()
                else:
                    self._module = import_module(self.module_name)
            except ImportError as exc:  # pragma: no cover - error path
                hint = " or ".join(self.install_hints) or "pip install"
                raise ImportError(
                    f"Optional dependency '{self.module_name}' is not installed. Install with: {hint}"
                ) from exc
        return self._module

    def optional(self) -> Optional[Any]:
        return self.load() if self.is_available() else None


OPTIONAL_DEPENDENCIES: Dict[str, OptionalDependency] = {
    # OCR
    "easyocr": OptionalDependency(
        "easyocr",
        ('pip install "natural-pdf[ocr-ai]"', "npdf install easyocr"),
        "EasyOCR engine for OCR workflows.",
    ),
    "pikepdf": OptionalDependency(
        "pikepdf",
        ('pip install "natural-pdf[ocr-export]"',),
        "Required for creating searchable PDFs.",
    ),
    # Deskew
    "deskew": OptionalDependency(
        "deskew",
        ('pip install "natural-pdf[deskew]"',),
        "Deskew helpers used by page.deskew().",
    ),
    # Search / embeddings
    "sentence_transformers": OptionalDependency(
        "sentence_transformers",
        ('pip install "natural-pdf[embeddings]"', 'pip install "natural-pdf[search]"'),
        "Embedding models for semantic search.",
    ),
    "lancedb": OptionalDependency(
        "lancedb",
        ('pip install "natural-pdf[search]"',),
        "Vector database backend for semantic search.",
    ),
    "pyarrow": OptionalDependency(
        "pyarrow",
        ('pip install "natural-pdf[search]"',),
        "Columnar storage format required by LanceDB search backend.",
    ),
    # ML Core
    "torch": OptionalDependency(
        "torch",
        (
            'pip install "natural-pdf[ai-core]"',
            'pip install "natural-pdf[qa]"',
            'pip install "natural-pdf[classification]"',
        ),
        "PyTorch runtime used by QA/classification/layout engines.",
    ),
    "transformers": OptionalDependency(
        "transformers",
        (
            'pip install "natural-pdf[ai-core]"',
            'pip install "natural-pdf[qa]"',
            'pip install "natural-pdf[classification]"',
        ),
        "Hugging Face transformers for QA/classification.",
    ),
    "torchvision": OptionalDependency(
        "torchvision",
        ('pip install "natural-pdf[ai-core]"',),
        "TorchVision models/utilities used by perception pipelines.",
    ),
    "huggingface_hub": OptionalDependency(
        "huggingface_hub",
        ('pip install "natural-pdf[ai-core]"',),
        "Model hub utilities required by AI engines.",
    ),
    # Layout
    "doclayout_yolo": OptionalDependency(
        "doclayout_yolo",
        ('pip install "natural-pdf[layout-ai]"', "npdf install yolo"),
        "YOLO-based layout detection models.",
    ),
    "timm": OptionalDependency(
        "timm",
        ('pip install "natural-pdf[layout-ai]"',),
        "Required backbone models for layout detectors.",
    ),
    # OCR/export helpers
    "img2pdf": OptionalDependency(
        "img2pdf",
        ('pip install "natural-pdf[deskew]"',),
        "Image to PDF conversion helper used by deskew/save routines.",
    ),
    "openai": OptionalDependency(
        "openai",
        ('pip install "natural-pdf[ai-api]"', "npdf install ai"),
        "Client library for API-driven layout/QA engines.",
    ),
}


def require(name: str) -> Any:
    """Import and return the requested optional dependency, raising if missing."""

    dep = OPTIONAL_DEPENDENCIES.get(name)
    if dep is None:
        raise KeyError(f"Unknown optional dependency '{name}'")
    return dep.load()


def is_available(name: str) -> bool:
    dep = OPTIONAL_DEPENDENCIES.get(name)
    return dep.is_available() if dep is not None else False


def list_optional_dependencies() -> Mapping[str, Dict[str, Any]]:
    return {
        name: {
            "available": dep.is_available(),
            "install_hints": tuple(dep.install_hints),
            "description": dep.description,
        }
        for name, dep in OPTIONAL_DEPENDENCIES.items()
    }


__all__ = [
    "OptionalDependency",
    "OPTIONAL_DEPENDENCIES",
    "require",
    "is_available",
    "list_optional_dependencies",
]
