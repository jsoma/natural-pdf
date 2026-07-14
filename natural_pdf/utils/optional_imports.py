"""Helpers for managing optional dependency imports consistently."""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from importlib import import_module, metadata, util
from typing import Any, Callable, Dict, Mapping, Optional, Sequence


@dataclass
class OptionalDependency:
    """Represents a lazily imported optional dependency."""

    module_name: str
    install_hints: Sequence[str]
    description: Optional[str] = None
    import_fn: Optional[Callable[[], Any]] = None
    package_names: Sequence[str] = ()
    applicable: Callable[[], bool] = lambda: True
    _module: Optional[Any] = field(default=None, init=False)
    _available: Optional[bool] = field(default=None, init=False)

    def is_applicable(self) -> bool:
        return bool(self.applicable())

    def is_available(self) -> bool:
        if not self.is_applicable():
            return False
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

    def versions(self) -> Dict[str, str]:
        versions: Dict[str, str] = {}
        for package_name in self.package_names or (self.module_name,):
            try:
                versions[package_name] = metadata.version(package_name)
            except metadata.PackageNotFoundError:
                continue
        return versions


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


OPTIONAL_DEPENDENCIES: Dict[str, OptionalDependency] = {
    # OCR
    "rapidocr": OptionalDependency(
        "rapidocr",
        ('pip install "natural-pdf[all]"', "pip install rapidocr"),
        "Default RapidOCR engine for OCR workflows.",
        package_names=("rapidocr",),
    ),
    "easyocr": OptionalDependency(
        "easyocr",
        ("pip install easyocr",),
        "Opt-in EasyOCR engine for OCR workflows.",
    ),
    "surya": OptionalDependency(
        "surya",
        ("pip install surya-ocr",),
        "Opt-in Surya OCR engine.",
        package_names=("surya-ocr",),
    ),
    "doctr": OptionalDependency(
        "doctr",
        ("pip install python-doctr",),
        "Opt-in Doctr OCR engine.",
        package_names=("python-doctr",),
    ),
    "paddleocr": OptionalDependency(
        "paddleocr",
        ('pip install "natural-pdf[paddle]"', "pip install paddleocr"),
        "PaddleOCR and PaddleOCR-VL engine stack.",
    ),
    "paddlepaddle": OptionalDependency(
        "paddle",
        ('pip install "natural-pdf[paddle]"', "pip install paddlepaddle"),
        "PaddlePaddle runtime for PaddleOCR.",
        package_names=("paddlepaddle",),
    ),
    "paddlex": OptionalDependency(
        "paddlex",
        ('pip install "natural-pdf[paddle]"', "pip install paddlex[ocr]"),
        "PaddleX OCR pipeline dependency.",
    ),
    "chardet": OptionalDependency(
        "chardet",
        ('pip install "natural-pdf[paddle]"', "pip install chardet"),
        "Character encoding detection dependency used by Paddle OCR tooling.",
    ),
    "numpy": OptionalDependency(
        "numpy",
        ('pip install "natural-pdf[paddle]"', "pip install 'numpy<2.0'"),
        "Numeric runtime dependency; the Paddle extra pins this below 2.0.",
    ),
    # Export
    "pikepdf": OptionalDependency(
        "pikepdf",
        ('pip install "natural-pdf[export]"',),
        "Required for creating searchable PDFs.",
    ),
    "img2pdf": OptionalDependency(
        "img2pdf",
        ('pip install "natural-pdf[export]"',),
        "Image to PDF conversion helper used by deskew/save routines.",
    ),
    "openpyxl": OptionalDependency(
        "openpyxl",
        ('pip install "natural-pdf[export]"',),
        "Spreadsheet writer used for Excel exports.",
    ),
    "jupytext": OptionalDependency(
        "jupytext",
        ('pip install "natural-pdf[export]"',),
        "Notebook/text conversion support for export workflows.",
    ),
    "nbformat": OptionalDependency(
        "nbformat",
        ('pip install "natural-pdf[export]"',),
        "Notebook format support for export workflows.",
    ),
    # Search / embeddings
    "sentence_transformers": OptionalDependency(
        "sentence_transformers",
        ('pip install "natural-pdf[all]"', "pip install sentence-transformers"),
        "Embedding models for semantic search.",
        package_names=("sentence-transformers",),
    ),
    # ML Core
    "torch": OptionalDependency(
        "torch",
        ('pip install "natural-pdf[all]"', "pip install torch"),
        "PyTorch runtime used by QA/classification/layout engines.",
    ),
    "transformers": OptionalDependency(
        "transformers",
        ('pip install "natural-pdf[all]"', "pip install transformers"),
        "Hugging Face transformers for QA/classification.",
    ),
    "torchvision": OptionalDependency(
        "torchvision",
        ('pip install "natural-pdf[all]"', "pip install torchvision"),
        "TorchVision models/utilities used by perception pipelines.",
    ),
    "huggingface_hub": OptionalDependency(
        "huggingface_hub",
        ("pip install natural-pdf", "pip install huggingface_hub"),
        "Model hub utilities required by AI engines.",
    ),
    "mlx_vlm": OptionalDependency(
        "mlx_vlm",
        (
            'pip install "natural-pdf[ai]"',
            "pip install mlx-vlm",
        ),
        "MLX local VLM runtime for Apple Silicon OCR and extraction.",
        package_names=("mlx-vlm",),
        applicable=_is_apple_silicon,
    ),
    # Layout
    "doclayout_yolo": OptionalDependency(
        "doclayout_yolo",
        ('pip install "natural-pdf[all]"', "pip install doclayout_yolo"),
        "YOLO-based layout detection models.",
    ),
    "timm": OptionalDependency(
        "timm",
        ('pip install "natural-pdf[all]"', "pip install timm"),
        "Required backbone models for layout detectors.",
    ),
}

OPTIONAL_DEPENDENCY_GROUPS: Dict[str, tuple[str, ...]] = {
    "ai": (
        "rapidocr",
        "torch",
        "torchvision",
        "huggingface_hub",
        "mlx_vlm",
        "transformers",
        "sentence_transformers",
        "timm",
        "doclayout_yolo",
    ),
    "export": ("pikepdf", "img2pdf", "openpyxl", "jupytext", "nbformat"),
    "paddle": ("chardet", "paddlepaddle", "paddleocr", "paddlex", "numpy"),
    "all": (
        "rapidocr",
        "torch",
        "torchvision",
        "huggingface_hub",
        "mlx_vlm",
        "transformers",
        "sentence_transformers",
        "timm",
        "doclayout_yolo",
        "pikepdf",
        "img2pdf",
        "openpyxl",
        "jupytext",
        "nbformat",
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
            "versions": dep.versions(),
            "install_hints": tuple(dep.install_hints),
            "description": dep.description,
            "package_names": tuple(dep.package_names or (dep.module_name,)),
            "module_name": dep.module_name,
            "applicable": dep.is_applicable(),
        }
        for name, dep in OPTIONAL_DEPENDENCIES.items()
    }


def list_dependency_groups() -> Mapping[str, tuple[str, ...]]:
    return {
        group: tuple(
            dep_name
            for dep_name in dependency_names
            if OPTIONAL_DEPENDENCIES[dep_name].is_applicable()
        )
        for group, dependency_names in OPTIONAL_DEPENDENCY_GROUPS.items()
    }


__all__ = [
    "OptionalDependency",
    "OPTIONAL_DEPENDENCIES",
    "OPTIONAL_DEPENDENCY_GROUPS",
    "require",
    "is_available",
    "list_optional_dependencies",
    "list_dependency_groups",
]
