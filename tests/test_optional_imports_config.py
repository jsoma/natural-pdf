"""Tests covering optional dependency metadata and extras wiring."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest

try:  # Python 3.11+
    import tomllib as toml_loader
except ModuleNotFoundError:  # pragma: no cover - fallback for 3.10
    import tomli as toml_loader  # type: ignore[import]

from natural_pdf.utils import optional_imports as oi

REQUIRED_DEPENDENCIES = {
    "pikepdf",
    "deskew",
    "easyocr",
    "lancedb",
    "pyarrow",
    "sentence_transformers",
    "torch",
    "transformers",
    "torchvision",
    "huggingface_hub",
    "doclayout_yolo",
    "timm",
    "img2pdf",
    "openai",
}


def _load_optional_extras() -> Dict[str, list[str]]:
    pyproject_path = Path("pyproject.toml")
    data = toml_loader.loads(pyproject_path.read_text())
    return data["project"]["optional-dependencies"]


def test_optional_dependency_registry_is_complete():
    missing = REQUIRED_DEPENDENCIES - set(oi.OPTIONAL_DEPENDENCIES)
    assert not missing, f"Registry missing entries: {sorted(missing)}"


@pytest.mark.parametrize("dep_name", sorted(REQUIRED_DEPENDENCIES))
def test_optional_dependency_has_install_hints(dep_name: str):
    dep = oi.OPTIONAL_DEPENDENCIES[dep_name]
    hints = tuple(dep.install_hints)
    assert hints and all(hints), f"Missing install hints for {dep_name}"


def test_list_optional_dependencies_matches_registry():
    info = oi.list_optional_dependencies()
    assert set(info.keys()) == set(oi.OPTIONAL_DEPENDENCIES.keys())


def test_require_unknown_dependency_raises_key_error():
    with pytest.raises(KeyError):
        oi.require("nonexistent-dependency")


def test_search_extra_stays_lightweight():
    extras = _load_optional_extras()
    search_requirements = extras["search"]
    assert "sentence-transformers" in search_requirements
    assert not any(req.startswith("natural-pdf[") for req in search_requirements)
    assert {"lancedb", "pyarrow"}.issubset(set(search_requirements))
