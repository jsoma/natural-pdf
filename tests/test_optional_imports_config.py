"""Tests covering optional dependency metadata and extras wiring."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest

try:  # Python 3.11+
    import tomllib as toml_loader
except ModuleNotFoundError:  # pragma: no cover - fallback for 3.10
    import tomli as toml_loader  # type: ignore[import]

from natural_pdf.cli import EXTRA_GROUPS
from natural_pdf.utils import optional_imports as oi

REQUIRED_DEPENDENCIES = {
    "rapidocr",
    "pikepdf",
    "easyocr",
    "sentence_transformers",
    "torch",
    "transformers",
    "torchvision",
    "huggingface_hub",
    "doclayout_yolo",
    "timm",
    "img2pdf",
}


def _load_optional_extras() -> Dict[str, list[str]]:
    pyproject_path = Path("pyproject.toml")
    data = toml_loader.loads(pyproject_path.read_text())
    return data["project"]["optional-dependencies"]


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


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


def test_public_runtime_extras_contract():
    extras = _load_optional_extras()

    assert {"ai", "export", "paddle", "all"} <= set(extras)
    assert "rapidocr_onnxruntime" in extras["ai"]
    assert extras["all"] == ["natural-pdf[ai]", "natural-pdf[export]"]

    assert "natural-pdf[test]" not in extras["all"]
    assert "natural-pdf[quality]" not in extras["all"]
    assert "natural-pdf[dev]" not in extras["all"]
    assert "natural-pdf[paddle]" not in extras["all"]


def test_cli_public_groups_align_with_runtime_extras():
    extras = _load_optional_extras()

    assert {"all", "ai", "export", "paddle"} <= set(EXTRA_GROUPS)
    assert "rapidocr" in EXTRA_GROUPS["all"]
    assert "rapidocr" in EXTRA_GROUPS["ai"]
    assert "pikepdf" in EXTRA_GROUPS["export"]
    assert "paddlepaddle" in EXTRA_GROUPS["paddle"]
    assert {"ai", "export", "paddle", "all"} <= set(extras)


def test_default_ocr_install_hint_matches_public_contract():
    rapidocr = oi.OPTIONAL_DEPENDENCIES["rapidocr"]
    assert 'pip install "natural-pdf[all]"' in rapidocr.install_hints
    assert any("rapidocr_onnxruntime" in hint for hint in rapidocr.install_hints)


def test_docs_describe_all_as_recommended_core_complete_install():
    readme = _read_text("README.md")
    install_doc = _read_text("docs/installation/index.md")

    assert 'pip install "natural-pdf[all]"' in readme
    assert 'pip install "natural-pdf[all]"' in install_doc
    assert "recommended feature-complete install" in readme
    assert "recommended core-complete install" in install_doc
    assert "not install every optional backend" in readme
    assert "does not include every optional backend" in install_doc


def test_search_docs_match_sentence_transformers_runtime():
    readme = _read_text("README.md")
    install_doc = _read_text("docs/installation/index.md")

    assert "Haystack" not in readme
    assert "sentence-transformer embeddings" in readme
    assert "sentence-transformers" in readme or "sentence-transformers" in install_doc
    assert (
        'pip install "natural-pdf[all]"'
        in oi.OPTIONAL_DEPENDENCIES["sentence_transformers"].install_hints
    )
