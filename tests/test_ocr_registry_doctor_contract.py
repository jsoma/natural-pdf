from __future__ import annotations

import re
import sys

from packaging.requirements import Requirement


def _load_pyproject():
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10
        import tomli as tomllib

    with open("pyproject.toml", "rb") as handle:
        return tomllib.load(handle)


def _dist_name(requirement: str) -> str:
    return Requirement(requirement).name.lower().replace("-", "_")


def _active_extra_deps(requirements: list[str]) -> set[str]:
    deps = set()
    for requirement in requirements:
        parsed = Requirement(requirement)
        if parsed.marker is not None and not parsed.marker.evaluate():
            continue
        deps.add(_dist_name(requirement))
    return deps


def test_natural_pdf_ocr_no_longer_exports_legacy_public_symbols():
    import natural_pdf.ocr as ocr

    for name in (
        "OCRFactory",
        "ENGINE_REGISTRY",
        "register_ocr_engines",
        "run_ocr_apply",
        "run_ocr_engine",
        "run_ocr_extract",
        "get_engine",
        "list_available_engines",
    ):
        assert not hasattr(ocr, name)


def test_optional_dependency_groups_match_pyproject_extras():
    from natural_pdf.utils.optional_imports import (
        list_dependency_groups,
        list_optional_dependencies,
    )

    pyproject = _load_pyproject()
    extras = pyproject["project"]["optional-dependencies"]
    dep_info = list_optional_dependencies()
    groups = list_dependency_groups()

    for group_name in ("ai", "export", "paddle", "quality"):
        expected = _active_extra_deps(extras[group_name])
        actual = {
            package_name.lower().replace("-", "_")
            for dep_name in groups[group_name]
            for package_name in dep_info[dep_name]["package_names"]
        }
        assert actual == expected

    ai = _active_extra_deps(extras["ai"])
    assert "easyocr" not in ai
    assert "rapidocr" in ai
    assert "doclayout_yolo" in ai
    assert "easyocr" not in groups["all"]
    assert set(groups["all"]) == set(groups["ai"]) | set(groups["export"]) | set(groups["quality"])
    assert extras["all"] == [
        "natural-pdf[ai]",
        "natural-pdf[export]",
        "natural-pdf[quality]",
    ]


def test_npdf_doctor_and_list_are_same_output(monkeypatch, capsys):
    from natural_pdf.cli import main

    monkeypatch.setattr(sys, "argv", ["npdf", "doctor"])
    main()
    doctor_output = capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["npdf", "list"])
    main()
    list_output = capsys.readouterr().out

    assert doctor_output == list_output
    assert "Natural PDF doctor" in doctor_output
    assert "easyocr" not in re.search(
        r"natural-pdf\[ai\].*",
        doctor_output,
    ).group(0)


def test_npdf_doctor_does_not_import_optional_ocr_packages(monkeypatch, capsys):
    from natural_pdf.cli import main

    for name in ("easyocr", "rapidocr", "paddleocr", "surya", "doctr"):
        sys.modules.pop(name, None)

    monkeypatch.setattr(sys, "argv", ["npdf", "doctor"])
    main()
    capsys.readouterr()

    for name in ("easyocr", "rapidocr", "paddleocr", "surya", "doctr"):
        assert name not in sys.modules


def test_extract_ocr_elements_no_longer_imports_legacy_runner():
    import natural_pdf.services.ocr_service as ocr_service

    assert not hasattr(ocr_service, "run_ocr_extract")
