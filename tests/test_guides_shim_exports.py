"""The natural_pdf.analyzers.guides shim must keep every pre-move export importable.

The canonical package moved to natural_pdf.guides; the old path is a permanent
compatibility shim. Its export surface must be a superset of what the pre-move
package (origin/main) exported.
"""

import importlib

# __all__ of natural_pdf/analyzers/guides/__init__.py before the move
# (git show origin/main:natural_pdf/analyzers/guides/__init__.py).
PRE_MOVE_EXPORTS = ["Guides", "GuidesList", "GuidesOcrResult"]


def test_pre_move_exports_still_importable():
    module = importlib.import_module("natural_pdf.analyzers.guides")
    for name in PRE_MOVE_EXPORTS:
        assert hasattr(module, name), f"shim lost pre-move export {name!r}"
        assert name in module.__all__, f"{name!r} missing from shim __all__"


def test_guides_ocr_result_aliases_canonical_class():
    # GuidesOcrResult was renamed to GuideOCRResult in the OCR-contract
    # refactor; the shim keeps the old name as an alias.
    from natural_pdf.analyzers.guides import GuidesOcrResult
    from natural_pdf.guides import GuideOCRResult

    assert GuidesOcrResult is GuideOCRResult


def test_guides_ocr_result_importable_from_every_historical_path():
    """GuidesOcrResult was defined in analyzers/guides/base.py pre-move. The
    shim aliases that module to natural_pdf.guides.base, so the alias must
    exist on the canonical modules — every historical import path (and its
    canonical counterpart) must keep working."""
    # Historical: from natural_pdf.analyzers.guides import GuidesOcrResult
    from natural_pdf.analyzers.guides import GuidesOcrResult as pkg_alias

    # Historical: from natural_pdf.analyzers.guides.base import GuidesOcrResult
    # (the shim makes this the canonical natural_pdf.guides.base module)
    from natural_pdf.analyzers.guides.base import GuidesOcrResult as shim_base_alias

    # Shimmed ocr module (same module object as natural_pdf.guides.ocr)
    from natural_pdf.analyzers.guides.ocr import GuidesOcrResult as shim_ocr_alias
    from natural_pdf.guides import GuideOCRResult

    # Canonical modules directly
    from natural_pdf.guides.base import GuidesOcrResult as base_alias
    from natural_pdf.guides.ocr import GuidesOcrResult as ocr_alias

    assert pkg_alias is GuideOCRResult
    assert shim_base_alias is GuideOCRResult
    assert base_alias is GuideOCRResult
    assert ocr_alias is GuideOCRResult
    assert shim_ocr_alias is GuideOCRResult
