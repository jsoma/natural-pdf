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
