"""Shared replacement contract for OCR mutation APIs."""

from __future__ import annotations

from typing import Literal, cast

OCRReplaceMode = Literal["ocr", "all", "none"]

_VALID_REPLACE_MODES = frozenset({"ocr", "all", "none"})


def normalize_ocr_replace_mode(value: object) -> OCRReplaceMode:
    """Validate and normalize an OCR replacement mode.

    ``"ocr"`` replaces prior OCR artifacts in the target geometry, ``"all"``
    replaces native and OCR text there, and ``"none"`` appends without
    removing existing text. Boolean values are intentionally rejected: their
    historical meaning was both destructive and inconsistent across OCR hosts.
    """

    if isinstance(value, bool) or not isinstance(value, str):
        raise TypeError(
            "replace must be one of 'ocr', 'all', or 'none'; boolean replacement "
            "values are no longer supported"
        )

    normalized = value.strip().lower()
    if normalized not in _VALID_REPLACE_MODES:
        raise ValueError("replace must be one of 'ocr', 'all', or 'none'")
    return cast(OCRReplaceMode, normalized)
