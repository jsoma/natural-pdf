"""Deprecated shim for text extraction helpers; use natural_pdf.text.operations instead."""

from __future__ import annotations

import warnings

from natural_pdf.text.operations import *  # noqa: F401,F403

warnings.warn(
    "natural_pdf.utils.text_extraction is deprecated; import from natural_pdf.text.operations instead.",
    DeprecationWarning,
    stacklevel=2,
)
