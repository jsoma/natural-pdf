"""Compatibility shim — canonical module is natural_pdf.layout.pdfplumber_table_finder.

The module object below replaces this one in ``sys.modules`` so that
``natural_pdf.analyzers.layout.pdfplumber_table_finder`` and ``natural_pdf.layout.pdfplumber_table_finder`` are the
same module (monkeypatching either path affects both).
"""

import sys as _sys

from natural_pdf.layout import pdfplumber_table_finder as _canonical

_sys.modules[__name__] = _canonical
