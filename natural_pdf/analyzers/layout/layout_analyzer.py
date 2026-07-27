"""Compatibility shim — canonical module is natural_pdf.layout.layout_analyzer.

The module object below replaces this one in ``sys.modules`` so that
``natural_pdf.analyzers.layout.layout_analyzer`` and ``natural_pdf.layout.layout_analyzer`` are the
same module (monkeypatching either path affects both).
"""

import sys as _sys

from natural_pdf.layout import layout_analyzer as _canonical

_sys.modules[__name__] = _canonical
