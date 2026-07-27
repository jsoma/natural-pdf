"""Compatibility shim — canonical module is natural_pdf.guides.text_detect.

The module object below replaces this one in ``sys.modules`` so that
``natural_pdf.analyzers.guides.text_detect`` and ``natural_pdf.guides.text_detect`` are the
same module (monkeypatching either path affects both).
"""

import sys as _sys

from natural_pdf.guides import text_detect as _canonical

_sys.modules[__name__] = _canonical
