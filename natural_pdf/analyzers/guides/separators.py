"""Compatibility shim — canonical module is natural_pdf.guides.separators.

The module object below replaces this one in ``sys.modules`` so that
``natural_pdf.analyzers.guides.separators`` and ``natural_pdf.guides.separators`` are the
same module (monkeypatching either path affects both).
"""

import sys as _sys

from natural_pdf.guides import separators as _canonical

_sys.modules[__name__] = _canonical
