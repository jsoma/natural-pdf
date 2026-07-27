"""Compatibility shim — canonical module is natural_pdf.guides._generation.

The module object below replaces this one in ``sys.modules`` so that
``natural_pdf.analyzers.guides._generation`` and ``natural_pdf.guides._generation`` are the
same module (monkeypatching either path affects both).
"""

import sys as _sys

from natural_pdf.guides import _generation as _canonical

_sys.modules[__name__] = _canonical
