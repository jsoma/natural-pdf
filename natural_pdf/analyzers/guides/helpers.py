"""Compatibility shim — canonical module is natural_pdf.guides.helpers.

The module object below replaces this one in ``sys.modules`` so that
``natural_pdf.analyzers.guides.helpers`` and ``natural_pdf.guides.helpers`` are the
same module (monkeypatching either path affects both).
"""

import sys as _sys

from natural_pdf.guides import helpers as _canonical

_sys.modules[__name__] = _canonical
