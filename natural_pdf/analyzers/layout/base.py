"""Compatibility shim — canonical module is natural_pdf.layout.base.

The module object below replaces this one in ``sys.modules`` so that
``natural_pdf.analyzers.layout.base`` and ``natural_pdf.layout.base`` are the
same module (monkeypatching either path affects both).
"""

import sys as _sys

from natural_pdf.layout import base as _canonical

_sys.modules[__name__] = _canonical
