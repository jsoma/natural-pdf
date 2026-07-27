"""Compatibility shim — canonical module is natural_pdf.checkbox.vector.

The module object below replaces this one in ``sys.modules`` so that
``natural_pdf.analyzers.checkbox.vector`` and ``natural_pdf.checkbox.vector`` are the
same module (monkeypatching either path affects both).
"""

import sys as _sys

from natural_pdf.checkbox import vector as _canonical

_sys.modules[__name__] = _canonical
