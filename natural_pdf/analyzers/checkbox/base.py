"""Compatibility shim — canonical module is natural_pdf.checkbox.base.

The module object below replaces this one in ``sys.modules`` so that
``natural_pdf.analyzers.checkbox.base`` and ``natural_pdf.checkbox.base`` are the
same module (monkeypatching either path affects both).
"""

import sys as _sys

from natural_pdf.checkbox import base as _canonical

_sys.modules[__name__] = _canonical
