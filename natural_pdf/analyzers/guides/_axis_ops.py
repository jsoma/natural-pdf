"""Compatibility shim — canonical module is natural_pdf.guides._axis_ops.

The module object below replaces this one in ``sys.modules`` so that
``natural_pdf.analyzers.guides._axis_ops`` and ``natural_pdf.guides._axis_ops`` are the
same module (monkeypatching either path affects both).
"""

import sys as _sys

from natural_pdf.guides import _axis_ops as _canonical

_sys.modules[__name__] = _canonical
