"""Compatibility shim — canonical module is natural_pdf.guides._grid_types.

The module object below replaces this one in ``sys.modules`` so that
``natural_pdf.analyzers.guides._grid_types`` and ``natural_pdf.guides._grid_types`` are the
same module (monkeypatching either path affects both).
"""

import sys as _sys

from natural_pdf.guides import _grid_types as _canonical

_sys.modules[__name__] = _canonical
