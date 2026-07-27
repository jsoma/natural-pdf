"""Compatibility shim — canonical module is natural_pdf.checkbox.checkbox_manager.

The module object below replaces this one in ``sys.modules`` so that
``natural_pdf.analyzers.checkbox.checkbox_manager`` and ``natural_pdf.checkbox.checkbox_manager`` are the
same module (monkeypatching either path affects both).
"""

import sys as _sys

from natural_pdf.checkbox import checkbox_manager as _canonical

_sys.modules[__name__] = _canonical
