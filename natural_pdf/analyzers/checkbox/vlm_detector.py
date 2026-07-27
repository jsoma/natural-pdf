"""Compatibility shim — canonical module is natural_pdf.checkbox.vlm_detector.

The module object below replaces this one in ``sys.modules`` so that
``natural_pdf.analyzers.checkbox.vlm_detector`` and ``natural_pdf.checkbox.vlm_detector`` are the
same module (monkeypatching either path affects both).
"""

import sys as _sys

from natural_pdf.checkbox import vlm_detector as _canonical

_sys.modules[__name__] = _canonical
