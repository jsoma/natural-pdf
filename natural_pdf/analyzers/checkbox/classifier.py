"""Compatibility shim — canonical module is natural_pdf.checkbox.classifier.

The module object below replaces this one in ``sys.modules`` so that
``natural_pdf.analyzers.checkbox.classifier`` and ``natural_pdf.checkbox.classifier`` are the
same module (monkeypatching either path affects both).
"""

import sys as _sys

from natural_pdf.checkbox import classifier as _canonical

_sys.modules[__name__] = _canonical
