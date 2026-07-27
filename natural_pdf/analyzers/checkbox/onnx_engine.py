"""Compatibility shim — canonical module is natural_pdf.checkbox.onnx_engine.

The module object below replaces this one in ``sys.modules`` so that
``natural_pdf.analyzers.checkbox.onnx_engine`` and ``natural_pdf.checkbox.onnx_engine`` are the
same module (monkeypatching either path affects both).
"""

import sys as _sys

from natural_pdf.checkbox import onnx_engine as _canonical

_sys.modules[__name__] = _canonical
