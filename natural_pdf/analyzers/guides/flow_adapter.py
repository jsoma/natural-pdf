"""Compatibility shim — canonical module is natural_pdf.guides.flow_adapter.

The module object below replaces this one in ``sys.modules`` so that
``natural_pdf.analyzers.guides.flow_adapter`` and ``natural_pdf.guides.flow_adapter`` are the
same module (monkeypatching either path affects both).
"""

import sys as _sys

from natural_pdf.guides import flow_adapter as _canonical

_sys.modules[__name__] = _canonical
