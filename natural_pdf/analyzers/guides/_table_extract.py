"""Compatibility shim — canonical module is natural_pdf.guides._table_extract.

The module object below replaces this one in ``sys.modules`` so that
``natural_pdf.analyzers.guides._table_extract`` and ``natural_pdf.guides._table_extract`` are the
same module (monkeypatching either path affects both).
"""

import sys as _sys

from natural_pdf.guides import _table_extract as _canonical

_sys.modules[__name__] = _canonical
