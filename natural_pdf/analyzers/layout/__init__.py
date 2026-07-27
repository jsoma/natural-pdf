"""Compatibility shim — the canonical package is :mod:`natural_pdf.layout`.

Kept permanently so existing ``natural_pdf.analyzers.layout`` imports continue
to work. Submodules (``layout_manager``, ``paddle``, ...) are thin stubs that
alias the canonical modules in ``sys.modules``.
"""

from natural_pdf.layout import LayoutDetector, register_layout_engine, register_layout_engines

__all__ = ["LayoutDetector", "register_layout_engine", "register_layout_engines"]
