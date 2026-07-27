"""Compatibility shim — the canonical package is :mod:`natural_pdf.checkbox`.

Kept permanently so existing ``natural_pdf.analyzers.checkbox`` imports
continue to work. Submodules (``checkbox_manager``, ``vector``, ...) are thin
stubs that alias the canonical modules in ``sys.modules``.
"""

from natural_pdf.checkbox import (
    BaseCheckboxOptions,
    CheckboxOptions,
    DefaultCheckboxOptions,
    OnnxCheckboxOptions,
    VectorCheckboxOptions,
    VLMCheckboxOptions,
    engine_name_for_options,
    get_options_class_for_engine,
    register_checkbox_engines,
)

__all__ = [
    "BaseCheckboxOptions",
    "CheckboxOptions",
    "DefaultCheckboxOptions",
    "OnnxCheckboxOptions",
    "VectorCheckboxOptions",
    "VLMCheckboxOptions",
    "engine_name_for_options",
    "get_options_class_for_engine",
    "register_checkbox_engines",
]
