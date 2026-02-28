"""Checkbox detection analyzers for natural-pdf."""

# Trigger engine registration at import time
from .checkbox_manager import (
    engine_name_for_options,
    get_options_class_for_engine,
    register_checkbox_engines,
)
from .checkbox_options import (
    BaseCheckboxOptions,
    DefaultCheckboxOptions,
    OnnxCheckboxOptions,
    VectorCheckboxOptions,
    VLMCheckboxOptions,
)

# Backward-compat alias
CheckboxOptions = BaseCheckboxOptions

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
