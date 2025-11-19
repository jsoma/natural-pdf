"""Checkbox detection analyzers for natural-pdf."""

from .checkbox_options import CheckboxOptions, RTDETRCheckboxOptions
from .registry import (
    DEFAULT_ENGINE,
    ENGINE_REGISTRY,
    get_detector,
    list_available_engines,
    prepare_checkbox_options,
    register_checkbox_engine,
)

__all__ = [
    "CheckboxOptions",
    "RTDETRCheckboxOptions",
    "DEFAULT_ENGINE",
    "ENGINE_REGISTRY",
    "register_checkbox_engine",
    "list_available_engines",
    "get_detector",
    "prepare_checkbox_options",
]
