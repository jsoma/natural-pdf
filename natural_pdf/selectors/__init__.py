"""Selector utilities and extension hooks."""

# Ensure built-in clauses are registered
from natural_pdf.selectors import _clauses as _builtin_clauses  # noqa: F401
from natural_pdf.selectors.parser import (
    build_text_contains_selector,
    parse_selector,
    selector_to_filter_func,
)
from natural_pdf.selectors.registry import (
    ClauseEvalContext,
    get_attribute_handler,
    get_pseudo_handler,
    register_attribute,
    register_pseudo,
    unregister_attribute,
    unregister_pseudo,
)

__all__ = [
    "ClauseEvalContext",
    "build_text_contains_selector",
    "get_attribute_handler",
    "get_pseudo_handler",
    "parse_selector",
    "register_attribute",
    "register_pseudo",
    "selector_to_filter_func",
    "unregister_attribute",
    "unregister_pseudo",
]
