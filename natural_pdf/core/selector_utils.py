from __future__ import annotations

import contextlib
from typing import Any, Dict, Optional, Sequence, Union

from natural_pdf.selectors.parser import build_text_contains_selector, parse_selector

TextInput = Union[str, Sequence[str]]


def normalize_selector_input(
    selector: Optional[str],
    text: Optional[TextInput],
    *,
    logger,
    context: str,
) -> str:
    """
    Normalize selector/text inputs into a selector string with consistent validation.
    """
    if selector is not None and text is not None:
        raise ValueError("Provide either 'selector' or 'text', not both.")
    if selector is None and text is None:
        raise ValueError("Provide either 'selector' or 'text'.")

    if text is not None:
        effective_selector = build_text_contains_selector(text)
        if logger:
            logger.debug(
                "Using text shortcut: %s(text=%r) -> %s('%s')",
                context,
                text,
                context,
                effective_selector,
            )
        return effective_selector

    return selector or ""


def execute_selector_query(
    host: Any,
    selector: str,
    *,
    text_tolerance: Optional[Dict[str, Any]] = None,
    auto_text_tolerance: Optional[Union[bool, Dict[str, Any]]] = None,
    regex: bool = False,
    case: bool = True,
    reading_order: bool = True,
    near_threshold: Optional[float] = None,
) -> "ElementCollection":
    """
    Execute a selector query against a host object that exposes `_apply_selector`
    and `_temporary_text_settings`.
    """
    if text_tolerance is not None and not isinstance(text_tolerance, dict):
        raise TypeError("text_tolerance must be a dict of tolerance overrides.")

    selector_obj = parse_selector(selector)  # type: ignore[arg-type]

    selector_kwargs: Dict[str, Any] = {
        "regex": regex,
        "case": case,
        "reading_order": reading_order,
    }
    if near_threshold is not None:
        selector_kwargs["near_threshold"] = near_threshold

    temporary_text_settings = getattr(host, "_temporary_text_settings", None)
    cm = (
        host._temporary_text_settings(  # type: ignore[attr-defined]
            text_tolerance=text_tolerance,
            auto_text_tolerance=auto_text_tolerance,
        )
        if callable(temporary_text_settings)
        else contextlib.nullcontext()
    )

    with cm:
        return host._apply_selector(selector_obj, **selector_kwargs)  # type: ignore[attr-defined]
