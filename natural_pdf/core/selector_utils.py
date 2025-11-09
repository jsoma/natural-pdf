from __future__ import annotations

import contextlib
from typing import Any, Dict, List, Optional, Sequence, Union

from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.selectors.parser import build_text_contains_selector, parse_selector
from natural_pdf.selectors.registry import (
    ClauseEvalContext,
    get_post_handler,
    get_relational_handler,
)

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
    engine: Optional[str] = None,
) -> ElementCollection:
    """Execute a selector query using either the native engine or provider-backed engines."""
    from natural_pdf.selectors.selector_provider import (
        NATIVE_SELECTOR_ENGINE,
        resolve_selector_engine_name,
        run_selector_engine,
    )

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

    resolved_engine = resolve_selector_engine_name(host, engine)
    if resolved_engine and resolved_engine != NATIVE_SELECTOR_ENGINE:
        return run_selector_engine(
            host,
            selector,
            engine_name=resolved_engine,
            text_tolerance=text_tolerance,
            auto_text_tolerance=auto_text_tolerance,
            regex=regex,
            case=case,
            reading_order=reading_order,
            near_threshold=near_threshold,
        )

    return _run_native_selector(
        host,
        selector,
        text_tolerance=text_tolerance,
        auto_text_tolerance=auto_text_tolerance,
        regex=regex,
        case=case,
        reading_order=reading_order,
        near_threshold=near_threshold,
        selector_obj=selector_obj,
        selector_kwargs=selector_kwargs,
    )


def _run_native_selector(
    host: Any,
    selector: str,
    *,
    text_tolerance: Optional[Dict[str, Any]] = None,
    auto_text_tolerance: Optional[Union[bool, Dict[str, Any]]] = None,
    regex: bool = False,
    case: bool = True,
    reading_order: bool = True,
    near_threshold: Optional[float] = None,
    selector_obj: Optional[Dict[str, Any]] = None,
    selector_kwargs: Optional[Dict[str, Any]] = None,
) -> ElementCollection:
    selector_kwargs = selector_kwargs or {
        "regex": regex,
        "case": case,
        "reading_order": reading_order,
    }
    if near_threshold is not None:
        selector_kwargs["near_threshold"] = near_threshold
    selector_kwargs.setdefault("selector_context", host)

    selector_obj = selector_obj or parse_selector(selector)

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


def _apply_relational_post_pseudos(
    host: Any,
    selector_obj: Dict[str, Any],
    elements: List[Any],
    selector_kwargs: Dict[str, Any],
) -> List[Any]:
    """Apply registered relational and post-collection pseudo handlers."""
    relational = selector_obj.get("relational_pseudos")
    post = selector_obj.get("post_pseudos")
    if not relational and not post:
        return elements

    ctx_options = dict(selector_kwargs)
    ctx_options.pop("selector_context", None)
    context = ClauseEvalContext(selector_context=host, aggregates={}, options=ctx_options)

    result = list(elements)
    for pseudo in relational or []:
        handler = get_relational_handler(pseudo.get("name"))
        if handler:
            result = handler(result, pseudo, context)
    for pseudo in post or []:
        handler = get_post_handler(pseudo.get("name"))
        if handler:
            result = handler(result, pseudo, context)
    return result
