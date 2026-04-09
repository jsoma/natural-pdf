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


def _stable_unique_elements(elements: Sequence[Any]) -> List[Any]:
    unique: List[Any] = []
    seen_ids: set[int] = set()
    for element in elements:
        marker = id(element)
        if marker in seen_ids:
            continue
        seen_ids.add(marker)
        unique.append(element)
    return unique


def _resolve_selector_pool(host: Any, selector_type: Optional[str]) -> List[Any]:
    pool_resolver = getattr(host, "_get_element_pool", None)
    if not callable(pool_resolver):
        raise TypeError(f"Host type {type(host)!r} does not expose selector pools.")
    return list(pool_resolver((selector_type or "any").lower()))


def _jaro_winkler_similarity(s1: str, s2: str, prefix_weight: float = 0.1) -> float:
    if s1 == s2:
        return 1.0

    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    match_distance = max(len1, len2) // 2 - 1
    if match_distance < 0:
        match_distance = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j]:
                continue
            if s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    transpositions = 0
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    transpositions /= 2

    jaro = ((matches / len1) + (matches / len2) + (matches - transpositions) / matches) / 3.0

    prefix = 0
    for c1, c2 in zip(s1, s2):
        if c1 != c2:
            break
        prefix += 1
        if prefix == 4:
            break

    jaro_winkler = jaro + prefix * prefix_weight * (1.0 - jaro)
    return max(0.0, min(1.0, jaro_winkler))


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
        instance_override = getattr(getattr(host, "__dict__", {}), "get", lambda *_: None)(
            "_apply_selector"
        )
        if callable(instance_override):
            return instance_override(selector_obj, **selector_kwargs)
        return execute_parsed_selector(
            host,
            selector_obj,
            selector_kwargs=selector_kwargs,
        )


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


def _sort_selector_matches(
    selector_obj: Dict[str, Any],
    elements: List[Any],
    selector_kwargs: Dict[str, Any],
    *,
    selector_type: Optional[str] = None,
    logger=None,
) -> List[Any]:
    selector_type = (selector_type or selector_obj.get("type", "any")).lower()
    has_contains = any(
        pseudo.get("name") in ("contains", "ocr", "closest")
        for pseudo in selector_obj.get("pseudo_classes", [])
    )
    if selector_type in ("rect", "region", "form_cell") and has_contains:
        return sorted(elements, key=lambda el: el.width * el.height if hasattr(el, "width") else 0)

    if selector_kwargs.get("reading_order", True):
        if all(hasattr(el, "top") and hasattr(el, "x0") for el in elements):
            return sorted(elements, key=lambda el: (el.top, el.x0))
        if elements and logger is not None:
            logger.warning(
                "Cannot sort elements in reading order: Missing required attributes (top, x0)."
            )
    return list(elements)


def _apply_special_match_pseudos(
    selector_obj: Dict[str, Any],
    elements: List[Any],
    selector_kwargs: Dict[str, Any],
) -> List[Any]:
    matches = list(elements)

    for pseudo in selector_obj.get("pseudo_classes", []):
        name = pseudo.get("name")
        if name != "closest" or pseudo.get("args") is None:
            continue

        search_text = str(pseudo["args"]).strip()
        threshold = 0.0
        if not search_text:
            return []

        if "@" in search_text and search_text.count("@") == 1:
            text_part, threshold_part = search_text.rsplit("@", 1)
            try:
                threshold = float(threshold_part)
                search_text = text_part.strip()
            except (ValueError, TypeError):
                pass

        ignore_case = not selector_kwargs.get("case", True)
        scored_elements = []
        for el in matches:
            if not getattr(el, "text", None):
                continue
            el_text = el.text.strip()
            search_term = search_text
            if ignore_case:
                el_text = el_text.lower()
                search_term = search_term.lower()

            ratio = _jaro_winkler_similarity(search_term, el_text)
            contains_match = search_term in el_text
            if ratio >= threshold:
                scored_elements.append((contains_match, ratio, el))

        scored_elements.sort(key=lambda x: (x[0], x[1]), reverse=True)
        matches = [entry[2] for entry in scored_elements]
        break

    for pseudo in selector_obj.get("pseudo_classes", []):
        name = pseudo.get("name")
        if name != "ocr" or pseudo.get("args") is None:
            continue

        from natural_pdf.selectors.ocr_match import DEFAULT_THRESHOLD, ocr_substring_score

        search_text = str(pseudo["args"]).strip()
        threshold = DEFAULT_THRESHOLD
        if not search_text:
            return []

        if "@" in search_text and search_text.count("@") == 1:
            text_part, threshold_part = search_text.rsplit("@", 1)
            try:
                threshold = float(threshold_part)
                search_text = text_part.strip()
            except (ValueError, TypeError):
                pass

        scored_elements = []
        for el in matches:
            if not getattr(el, "text", None):
                continue
            el_text = el.text.strip()
            score = ocr_substring_score(search_text, el_text)
            if score >= threshold:
                scored_elements.append((score, el))

        scored_elements.sort(key=lambda x: x[0], reverse=True)
        matches = [entry[1] for entry in scored_elements]
        break

    return matches


def execute_selector_branch(
    host: Any,
    selector_obj: Dict[str, Any],
    elements: Sequence[Any],
    *,
    selector_kwargs: Optional[Dict[str, Any]] = None,
    selector_type: Optional[str] = None,
    logger=None,
) -> List[Any]:
    from natural_pdf.selectors.parser import _calculate_aggregates, build_execution_plan

    branch_kwargs = dict(selector_kwargs or {})
    branch_kwargs.setdefault("selector_context", host)

    branch_type = (selector_type or selector_obj.get("type", "any")).lower()
    element_list = list(elements)

    has_aggregates = any(
        isinstance(attr.get("value"), dict) and attr["value"].get("type") == "aggregate"
        for attr in selector_obj.get("attributes", [])
    )
    aggregates: Dict[str, Any] = {}
    if has_aggregates:
        aggregates = _calculate_aggregates(element_list, selector_obj)

    filter_func, post_pseudos, relational_pseudos = build_execution_plan(
        selector_obj, aggregates=aggregates, **branch_kwargs
    )
    matching_elements = [element for element in element_list if filter_func(element)]
    matching_elements = _apply_relational_post_pseudos(
        host,
        {"relational_pseudos": relational_pseudos},
        matching_elements,
        branch_kwargs,
    )
    matching_elements = _sort_selector_matches(
        selector_obj,
        matching_elements,
        branch_kwargs,
        selector_type=branch_type,
        logger=logger,
    )
    matching_elements = _apply_special_match_pseudos(
        selector_obj,
        matching_elements,
        branch_kwargs,
    )
    return _apply_relational_post_pseudos(
        host,
        {"post_pseudos": post_pseudos},
        matching_elements,
        branch_kwargs,
    )


def execute_parsed_selector(
    host: Any,
    selector_obj: Dict[str, Any],
    *,
    selector_kwargs: Optional[Dict[str, Any]] = None,
    logger=None,
    context: Any = None,
) -> ElementCollection:
    """Execute a parsed selector against a host using branch-local execution."""
    branch_kwargs = dict(selector_kwargs or {})
    branch_kwargs.setdefault("selector_context", host)

    if selector_obj.get("type") == "or":
        matching_elements: List[Any] = []
        for sub_selector in selector_obj.get("selectors", []):
            sub_type = sub_selector.get("type", "any").lower()
            pool = _resolve_selector_pool(host, sub_type)
            matching_elements.extend(
                execute_selector_branch(
                    host,
                    sub_selector,
                    pool,
                    selector_kwargs=branch_kwargs,
                    selector_type=sub_type,
                    logger=logger,
                )
            )
        matching_elements = _stable_unique_elements(matching_elements)
    else:
        element_type = selector_obj.get("type", "any").lower()
        elements_to_search = _resolve_selector_pool(host, element_type)
        matching_elements = execute_selector_branch(
            host,
            selector_obj,
            elements_to_search,
            selector_kwargs=branch_kwargs,
            selector_type=element_type,
            logger=logger,
        )

    return ElementCollection(
        matching_elements,
        context=context if context is not None else getattr(host, "_context", None),
    )
