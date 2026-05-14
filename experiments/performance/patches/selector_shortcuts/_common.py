"""Shared helpers for selector shortcut experiments."""

from __future__ import annotations

from typing import Any, Iterable, Optional

from experiments.performance import vector_metrics as metrics


def page_has_exclusions(page: Any) -> bool:
    if bool(getattr(page, "_exclusions", None)):
        return True
    parent = getattr(page, "_parent", None)
    return bool(parent is not None and getattr(parent, "_exclusions", None))


def text_of(element: Any) -> str:
    value = getattr(element, "text", None)
    if value is None:
        extractor = getattr(element, "extract_text", None)
        if callable(extractor):
            try:
                value = extractor()
            except Exception:
                value = ""
    return str(value or "")


def reading_key(element: Any) -> tuple[float, float]:
    return (float(getattr(element, "top", 0.0)), float(getattr(element, "x0", 0.0)))


def first_by_reading_order(elements: Iterable[Any]) -> Optional[Any]:
    best = None
    best_key = None
    for element in elements:
        key = reading_key(element)
        if best is None or key < best_key:  # type: ignore[operator]
            best = element
            best_key = key
    return best


def parse_simple_contains(selector_obj: dict[str, Any], *, regex: bool) -> Optional[str]:
    if regex:
        return None
    if (selector_obj.get("type") or "").lower() != "text":
        return None
    if selector_obj.get("attributes") or selector_obj.get("filters"):
        return None
    if selector_obj.get("relational_pseudos") or selector_obj.get("post_pseudos"):
        return None
    pseudos = selector_obj.get("pseudo_classes") or []
    if len(pseudos) != 1:
        return None
    pseudo = pseudos[0]
    if pseudo.get("name") != "contains":
        return None
    value = pseudo.get("args")
    return "" if value is None else str(value)


def parse_simple_aggregate(selector_obj: dict[str, Any]) -> Optional[tuple[str, str]]:
    if (selector_obj.get("type") or "").lower() != "text":
        return None
    if selector_obj.get("pseudo_classes") or selector_obj.get("filters"):
        return None
    if selector_obj.get("relational_pseudos") or selector_obj.get("post_pseudos"):
        return None
    attrs = selector_obj.get("attributes") or []
    if len(attrs) != 1:
        return None
    attr = attrs[0]
    if attr.get("op") not in {"=", "=="}:
        return None
    value = attr.get("value")
    if not isinstance(value, dict) or value.get("type") != "aggregate":
        return None
    func = value.get("func")
    if func not in {"max", "min"}:
        return None
    name = attr.get("name")
    if not isinstance(name, str) or not name:
        return None
    return name, func


def text_pool(host: Any) -> list[Any]:
    resolver = getattr(host, "_get_element_pool", None)
    if callable(resolver):
        return list(resolver("text"))
    page = getattr(host, "page", None)
    if page is not None:
        return list(getattr(page, "words", []))
    return []


def in_region(region: Any, element: Any, overlap: str = "full") -> bool:
    if overlap == "partial":
        return bool(region.intersects(element))
    if overlap == "center":
        return bool(region.is_element_center_inside(element))
    return bool(region.contains(element))


def maybe_filter_exclusions(page: Any, elements: list[Any], *, apply_exclusions: bool) -> list[Any]:
    if not apply_exclusions or not elements:
        return elements
    return list(page._filter_elements_by_exclusions(elements))


def find_contains(
    host: Any,
    *,
    selector_obj: dict[str, Any],
    regex: bool,
    case: bool,
    apply_exclusions: bool,
    overlap: str = "full",
) -> Any:
    needle = parse_simple_contains(selector_obj, regex=regex)
    if needle is None:
        metrics.count("shortcut.find_contains.fallback_parse")
        return None

    page = getattr(host, "page", host)
    if apply_exclusions and page_has_exclusions(page):
        metrics.count("shortcut.find_contains.fallback_exclusions")
        return None

    compare_needle = needle if case else needle.lower()
    candidates = []
    for element in text_pool(host):
        if host is not page and not in_region(host, element, overlap=overlap):
            continue
        haystack = text_of(element)
        if (compare_needle in haystack) if case else (compare_needle in haystack.lower()):
            candidates.append(element)

    metrics.count("shortcut.find_contains.fast_path")
    metrics.count("shortcut.find_contains.candidate_count", len(candidates))
    candidate = first_by_reading_order(candidates)
    if candidate is None:
        return None
    filtered = maybe_filter_exclusions(page, [candidate], apply_exclusions=apply_exclusions)
    return filtered[0] if filtered else None


def find_aggregate(
    host: Any,
    *,
    selector_obj: dict[str, Any],
    apply_exclusions: bool,
    overlap: str = "full",
) -> Any:
    aggregate = parse_simple_aggregate(selector_obj)
    if aggregate is None:
        metrics.count("shortcut.find_aggregate.fallback_parse")
        return None
    attr, func = aggregate

    page = getattr(host, "page", host)
    if apply_exclusions and page_has_exclusions(page):
        metrics.count("shortcut.find_aggregate.fallback_exclusions")
        return None

    best_value = None
    best_elements: list[Any] = []
    for element in text_pool(host):
        if host is not page and not in_region(host, element, overlap=overlap):
            continue
        raw_value = getattr(element, attr, None)
        if raw_value is None:
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if best_value is None:
            best_value = value
            best_elements = [element]
        elif (func == "max" and value > best_value) or (func == "min" and value < best_value):
            best_value = value
            best_elements = [element]
        elif value == best_value:
            best_elements.append(element)

    metrics.count("shortcut.find_aggregate.fast_path")
    metrics.count("shortcut.find_aggregate.candidate_count", len(best_elements))
    candidate = first_by_reading_order(best_elements)
    if candidate is None:
        return None
    filtered = maybe_filter_exclusions(page, [candidate], apply_exclusions=apply_exclusions)
    return filtered[0] if filtered else None
