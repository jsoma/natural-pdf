"""Built-in selector clause registrations.

This module is intentionally declarative: every pseudo-class we support in the
core engine is registered through the clause-pack API so third parties can
follow the exact same pattern.  Each handler receives the parsed AST node plus a
``ClauseEvalContext`` (which provides the selector host/context/tolerances) and
either returns per-element filters (for simple predicates) or rewrites the
matched element list (for relational or collection-wide pseudos).  External
packages can import the ``register_*`` decorators below to plug in new clauses
without touching the parser.

When adding a new clause, prefer small helper factories so aliases remain
readable and logic stays testable.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from natural_pdf.selectors.registry import (
    ClauseEvalContext,
    register_post_pseudo,
    register_pseudo,
    register_relational_pseudo,
)

logger = logging.getLogger(__name__)


def _element_text(element: Any) -> str:
    text = getattr(element, "text", "")
    if text is None:
        return ""
    return str(text)


def _resolve_reference_elements(ctx: ClauseEvalContext, selector: Any) -> List[Any]:
    host = ctx.selector_context
    if host is None or not selector:
        return []

    selector_str = str(selector)
    find_kwargs = {}
    for key in (
        "regex",
        "case",
        "text_tolerance",
        "auto_text_tolerance",
        "reading_order",
        "near_threshold",
        "engine",
    ):
        if key in ctx.options:
            find_kwargs[key] = ctx.options[key]

    try:
        collection = host.find_all(selector=selector_str, **find_kwargs)
    except Exception:  # pragma: no cover - defensive
        return []

    if collection is None:
        return []
    return list(getattr(collection, "elements", collection))


@register_pseudo("contains", replace=True)
def _contains_clause(pseudo: Dict[str, Any], ctx: ClauseEvalContext):
    args = pseudo.get("args")
    if args is None:
        return None

    search_term = str(args)
    use_regex = ctx.options.get("regex", False)
    ignore_case = not ctx.options.get("case", True)

    if use_regex:
        try:
            pattern = re.compile(search_term, re.IGNORECASE if ignore_case else 0)
        except re.error as exc:  # pragma: no cover - defensive logging
            raise ValueError(f"Invalid regex '{search_term}' in :contains selector: {exc}") from exc

        def regex_filter(element: Any) -> bool:
            return bool(pattern.search(_element_text(element)))

        return {
            "name": f"pseudo-class :contains({search_term!r}, regex=True)",
            "func": regex_filter,
        }

    def literal_filter(element: Any) -> bool:
        text = _element_text(element)
        if ignore_case:
            return search_term.lower() in text.lower()
        return search_term in text

    return {"name": f"pseudo-class :contains({search_term!r})", "func": literal_filter}


@register_pseudo("regex", replace=True)
def _regex_clause(pseudo: Dict[str, Any], ctx: ClauseEvalContext):
    pattern = pseudo.get("args")
    if not isinstance(pattern, str):
        raise ValueError(":regex pseudo-class requires a string argument")

    ignore_case = not ctx.options.get("case", True)
    flags = re.IGNORECASE if ignore_case else 0
    try:
        compiled = re.compile(pattern, flags)
    except re.error as exc:  # pragma: no cover - defensive logging
        raise ValueError(f"Invalid regex '{pattern}' in :regex selector: {exc}") from exc

    def _filter(element: Any) -> bool:
        return bool(compiled.search(_element_text(element)))

    return {"name": f"pseudo-class :regex({pattern!r})", "func": _filter}


def _register_text_boundary(name: str, *, check: str):
    @register_pseudo(name, replace=True)
    def _handler(pseudo: Dict[str, Any], _ctx: ClauseEvalContext):
        args = pseudo.get("args")
        if args is None:
            return None
        needle = str(args)

        def _filter(element: Any) -> bool:
            text = _element_text(element)
            if check == "starts":
                return text.startswith(needle)
            return text.endswith(needle)

        return {"name": f"pseudo-class :{name}({needle!r})", "func": _filter}


def _register_aliases(base_name: str, aliases: list[str], *, check: str):
    _register_text_boundary(base_name, check=check)
    for alias in aliases:
        _register_text_boundary(alias, check=check)


_register_aliases("startswith", ["starts-with"], check="starts")
_register_aliases("endswith", ["ends-with"], check="ends")


def _register_boolean(names: list[str], attr: str, *, invert: bool = False):
    def factory(label: str):
        @register_pseudo(label, replace=True)
        def _handler(_pseudo: Dict[str, Any], _ctx: ClauseEvalContext):
            def _filter(element: Any) -> bool:
                value = bool(getattr(element, attr, False))
                return not value if invert else value

            return {"name": f"pseudo-class :{label}", "func": _filter}

        return _handler

    for name in names:
        factory(name)


_register_boolean(["bold"], "bold")
_register_boolean(["italic"], "italic")
_register_boolean(["horizontal"], "is_horizontal")
_register_boolean(["vertical"], "is_vertical")
_register_boolean(["checked"], "is_checked")
_register_boolean(["unchecked"], "is_checked", invert=True)
_register_boolean(["strike", "strikethrough", "strikeout"], "strike")
_register_boolean(["underline", "underlined"], "underline")
_register_boolean(["highlight", "highlighted"], "is_highlighted")


def _single_reference(ctx: ClauseEvalContext, pseudo: Dict[str, Any]) -> Any:
    refs = _resolve_reference_elements(ctx, pseudo.get("args"))
    return refs[0] if refs else None


@register_post_pseudo("first", replace=True)
def _post_first(elements: List[Any], _pseudo: Dict[str, Any], _ctx: ClauseEvalContext) -> List[Any]:
    return elements[:1] if elements else []


@register_post_pseudo("last", replace=True)
def _post_last(elements: List[Any], _pseudo: Dict[str, Any], _ctx: ClauseEvalContext) -> List[Any]:
    if not elements:
        return []
    return elements[-1:]


def _register_relational(name: str, predicate):
    @register_relational_pseudo(name, replace=True)
    def _handler(elements: List[Any], pseudo: Dict[str, Any], ctx: ClauseEvalContext) -> List[Any]:
        ref = _single_reference(ctx, pseudo)
        if ref is None:
            return []
        return [el for el in elements if predicate(el, ref, ctx)]

    return _handler


def _has_attrs(element: Any, *attrs: str) -> bool:
    return all(hasattr(element, attr) for attr in attrs)


def _rel_above(el: Any, ref: Any, _ctx: ClauseEvalContext) -> bool:
    return _has_attrs(el, "bottom") and _has_attrs(ref, "top") and el.bottom <= ref.top  # type: ignore[attr-defined]


def _rel_below(el: Any, ref: Any, _ctx: ClauseEvalContext) -> bool:
    return _has_attrs(el, "top") and _has_attrs(ref, "bottom") and el.top >= ref.bottom  # type: ignore[attr-defined]


def _rel_left_of(el: Any, ref: Any, _ctx: ClauseEvalContext) -> bool:
    return _has_attrs(el, "x1") and _has_attrs(ref, "x0") and el.x1 <= ref.x0  # type: ignore[attr-defined]


def _rel_right_of(el: Any, ref: Any, _ctx: ClauseEvalContext) -> bool:
    return _has_attrs(el, "x0") and _has_attrs(ref, "x1") and el.x0 >= ref.x1  # type: ignore[attr-defined]


def _rel_near(el: Any, ref: Any, ctx: ClauseEvalContext) -> bool:
    if not _has_attrs(el, "x0", "x1", "top", "bottom") or not _has_attrs(
        ref, "x0", "x1", "top", "bottom"
    ):
        return False
    el_center_x = (el.x0 + el.x1) / 2  # type: ignore[attr-defined]
    el_center_y = (el.top + el.bottom) / 2
    ref_center_x = (ref.x0 + ref.x1) / 2  # type: ignore[attr-defined]
    ref_center_y = (ref.top + ref.bottom) / 2
    distance = ((el_center_x - ref_center_x) ** 2 + (el_center_y - ref_center_y) ** 2) ** 0.5
    threshold = ctx.options.get("near_threshold")
    if threshold is None:
        threshold = 50
    return distance <= threshold


_register_relational("above", _rel_above)
_register_relational("below", _rel_below)
_register_relational("left-of", _rel_left_of)
_register_relational("right-of", _rel_right_of)
_register_relational("near", _rel_near)
