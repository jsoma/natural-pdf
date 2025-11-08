"""Built-in selector clause registrations."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict

from natural_pdf.selectors.registry import ClauseEvalContext, register_pseudo

logger = logging.getLogger(__name__)


def _element_text(element: Any) -> str:
    text = getattr(element, "text", "")
    if text is None:
        return ""
    return str(text)


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
            logger.warning(
                "Invalid regex '%s' in :contains selector: %s. Falling back to literal search.",
                search_term,
                exc,
            )
            pattern = None

        def regex_filter(element: Any) -> bool:
            if pattern is None:
                text = _element_text(element)
                haystack = text.lower() if ignore_case else text
                needle = search_term.lower() if ignore_case else search_term
                return needle in haystack
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
        logger.warning("Invalid regex '%s' in :regex selector: %s", pattern, exc)

        def always_false(_element: Any) -> bool:
            return False

        return {"name": f"pseudo-class :regex({pattern!r})", "func": always_false}

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
