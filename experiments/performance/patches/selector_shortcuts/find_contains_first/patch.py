"""Short-circuit Page/Region.find for simple text:contains selectors."""

from __future__ import annotations

from contextlib import contextmanager

from experiments.performance.patches.selector_shortcuts._common import find_contains

METADATA = {
    "track": "selector_shortcuts",
    "candidate": "find_contains_first",
    "cache_only": False,
    "hypothesis": "A common find('text:contains(...)') call can scan to the first reading-order match without building a full ElementCollection.",
}


@contextmanager
def install():
    from natural_pdf.selectors.parser import parse_selector
    from natural_pdf.services.selector_service import SelectorService

    original_find = SelectorService.find

    def patched_find(self, host, **kwargs):
        selector = kwargs.get("selector")
        if not selector or kwargs.get("engine") or kwargs.get("text_tolerance"):
            return original_find(self, host, **kwargs)
        selector_obj = parse_selector(selector)
        result = find_contains(
            host,
            selector_obj=selector_obj,
            regex=bool(kwargs.get("regex", False)),
            case=bool(kwargs.get("case", True)),
            apply_exclusions=bool(kwargs.get("apply_exclusions", True)),
            overlap=(kwargs.get("overlap") or "full").lower(),
        )
        if result is not None:
            return result
        return original_find(self, host, **kwargs)

    SelectorService.find = patched_find
    try:
        yield
    finally:
        SelectorService.find = original_find
