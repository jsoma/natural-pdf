"""Combine simple contains and aggregate find() shortcuts."""

from __future__ import annotations

from contextlib import contextmanager

from experiments.performance.patches.selector_shortcuts._common import find_aggregate, find_contains

METADATA = {
    "track": "selector_shortcuts",
    "candidate": "find_combined_first",
    "cache_only": False,
    "hypothesis": "Combining first-match shortcuts should help real workflows that mix text anchors and aggregate title lookups.",
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
        common_kwargs = {
            "selector_obj": selector_obj,
            "apply_exclusions": bool(kwargs.get("apply_exclusions", True)),
            "overlap": (kwargs.get("overlap") or "full").lower(),
        }
        result = find_contains(
            host,
            regex=bool(kwargs.get("regex", False)),
            case=bool(kwargs.get("case", True)),
            **common_kwargs,
        )
        if result is not None:
            return result
        result = find_aggregate(host, **common_kwargs)
        if result is not None:
            return result
        return original_find(self, host, **kwargs)

    SelectorService.find = patched_find
    try:
        yield
    finally:
        SelectorService.find = original_find
