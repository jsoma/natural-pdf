"""Shared helpers for region-first selector experiments."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable, Optional

from experiments.performance import vector_metrics as metrics
from experiments.performance.patches.selector_shortcuts._common import page_has_exclusions


def _selector_types(selector_obj: dict[str, Any]) -> set[str]:
    if selector_obj.get("type") == "or":
        return {str(sub.get("type", "any")).lower() for sub in selector_obj.get("selectors", [])}
    return {str(selector_obj.get("type", "any")).lower()}


def _stable_unique(elements: Iterable[Any]) -> list[Any]:
    unique = []
    seen = set()
    for element in elements:
        marker = id(element)
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(element)
    return unique


def _region_pool(region: Any, element_type: str, *, allowed_types: Optional[set[str]]) -> list[Any]:
    et = (element_type or "any").lower()
    normalized = "text" if et == "word" else et
    if allowed_types is not None and normalized not in allowed_types:
        metrics.count("shortcut.region_first.pool_fallback_type")
        return list(region.page._get_element_pool(element_type))

    page_pool = list(region.page._get_element_pool(element_type))
    if getattr(region, "_npdf_perf_apply_exclusions", True):
        page_pool = list(region.page._filter_elements_by_exclusions(page_pool))

    overlap = getattr(region, "_npdf_perf_overlap_mode", "full")
    filtered = region._filter_elements_by_overlap_mode(page_pool, overlap)
    metrics.count("shortcut.region_first.pool_fast_path")
    metrics.count("shortcut.region_first.pool_input", len(page_pool))
    metrics.count("shortcut.region_first.pool_output", len(filtered))
    return filtered


def install_region_first(
    *,
    candidate: str,
    allowed_types: Optional[set[str]] = None,
    require_no_exclusions: bool = False,
):
    @contextmanager
    def manager():
        from natural_pdf.elements.element_collection import ElementCollection
        from natural_pdf.elements.region import Region
        from natural_pdf.selectors.parser import parse_selector
        from natural_pdf.services import selector_service as selector_service_module
        from natural_pdf.services.selector_service import SelectorService

        original_find_all_region = SelectorService._find_all_region
        original_region_get_pool = getattr(Region, "_get_element_pool", None)

        def region_get_element_pool(self, element_type):
            return _region_pool(self, element_type, allowed_types=allowed_types)

        def patched_find_all_region(self, region, **kwargs):
            overlap_mode = (kwargs.get("overlap") or "full").lower()
            if overlap_mode not in {"full", "partial", "center"}:
                return original_find_all_region(self, region, **kwargs)

            normalized_kwargs = self._normalized_selector_kwargs(
                dict(kwargs),
                context="Region.find_all",
            )
            selector_obj = parse_selector(normalized_kwargs["selector"])
            types = _selector_types(selector_obj)
            normalized_types = {"text" if item == "word" else item for item in types}
            if allowed_types is not None and not normalized_types <= allowed_types:
                metrics.count(f"shortcut.region_first.{candidate}.fallback_selector_type")
                return original_find_all_region(self, region, **kwargs)
            if require_no_exclusions and normalized_kwargs.get("apply_exclusions", True):
                if page_has_exclusions(region.page):
                    metrics.count(f"shortcut.region_first.{candidate}.fallback_exclusions")
                    return original_find_all_region(self, region, **kwargs)

            prior_overlap = getattr(region, "_npdf_perf_overlap_mode", None)
            prior_apply = getattr(region, "_npdf_perf_apply_exclusions", None)
            region._npdf_perf_overlap_mode = overlap_mode
            region._npdf_perf_apply_exclusions = bool(
                normalized_kwargs.get("apply_exclusions", True)
            )
            try:
                result = selector_service_module.execute_selector_query(
                    region,
                    normalized_kwargs["selector"],
                    **self._page_query_options(region.page, normalized_kwargs),
                )
            finally:
                if prior_overlap is None:
                    region.__dict__.pop("_npdf_perf_overlap_mode", None)
                else:
                    region._npdf_perf_overlap_mode = prior_overlap
                if prior_apply is None:
                    region.__dict__.pop("_npdf_perf_apply_exclusions", None)
                else:
                    region._npdf_perf_apply_exclusions = prior_apply

            elements = _stable_unique(getattr(result, "elements", result or []))
            metrics.count(f"shortcut.region_first.{candidate}.fast_path")
            return ElementCollection(elements, context=getattr(region, "_context", self._context))

        Region._get_element_pool = region_get_element_pool
        SelectorService._find_all_region = patched_find_all_region
        try:
            yield
        finally:
            SelectorService._find_all_region = original_find_all_region
            if original_region_get_pool is None:
                delattr(Region, "_get_element_pool")
            else:
                Region._get_element_pool = original_region_get_pool

    return manager()
