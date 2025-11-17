from __future__ import annotations

import logging
from collections.abc import Iterable as IterableABC
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Sequence, Tuple, Union, cast

if TYPE_CHECKING:  # pragma: no cover
    from natural_pdf.elements.region import Region

from natural_pdf.core.exclusion_mixin import ExclusionEntry, ExclusionSpec
from natural_pdf.elements.base import extract_bbox
from natural_pdf.services.registry import register_delegate

logger = logging.getLogger(__name__)


class ExclusionService:
    """Service that owns exclusion bookkeeping and evaluation."""

    def __init__(self, context):
        self._context = context

    @register_delegate("exclusion", "add_exclusion")
    def add_exclusion(
        self,
        host: Any,
        exclusion: Any,
        label: Optional[str] = None,
        method: str = "region",
    ) -> Any:
        from natural_pdf.elements.element_collection import ElementCollection
        from natural_pdf.elements.region import Region

        if method not in {"region", "element"}:
            raise ValueError("Exclusion method must be 'region' or 'element'.")

        if isinstance(exclusion, str):
            finder = getattr(host, "find_all", None)
            if not callable(finder):
                raise TypeError("Host does not support selector-based exclusions.")
            matches = finder(exclusion, apply_exclusions=False)
            self._store_matches(host, matches, label, method)
            return host

        if isinstance(exclusion, ElementCollection):
            self._store_matches(host, exclusion, label, method)
            return host

        if isinstance(exclusion, Region):
            exclusion.label = label
            self._append_exclusion(host, (exclusion, label, method))
            return host

        if callable(exclusion):
            self._append_exclusion(host, (exclusion, label, method))
            return host

        if isinstance(exclusion, (list, tuple)):
            for item in exclusion:
                self.add_exclusion(host, item, label=label, method=method)
            return host

        if method == "element":
            if extract_bbox(exclusion) is None:
                raise TypeError(
                    "Exclusion items must expose a bbox when method='element'. "
                    f"Received: {type(exclusion)!r}"
                )
            self._append_exclusion(host, (exclusion, label, method))
            return host

        region = self._element_to_region(host, exclusion, label)
        if region is None:
            raise TypeError(
                f"Invalid exclusion type: {type(exclusion)}. Must be callable, Region, collection, selector, or expose bbox."
            )
        self._append_exclusion(host, (region, label, method))
        return host

    def evaluate_entries(
        self,
        host: Any,
        entries: Sequence[ExclusionSpec],
        include_callable: bool = True,
        debug: bool = False,
    ) -> List[Any]:
        from natural_pdf.elements.element_collection import ElementCollection
        from natural_pdf.elements.region import Region

        regions: List[Region] = []
        for idx, exclusion_data in enumerate(entries):
            if len(exclusion_data) == 2:
                exclusion_item, label = exclusion_data  # type: ignore[misc]
                method = "region"
            else:
                exclusion_item, label, method = exclusion_data  # type: ignore[misc]

            exclusion_label = label or f"exclusion {idx}"

            if callable(exclusion_item) and include_callable:
                ctx_factory = getattr(host, "without_exclusions", None)
                context_candidate = ctx_factory() if callable(ctx_factory) else None
                if isinstance(context_candidate, AbstractContextManager):
                    context_mgr: AbstractContextManager[Any] = context_candidate
                else:
                    context_mgr = nullcontext()
                try:
                    with context_mgr:
                        result = exclusion_item(host)
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.error("Exclusion callable '%s' failed: %s", exclusion_label, exc)
                    continue
                regions.extend(self._normalize_callable_result(host, result, label, debug=debug))
                continue

            if isinstance(exclusion_item, Region):
                regions.append(exclusion_item)
                continue

            if isinstance(exclusion_item, ElementCollection):
                regions.extend(
                    self._elements_to_regions(host, exclusion_item.elements, label, debug)
                )
                continue

            if isinstance(exclusion_item, (list, tuple)):
                regions.extend(self._elements_to_regions(host, exclusion_item, label, debug))
                continue

            if isinstance(exclusion_item, str):
                if method == "element":
                    continue  # handled later when filtering elements
                finder = getattr(host, "find_all", None)
                if not callable(finder):
                    continue
                matches = finder(exclusion_item, apply_exclusions=False)
                regions.extend(self._elements_to_regions(host, matches, label, debug))
                continue

            if (
                not callable(exclusion_item)
                and hasattr(exclusion_item, "bbox")
                and hasattr(exclusion_item, "expand")
            ):
                if method == "element":
                    continue
                try:
                    expanded = exclusion_item.expand()
                    if isinstance(expanded, Region):
                        expanded.label = label
                        regions.append(expanded)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("Failed to convert element exclusion to region: %s", exc)
                continue

            region = self._element_to_region(host, exclusion_item, label)
            if region is not None:
                regions.append(region)

        return regions

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _store_matches(self, host: Any, matches: Any, label: Optional[str], method: str):
        from natural_pdf.elements.element_collection import ElementCollection

        iterable: Iterable[Any]
        if isinstance(matches, ElementCollection):
            iterable = matches.elements
        elif isinstance(matches, IterableABC):
            iterable = matches
        else:
            raise TypeError(f"Exclusion matches must be iterable; received {type(matches)!r}")

        for match in iterable:
            if method == "element":
                if extract_bbox(match) is None:
                    raise TypeError(
                        "Exclusion items must expose a bbox when method='element'. "
                        f"Received: {type(match)!r}"
                    )
                self._append_exclusion(host, (match, label, method))
            else:
                region = self._element_to_region(host, match, label)
                if region is None:
                    raise TypeError(
                        f"Invalid exclusion type: {type(match)}. Must be callable, Region, collection, or expose bbox."
                    )
                self._append_exclusion(host, (region, label, method))

    def _append_exclusion(self, host: Any, data: ExclusionEntry) -> None:
        exclusions = cast(List[ExclusionSpec], getattr(host, "_exclusions", None))
        if exclusions is None:
            exclusions = []
        exclusions.append(data)
        host._exclusions = exclusions
        invalidator = getattr(host, "_invalidate_exclusion_cache", None)
        if callable(invalidator):
            invalidator()

    def _element_to_region(
        self, host: Any, element: Any, label: Optional[str]
    ) -> Optional["Region"]:
        from natural_pdf.elements.region import Region

        converter = getattr(host, "_element_to_region", None)
        if callable(converter):
            candidate = converter(element, label=label)
            if isinstance(candidate, Region):
                return candidate
        bbox = extract_bbox(element)
        if bbox is None:
            return None
        page = getattr(element, "page", None) or getattr(host, "page", None)
        if page is None and hasattr(host, "width"):
            page = host
        if page is None:
            return None
        region = Region(page, bbox)
        region.label = label
        return region

    def _elements_to_regions(
        self,
        host: Any,
        elements: Any,
        label: Optional[str],
        debug: bool,
    ) -> List["Region"]:
        from natural_pdf.elements.element_collection import ElementCollection

        if isinstance(elements, ElementCollection):
            iterable: Iterable[Any] = elements.elements
        elif isinstance(elements, IterableABC):
            iterable = elements
        else:
            raise TypeError(f"Element source must be iterable, got {type(elements)!r}")

        results: List["Region"] = []
        for elem in iterable:
            region = self._element_to_region(host, elem, label)
            if region is not None:
                results.append(region)
            elif debug:
                logger.debug("Failed to convert element %r into region", elem)
        return results

    def _normalize_callable_result(
        self,
        host: Any,
        result: Any,
        label: Optional[str],
        *,
        debug: bool,
    ) -> List["Region"]:
        from natural_pdf.elements.element_collection import ElementCollection
        from natural_pdf.elements.region import Region

        if isinstance(result, Region):
            result.label = label
            return [result]
        if isinstance(result, ElementCollection):
            return self._elements_to_regions(host, result.elements, label, debug)
        if isinstance(result, Iterable):
            return self._elements_to_regions(host, result, label, debug)
        if result is None:
            return []
        region = self._element_to_region(host, result, label)
        if region is not None:
            return [region]
        if debug:
            logger.debug("Exclusion callable returned unsupported value %r", result)
        return []
