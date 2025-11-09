from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    cast,
)

from natural_pdf.elements.base import extract_bbox

if TYPE_CHECKING:
    from natural_pdf.elements.element_collection import ElementCollection
    from natural_pdf.elements.region import Region


logger = logging.getLogger(__name__)

ExclusionEntry = Tuple[Any, Optional[str], str]
ExclusionSpec = Union[ExclusionEntry, Tuple[Any, Optional[str]]]


class _ExclusionSelectorHost(Protocol):
    def find_all(
        self,
        selector: Optional[str] = None,
        *,
        text: Optional[Union[str, Sequence[str]]] = None,
        apply_exclusions: bool = True,
        regex: bool = False,
        case: bool = True,
        text_tolerance: Optional[dict] = None,
        auto_text_tolerance: Optional[Union[bool, dict]] = None,
        reading_order: bool = True,
    ) -> "ElementCollection": ...


class ExclusionMixin:
    _exclusions: List[ExclusionSpec]

    def _exclusion_element_manager(self):
        raise NotImplementedError

    def _element_to_region(self, element: Any, label: Optional[str] = None) -> Optional[Region]:
        raise NotImplementedError

    def _invalidate_exclusion_cache(self) -> None:
        pass

    def add_exclusion(
        self,
        exclusion: Any,
        label: Optional[str] = None,
        method: str = "region",
    ) -> Any:
        from natural_pdf.elements.element_collection import ElementCollection
        from natural_pdf.elements.region import Region

        if method not in {"region", "element"}:
            raise ValueError("Exclusion method must be 'region' or 'element'.")

        if isinstance(exclusion, str):
            selector_host = cast(_ExclusionSelectorHost, self)
            matches = selector_host.find_all(exclusion, apply_exclusions=False)
            self._store_exclusion_matches(matches, label, method)
            return self

        if isinstance(exclusion, ElementCollection):
            self._store_exclusion_matches(exclusion, label, method)
            return self

        if isinstance(exclusion, Region):
            exclusion.label = label
            self._append_exclusion((exclusion, label, method))
            return self

        if callable(exclusion):
            self._append_exclusion((exclusion, label, method))
            return self

        if isinstance(exclusion, (list, tuple)):
            for item in exclusion:
                self.add_exclusion(item, label=label, method=method)
            return self

        if method == "element":
            if extract_bbox(exclusion) is None:
                raise TypeError(
                    "Exclusion items must expose a bbox when method='element'. "
                    f"Received: {type(exclusion)!r}"
                )
            self._append_exclusion((exclusion, label, method))
            return self

        region = self._element_to_region(exclusion, label)
        if region is None:
            raise TypeError(
                f"Invalid exclusion type: {type(exclusion)}. Must be callable, Region, collection, or expose bbox."
            )
        self._append_exclusion((region, label, method))
        return self

    def _store_exclusion_matches(
        self,
        matches: Iterable[Any],
        label: Optional[str],
        method: str,
    ) -> None:
        from natural_pdf.elements.element_collection import ElementCollection

        iterable: Iterable[Any]
        if isinstance(matches, ElementCollection):
            iterable = matches.elements
        else:
            iterable = matches

        for match in iterable:
            if method == "element":
                if extract_bbox(match) is None:
                    raise TypeError(
                        "Exclusion items must expose a bbox when method='element'. "
                        f"Received: {type(match)!r}"
                    )
                self._append_exclusion((match, label, method))
            else:
                region = self._element_to_region(match, label)
                if region is None:
                    raise TypeError(
                        f"Invalid exclusion type: {type(match)}. Must be callable, Region, collection, or expose bbox."
                    )
                self._append_exclusion((region, label, method))

    def _append_exclusion(self, data: ExclusionEntry) -> None:
        exclusions = cast(List[ExclusionSpec], getattr(self, "_exclusions", []))
        exclusions.append(data)
        self._exclusions = exclusions
        self._invalidate_exclusion_cache()

    def _evaluate_exclusion_entries(
        self, entries: Sequence[ExclusionSpec], include_callable: bool, debug: bool
    ) -> List[Region]:
        from natural_pdf.elements.element_collection import ElementCollection
        from natural_pdf.elements.region import Region

        regions: List[Region] = []
        for idx, exclusion_data in enumerate(entries):
            if len(exclusion_data) == 2:
                exclusion_item, label = exclusion_data
                method = "region"
            else:
                exclusion_item, label, method = exclusion_data

            exclusion_label = label or f"exclusion {idx}"

            if callable(exclusion_item) and include_callable:
                ctx = getattr(self, "without_exclusions", nullcontext)
                try:
                    with ctx():
                        result = exclusion_item(self)
                except Exception as exc:
                    logger.error("Exclusion callable '%s' failed: %s", exclusion_label, exc)
                    continue

                if isinstance(result, Region):
                    result.label = label
                    regions.append(result)
                elif isinstance(result, ElementCollection):
                    for elem in result.elements:
                        region = self._element_to_region(elem, label)
                        if region is not None:
                            regions.append(region)
                continue

            if isinstance(exclusion_item, Region):
                regions.append(exclusion_item)
                continue

            if isinstance(exclusion_item, ElementCollection):
                for elem in exclusion_item.elements:
                    region = self._element_to_region(elem, label)
                    if region is not None:
                        regions.append(region)
                continue

            if isinstance(exclusion_item, (list, tuple)):
                for elem in exclusion_item:
                    region = self._element_to_region(elem, label)
                    if region is not None:
                        regions.append(region)
                continue

            region = self._element_to_region(exclusion_item, label)
            if region is not None:
                regions.append(region)

        return regions
