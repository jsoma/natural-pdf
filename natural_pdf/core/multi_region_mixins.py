from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Sequence, cast

from natural_pdf.core.exclusion_mixin import ExclusionEntry, ExclusionMixin, ExclusionSpec
from natural_pdf.core.ocr_mixin import OCRMixin
from natural_pdf.elements.base import DirectionalMixin

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from natural_pdf.elements.region import Region


class MultiRegionDirectionalMixin(DirectionalMixin):
    """Directional helpers for objects backed by multiple constituent regions."""

    def _directional_regions(self) -> Sequence["Region"]:
        regions = getattr(self, "constituent_regions", None)
        if regions is None:
            raise AttributeError(
                f"{self.__class__.__name__} must expose 'constituent_regions' to use "
                "MultiRegionDirectionalMixin."
            )
        return regions

    def _directional_flow(self):
        return getattr(self, "flow", None)

    def _spawn_from_regions(self, regions: List["Region"]):
        """Create a new region-like object from the supplied constituent regions."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Directional helpers
    # ------------------------------------------------------------------ #

    def above(
        self,
        height: float | None = None,
        width: str = "full",
        include_source: bool = False,
        until: str | None = None,
        include_endpoint: bool = True,
        offset: float | None = None,
        apply_exclusions: bool = True,
        multipage: Optional[bool] = None,
        within: Optional["Region"] = None,
        anchor: str = "start",
        **kwargs,
    ):
        regions = list(self._directional_regions())
        if not regions:
            return self._spawn_from_regions([])

        flow = self._directional_flow()
        arrangement = getattr(flow, "arrangement", "vertical")
        new_regions: List["Region"] = []

        if arrangement == "vertical":
            for idx, region in enumerate(regions):
                if idx == 0:
                    above_region = region.above(
                        height=height,
                        width="element",
                        include_source=include_source,
                        until=until,
                        include_endpoint=include_endpoint,
                        offset=offset,
                        apply_exclusions=apply_exclusions,
                        multipage=multipage,
                        within=within,
                        anchor=anchor,
                        **kwargs,
                    )
                    new_regions.append(cast("Region", above_region))
                elif include_source:
                    new_regions.append(region)
        else:
            for region in regions:
                above_region = region.above(
                    height=height,
                    width=width,
                    include_source=include_source,
                    until=until,
                    include_endpoint=include_endpoint,
                    offset=offset,
                    apply_exclusions=apply_exclusions,
                    multipage=multipage,
                    within=within,
                    anchor=anchor,
                    **kwargs,
                )
                new_regions.append(cast("Region", above_region))

        return self._spawn_from_regions(new_regions)

    def below(
        self,
        height: float | None = None,
        width: str = "full",
        include_source: bool = False,
        until: str | None = None,
        include_endpoint: bool = True,
        offset: float | None = None,
        apply_exclusions: bool = True,
        multipage: Optional[bool] = None,
        within: Optional["Region"] = None,
        anchor: str = "start",
        **kwargs,
    ):
        regions = list(self._directional_regions())
        if not regions:
            return self._spawn_from_regions([])

        flow = self._directional_flow()
        arrangement = getattr(flow, "arrangement", "vertical")
        new_regions: List["Region"] = []

        if arrangement == "vertical":
            for idx, region in enumerate(regions):
                if idx == len(regions) - 1:
                    below_region = region.below(
                        height=height,
                        width="element",
                        include_source=include_source,
                        until=until,
                        include_endpoint=include_endpoint,
                        offset=offset,
                        apply_exclusions=apply_exclusions,
                        multipage=multipage,
                        within=within,
                        anchor=anchor,
                        **kwargs,
                    )
                    new_regions.append(cast("Region", below_region))
                elif include_source:
                    new_regions.append(region)
        else:
            for region in regions:
                below_region = region.below(
                    height=height,
                    width=width,
                    include_source=include_source,
                    until=until,
                    include_endpoint=include_endpoint,
                    offset=offset,
                    apply_exclusions=apply_exclusions,
                    multipage=multipage,
                    within=within,
                    anchor=anchor,
                    **kwargs,
                )
                new_regions.append(cast("Region", below_region))

        return self._spawn_from_regions(new_regions)

    def left(
        self,
        width: float | None = None,
        height: str = "element",
        include_source: bool = False,
        until: str | None = None,
        include_endpoint: bool = True,
        offset: float | None = None,
        apply_exclusions: bool = True,
        multipage: Optional[bool] = None,
        within: Optional["Region"] = None,
        anchor: str = "start",
        **kwargs,
    ):
        regions = list(self._directional_regions())
        if not regions:
            return self._spawn_from_regions([])

        flow = self._directional_flow()
        arrangement = getattr(flow, "arrangement", "vertical")
        new_regions: List["Region"] = []

        if arrangement == "horizontal":
            for idx, region in enumerate(regions):
                if idx == 0:
                    left_region = region.left(
                        width=width,
                        height="element",
                        include_source=include_source,
                        until=until,
                        include_endpoint=include_endpoint,
                        offset=offset,
                        apply_exclusions=apply_exclusions,
                        multipage=multipage,
                        within=within,
                        anchor=anchor,
                        **kwargs,
                    )
                    new_regions.append(cast("Region", left_region))
                elif include_source:
                    new_regions.append(region)
        else:
            for region in regions:
                left_region = region.left(
                    width=width,
                    height=height,
                    include_source=include_source,
                    until=until,
                    include_endpoint=include_endpoint,
                    offset=offset,
                    apply_exclusions=apply_exclusions,
                    multipage=multipage,
                    within=within,
                    anchor=anchor,
                    **kwargs,
                )
                new_regions.append(cast("Region", left_region))

        return self._spawn_from_regions(new_regions)

    def right(
        self,
        width: float | None = None,
        height: str = "element",
        include_source: bool = False,
        until: str | None = None,
        include_endpoint: bool = True,
        offset: float | None = None,
        apply_exclusions: bool = True,
        multipage: Optional[bool] = None,
        within: Optional["Region"] = None,
        anchor: str = "start",
        **kwargs,
    ):
        regions = list(self._directional_regions())
        if not regions:
            return self._spawn_from_regions([])

        flow = self._directional_flow()
        arrangement = getattr(flow, "arrangement", "vertical")
        new_regions: List["Region"] = []

        if arrangement == "horizontal":
            for idx, region in enumerate(regions):
                if idx == len(regions) - 1:
                    right_region = region.right(
                        width=width,
                        height="element",
                        include_source=include_source,
                        until=until,
                        include_endpoint=include_endpoint,
                        offset=offset,
                        apply_exclusions=apply_exclusions,
                        multipage=multipage,
                        within=within,
                        anchor=anchor,
                        **kwargs,
                    )
                    new_regions.append(cast("Region", right_region))
                elif include_source:
                    new_regions.append(region)
        else:
            for region in regions:
                right_region = region.right(
                    width=width,
                    height=height,
                    include_source=include_source,
                    until=until,
                    include_endpoint=include_endpoint,
                    offset=offset,
                    apply_exclusions=apply_exclusions,
                    multipage=multipage,
                    within=within,
                    anchor=anchor,
                    **kwargs,
                )
                new_regions.append(cast("Region", right_region))

        return self._spawn_from_regions(new_regions)


class MultiRegionOCRMixin(OCRMixin):
    """Extend OCR operations across multiple constituent regions."""

    def _iter_ocr_regions(self) -> Iterable[Any]:  # pragma: no cover - hook
        raise NotImplementedError

    def apply_ocr(self, *args: Any, **kwargs: Any) -> Any:
        for region in self._iter_ocr_regions():
            region.apply_ocr(*args, **kwargs)
        return self

    def extract_ocr_elements(self, *args: Any, **kwargs: Any) -> List[Any]:
        elements: List[Any] = []
        for region in self._iter_ocr_regions():
            elements.extend(region.extract_ocr_elements(*args, **kwargs))
        return elements


class MultiRegionExclusionMixin(ExclusionMixin):
    """Aggregate exclusion regions across multiple constituent regions."""

    def _iter_exclusion_regions(self) -> Iterable[Any]:  # pragma: no cover - hook
        raise NotImplementedError

    def _get_exclusion_regions(
        self, include_callable: bool = True, debug: bool = False
    ) -> List["Region"]:
        local: List["Region"] = []
        entries = cast(List[ExclusionSpec], getattr(self, "_exclusions", []))
        if entries:
            local = self._evaluate_exclusion_entries(entries, include_callable, debug)

        aggregated: List["Region"] = []
        for region in self._iter_exclusion_regions():
            aggregated.extend(
                region._get_exclusion_regions(include_callable=include_callable, debug=debug)
            )

        return self._dedupe_regions((*local, *aggregated))

    @staticmethod
    def _dedupe_regions(regions: Iterable["Region"]) -> List["Region"]:
        deduped: List["Region"] = []
        seen: set[int] = set()
        for region in regions:
            marker = id(region)
            if marker in seen:
                continue
            seen.add(marker)
            deduped.append(region)
        return deduped
