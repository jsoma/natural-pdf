from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol, Union

import natural_pdf
from natural_pdf.services.registry import register_delegate

if TYPE_CHECKING:  # pragma: no cover
    from natural_pdf.elements.region import Region
    from natural_pdf.flows.collections import FlowElementCollection, FlowRegionCollection
    from natural_pdf.flows.region import FlowRegion


class _DirectionalHost(Protocol):
    def _direction(self, *args, **kwargs) -> Union["Region", "FlowRegion"]: ...


class NavigationService:
    """Directional helpers formerly implemented via DirectionalMixin methods."""

    def __init__(self, context):
        self._context = context

    def _option(self, host: _DirectionalHost, key: str):
        """Resolve PDFContext navigation defaults with global compatibility fallback."""

        global_default = getattr(natural_pdf.options.layout, key)
        host_context = getattr(host, "_context", self._context)
        context = host_context if hasattr(host_context, "get_option") else self._context
        return context.get_option("layout", key, host=host, default=global_default)

    def _offset(self, host: _DirectionalHost, offset: Optional[float]) -> float:
        if offset is not None:
            return offset
        return self._option(host, "directional_offset")

    @register_delegate("navigation", "above")
    def above(
        self,
        host: _DirectionalHost,
        height: Optional[float] = None,
        width: str = "full",
        include_source: bool = False,
        until: Optional[str] = None,
        include_endpoint: bool = True,
        offset: Optional[float] = None,
        apply_exclusions: bool = True,
        multipage: Optional[bool] = None,
        within: Optional["Region"] = None,
        anchor: str = "start",
        **kwargs,
    ):
        return host._direction(
            direction="above",
            size=height,
            cross_size=width,
            include_source=include_source,
            until=until,
            include_endpoint=include_endpoint,
            offset=self._offset(host, offset),
            apply_exclusions=apply_exclusions,
            multipage=multipage,
            within=within,
            anchor=anchor,
            **kwargs,
        )

    @register_delegate("navigation", "below")
    def below(
        self,
        host: _DirectionalHost,
        height: Optional[float] = None,
        width: str = "full",
        include_source: bool = False,
        until: Optional[str] = None,
        include_endpoint: bool = True,
        offset: Optional[float] = None,
        apply_exclusions: bool = True,
        multipage: Optional[bool] = None,
        within: Optional["Region"] = None,
        anchor: str = "start",
        **kwargs,
    ):
        return host._direction(
            direction="below",
            size=height,
            cross_size=width,
            include_source=include_source,
            until=until,
            include_endpoint=include_endpoint,
            offset=self._offset(host, offset),
            apply_exclusions=apply_exclusions,
            multipage=multipage,
            within=within,
            anchor=anchor,
            **kwargs,
        )

    @register_delegate("navigation", "left")
    def left(
        self,
        host: _DirectionalHost,
        width: Optional[float] = None,
        height: str = "element",
        include_source: bool = False,
        until: Optional[str] = None,
        include_endpoint: bool = True,
        offset: Optional[float] = None,
        apply_exclusions: bool = True,
        multipage: Optional[bool] = None,
        within: Optional["Region"] = None,
        anchor: str = "start",
        **kwargs,
    ):
        return host._direction(
            direction="left",
            size=width,
            cross_size=height,
            include_source=include_source,
            until=until,
            include_endpoint=include_endpoint,
            offset=self._offset(host, offset),
            apply_exclusions=apply_exclusions,
            multipage=multipage,
            within=within,
            anchor=anchor,
            **kwargs,
        )

    @register_delegate("navigation", "right")
    def right(
        self,
        host: _DirectionalHost,
        width: Optional[float] = None,
        height: str = "element",
        include_source: bool = False,
        until: Optional[str] = None,
        include_endpoint: bool = True,
        offset: Optional[float] = None,
        apply_exclusions: bool = True,
        multipage: Optional[bool] = None,
        within: Optional["Region"] = None,
        anchor: str = "start",
        **kwargs,
    ):
        return host._direction(
            direction="right",
            size=width,
            cross_size=height,
            include_source=include_source,
            until=until,
            include_endpoint=include_endpoint,
            offset=self._offset(host, offset),
            apply_exclusions=apply_exclusions,
            multipage=multipage,
            within=within,
            anchor=anchor,
            **kwargs,
        )

    def flow_element_collection(
        self,
        collection: "FlowElementCollection",
        method_name: str,
        **kwargs,
    ) -> "FlowRegionCollection":
        from natural_pdf.flows.collections import FlowRegionCollection

        results = []
        for flow_element in collection.flow_elements:
            method = getattr(flow_element, method_name)
            results.append(method(**kwargs))
        return FlowRegionCollection(results)

    def flow_region_collection(
        self,
        collection: "FlowRegionCollection",
        method_name: str,
        **kwargs,
    ) -> "FlowRegionCollection":
        from natural_pdf.flows.collections import FlowRegionCollection

        if not collection.flow_regions:
            return FlowRegionCollection([])

        resolved: list["FlowRegion"] = []
        for region in collection.flow_regions:
            method = getattr(region, method_name)
            resolved.append(method(**kwargs))
        return FlowRegionCollection(resolved)
