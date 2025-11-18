"""Navigation helpers that forward to the navigation service."""

from __future__ import annotations

from typing import Optional

from natural_pdf.services.base import resolve_service


def above(self, *args, **kwargs):
    """Delegate :meth:`above` to the navigation service."""

    service = resolve_service(self, "navigation")
    return service.above(self, *args, **kwargs)


def below(self, *args, **kwargs):
    """Delegate :meth:`below` to the navigation service."""

    service = resolve_service(self, "navigation")
    return service.below(self, *args, **kwargs)


def left(self, *args, **kwargs):
    """Delegate :meth:`left` to the navigation service."""

    service = resolve_service(self, "navigation")
    return service.left(self, *args, **kwargs)


def right(self, *args, **kwargs):
    """Delegate :meth:`right` to the navigation service."""

    service = resolve_service(self, "navigation")
    return service.right(self, *args, **kwargs)


def flow_element_collection_direction(host, method_name: str, **kwargs):
    """Route flow-element collection navigation through the navigation service."""

    from natural_pdf.flows.collections import FlowRegionCollection

    elements = getattr(host, "_flow_elements", None)
    if not elements:
        return FlowRegionCollection([])

    service_host = getattr(elements[0], "physical_object", elements[0])
    navigation = resolve_service(service_host, "navigation")
    return navigation.flow_element_collection(host, method_name, **kwargs)


def flow_region_collection_direction(host, method_name: str, **kwargs):
    """Route flow-region collection navigation through the navigation service."""

    from natural_pdf.flows.collections import FlowRegionCollection

    regions = getattr(host, "flow_regions", None)
    if not regions:
        return FlowRegionCollection([])

    navigation = resolve_service(regions[0], "navigation")
    return navigation.flow_region_collection(host, method_name, **kwargs)


def flow_element_collection_above(host, **kwargs):
    return flow_element_collection_direction(host, "above", **kwargs)


def flow_element_collection_below(host, **kwargs):
    return flow_element_collection_direction(host, "below", **kwargs)


def flow_element_collection_left(host, **kwargs):
    return flow_element_collection_direction(host, "left", **kwargs)


def flow_element_collection_right(host, **kwargs):
    return flow_element_collection_direction(host, "right", **kwargs)


def flow_region_collection_above(host, **kwargs):
    return flow_region_collection_direction(host, "above", **kwargs)


def flow_region_collection_below(host, **kwargs):
    return flow_region_collection_direction(host, "below", **kwargs)


def flow_region_collection_left(host, **kwargs):
    return flow_region_collection_direction(host, "left", **kwargs)


def flow_region_collection_right(host, **kwargs):
    return flow_region_collection_direction(host, "right", **kwargs)


__all__ = ["above", "below", "left", "right"]
