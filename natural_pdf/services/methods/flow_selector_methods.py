"""Flow-aware selector helper functions."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from natural_pdf.services.base import resolve_service


def flow_find(flow, *args, **kwargs):
    from natural_pdf.flows.element import FlowElement

    merged = kwargs
    if args and len(args) > 0:
        if len(args) > 1:
            raise TypeError("flow.find accepts at most one positional selector argument.")
        if "selector" in kwargs:
            raise TypeError("Provide selector either positionally or via keyword, not both.")
        merged = dict(kwargs)
        merged["selector"] = args[0]

    physical = resolve_service(flow, "selector").find(flow, **merged)
    if physical is None:
        return None
    if isinstance(physical, FlowElement):
        if getattr(physical, "flow", None) is flow:
            return physical
        physical = physical.physical_object
    return FlowElement(physical, flow)


def flow_find_all(flow, *args, **kwargs):
    from natural_pdf.flows.collections import FlowElementCollection

    merged = kwargs
    if args and len(args) > 0:
        if len(args) > 1:
            raise TypeError("flow.find_all accepts at most one positional selector argument.")
        if "selector" in kwargs:
            raise TypeError("Provide selector either positionally or via keyword, not both.")
        merged = dict(kwargs)
        merged["selector"] = args[0]

    collection = resolve_service(flow, "selector").find_all(flow, **merged)
    return FlowElementCollection.from_physical(flow, collection.elements)


def flow_element_find(element, *args, **kwargs):
    from natural_pdf.flows.element import FlowElement

    merged = kwargs
    if args and len(args) > 0:
        if len(args) > 1:
            raise TypeError("FlowElement.find accepts at most one positional selector argument.")
        if "selector" in kwargs:
            raise TypeError("Provide selector either positionally or via keyword.")
        merged = dict(kwargs)
        merged["selector"] = args[0]

    physical = resolve_service(element.physical_object, "selector").find(
        element.physical_object, **merged
    )
    if physical is None:
        return None
    if isinstance(physical, FlowElement):
        if getattr(physical, "flow", None) is element.flow:
            return physical
        return FlowElement(physical.physical_object, element.flow)
    return FlowElement(physical, element.flow)


def flow_element_find_all(element, *args, **kwargs):
    from natural_pdf.flows.collections import FlowElementCollection

    merged = kwargs
    if args and len(args) > 0:
        if len(args) > 1:
            raise TypeError(
                "FlowElement.find_all accepts at most one positional selector argument."
            )
        if "selector" in kwargs:
            raise TypeError("Provide selector either positionally or via keyword.")
        merged = dict(kwargs)
        merged["selector"] = args[0]

    collection = resolve_service(element.physical_object, "selector").find_all(
        element.physical_object, **merged
    )
    return FlowElementCollection.from_physical(
        element.flow,
        getattr(collection, "elements", collection),
    )
