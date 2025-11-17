from __future__ import annotations

from typing import Any

from natural_pdf.services.registry import register_delegate


class DescribeService:
    """Service powering describe/inspect helpers."""

    def __init__(self, context):
        self._context = context

    @register_delegate("describe", "describe")
    def describe(self, host) -> Any:
        from natural_pdf.core.page import Page
        from natural_pdf.describe import (
            describe_collection,
            describe_element,
            describe_page,
            describe_region,
        )
        from natural_pdf.describe.summary import ElementSummary
        from natural_pdf.elements.base import Element
        from natural_pdf.elements.element_collection import ElementCollection
        from natural_pdf.elements.region import Region

        if isinstance(host, Page):
            return describe_page(host)
        if isinstance(host, ElementCollection):
            return describe_collection(host)
        if isinstance(host, Region):
            return describe_region(host)
        if isinstance(host, Element):
            return describe_element(host)

        class_name = host.__class__.__name__
        data = {
            "object_type": class_name,
            "message": f"Describe not fully implemented for {class_name}",
        }
        return ElementSummary(data, f"{class_name} Summary")

    @register_delegate("describe", "inspect")
    def inspect(self, host, limit: int = 30) -> Any:
        from natural_pdf.describe import inspect_collection
        from natural_pdf.describe.summary import InspectionSummary
        from natural_pdf.elements.element_collection import ElementCollection

        if not isinstance(host, ElementCollection):
            raise TypeError("inspect() is only available on ElementCollection instances.")

        result = inspect_collection(host, limit=limit)
        if isinstance(result, InspectionSummary):
            return result
        return InspectionSummary(result, "Inspection")
