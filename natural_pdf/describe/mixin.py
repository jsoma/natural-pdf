"""
Mixin for describe functionality.
"""

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from natural_pdf.core.page import Page
    from natural_pdf.describe.summary import ElementSummary, InspectionSummary
    from natural_pdf.elements.base import Element
    from natural_pdf.elements.element_collection import ElementCollection
    from natural_pdf.elements.region import Region


class DescribeMixin:
    """
    Mixin providing describe functionality for pages, collections, and regions.

    Classes that inherit from this mixin get:
    - .describe() method for high-level summaries
    - .inspect() method for detailed tabular views (collections only)
    """

    def describe(self) -> "ElementSummary":
        """
        Describe this object with type-specific analysis.

        Returns:
            ElementSummary with analysis appropriate for the object type
        """
        from natural_pdf.core.page import Page
        from natural_pdf.describe import (
            describe_collection,
            describe_element,
            describe_page,
            describe_region,
        )
        from natural_pdf.elements.base import Element
        from natural_pdf.elements.element_collection import ElementCollection
        from natural_pdf.elements.region import Region

        # Determine the appropriate describe function based on actual type
        if isinstance(self, Page):
            return describe_page(self)
        if isinstance(self, ElementCollection):
            return describe_collection(self)
        if isinstance(self, Region):
            return describe_region(self)
        if isinstance(self, Element):
            return describe_element(self)

        # Fallback - try to determine based on available methods/attributes
        class_name = self.__class__.__name__
        if hasattr(self, "get_elements") and hasattr(self, "width") and hasattr(self, "height"):
            if hasattr(self, "number"):
                return describe_page(cast(Page, self))
            return describe_region(cast(Region, self))
        if hasattr(self, "__iter__") and hasattr(self, "__len__"):
            return describe_collection(cast(ElementCollection, self))

        from natural_pdf.describe.summary import ElementSummary

        data = {
            "object_type": class_name,
            "message": f"Describe not fully implemented for {class_name}",
        }
        return ElementSummary(data, f"{class_name} Summary")


class InspectMixin:
    """
    Mixin providing inspect functionality for collections.

    Classes that inherit from this mixin get:
    - .inspect() method for detailed tabular element views
    """

    def inspect(self, limit: int = 30) -> "InspectionSummary":
        """
        Inspect elements with detailed tabular view.

        Args:
            limit: Maximum elements per type to show (default: 30)

        Returns:
            InspectionSummary with element tables showing coordinates,
            properties, and other details for each element
        """
        from natural_pdf.describe import inspect_collection
        from natural_pdf.elements.element_collection import ElementCollection

        collection = cast(ElementCollection, self)
        return inspect_collection(collection, limit=limit)
