"""
Mixin for describe functionality.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from natural_pdf.describe.summary import ElementSummary, InspectionSummary


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
        from natural_pdf.describe import (
            describe_collection,
            describe_element,
            describe_page,
            describe_region,
        )

        # Determine the appropriate describe function based on class type
        class_name = self.__class__.__name__

        if class_name == "Page":
            return describe_page(self)
        elif class_name == "ElementCollection":
            return describe_collection(self)
        elif class_name == "Region":
            return describe_region(self)
        else:
            # Check if it's an individual element (inherits from Element base class)
            from natural_pdf.elements.base import Element

            if isinstance(self, Element):
                return describe_element(self)

            # Fallback - try to determine based on available methods/attributes
            if hasattr(self, "get_elements") and hasattr(self, "width") and hasattr(self, "height"):
                # Looks like a page or region
                if hasattr(self, "number"):
                    return describe_page(self)  # Page
                else:
                    return describe_region(self)  # Region
            elif hasattr(self, "__iter__") and hasattr(self, "__len__"):
                # Looks like a collection
                return describe_collection(self)
            else:
                # Unknown type - create a basic summary
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

        return inspect_collection(self, limit=limit)
