"""
Describe functionality for natural-pdf.

Provides summary and inspection methods for pages, collections, and regions.
"""

from .base import (
    describe_collection,
    describe_element,
    describe_page,
    describe_region,
    inspect_collection,
)
from .mixin import DescribeMixin, InspectMixin
from .summary import ElementSummary, InspectionSummary

__all__ = [
    "describe_page",
    "describe_collection",
    "inspect_collection",
    "describe_region",
    "describe_element",
    "ElementSummary",
    "InspectionSummary",
    "DescribeMixin",
    "InspectMixin",
]
