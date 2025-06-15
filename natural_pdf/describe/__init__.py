"""
Describe functionality for natural-pdf.

Provides summary and inspection methods for pages, collections, and regions.
"""

from .base import describe_page, describe_collection, inspect_collection, describe_region, describe_element
from .summary import ElementSummary, InspectionSummary
from .mixin import DescribeMixin, InspectMixin

__all__ = [
    'describe_page',
    'describe_collection', 
    'inspect_collection',
    'describe_region',
    'describe_element',
    'ElementSummary',
    'InspectionSummary',
    'DescribeMixin',
    'InspectMixin'
] 