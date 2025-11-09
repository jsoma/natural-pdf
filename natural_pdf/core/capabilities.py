"""Bundled mixins that represent high-level document capabilities."""

from __future__ import annotations

from natural_pdf.analyzers.checkbox.mixin import CheckboxDetectionMixin
from natural_pdf.analyzers.shape_detection_mixin import ShapeDetectionMixin
from natural_pdf.classification.mixin import ClassificationMixin
from natural_pdf.core.exclusion_mixin import ExclusionMixin
from natural_pdf.core.geometry_mixin import RegionGeometryMixin
from natural_pdf.core.interfaces import SupportsSections
from natural_pdf.core.mixins import SinglePageContextMixin
from natural_pdf.core.multi_region_mixins import (
    MultiRegionDirectionalMixin,
    MultiRegionExclusionMixin,
    MultiRegionOCRMixin,
)
from natural_pdf.core.ocr_mixin import OCRMixin
from natural_pdf.core.qa_mixin import DocumentQAMixin
from natural_pdf.core.table_mixin import TableExtractionMixin
from natural_pdf.describe.mixin import DescribeMixin
from natural_pdf.elements.base import DirectionalMixin
from natural_pdf.extraction.mixin import ExtractionMixin
from natural_pdf.selectors.host_mixin import SelectorHostMixin
from natural_pdf.vision.mixin import VisualSearchMixin


class AnalysisHostMixin(
    ClassificationMixin,
    ExtractionMixin,
    ShapeDetectionMixin,
    CheckboxDetectionMixin,
    DescribeMixin,
    VisualSearchMixin,
    ExclusionMixin,
    OCRMixin,
    DocumentQAMixin,
    SinglePageContextMixin,
    SupportsSections,
    SelectorHostMixin,
):
    """Bundle of analysis-oriented mixins used by Page/Region hosts."""


class SpatialRegionMixin(DirectionalMixin, RegionGeometryMixin):
    """Directional navigation + geometry helpers for region-like hosts."""


class TabularRegionMixin(SpatialRegionMixin, TableExtractionMixin):
    """Extends spatial regions with table extraction support."""


class MultiRegionAnalysisMixin(
    DocumentQAMixin,
    MultiRegionOCRMixin,
    MultiRegionExclusionMixin,
    MultiRegionDirectionalMixin,
    RegionGeometryMixin,
    SelectorHostMixin,
):
    """Analysis bundle for multi-region containers (e.g., FlowRegion)."""
