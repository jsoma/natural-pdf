"""Bundled mixins that represent high-level document capabilities."""

from __future__ import annotations

from natural_pdf.core.geometry_mixin import RegionGeometryMixin
from natural_pdf.core.mixins import SinglePageContextMixin
from natural_pdf.elements.base import DirectionalMixin


class SpatialRegionMixin(DirectionalMixin, RegionGeometryMixin):
    """Directional navigation + geometry helpers for region-like hosts."""


class MultiRegionAnalysisMixin(RegionGeometryMixin):
    """Analysis bundle for multi-region containers (e.g., FlowRegion)."""
