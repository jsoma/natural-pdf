"""Tests for the flows-system cleanup: extracted helpers, stack_images, and new
FlowRegion properties (is_empty, parts, map_parts, multi-page bbox)."""

from __future__ import annotations

from typing import Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from natural_pdf import PDF
from natural_pdf.flows import Flow
from natural_pdf.flows._utils import stack_images
from natural_pdf.flows.element import FlowElement
from natural_pdf.flows.region import FlowRegion

# ---------------------------------------------------------------------------
# Helpers to build lightweight fakes without loading real PDFs
# ---------------------------------------------------------------------------


def _make_page(index: int = 0, page_number: int = 1):
    page = MagicMock()
    page.index = index
    page.page_number = page_number
    page.number = page_number
    return page


def _make_region(page, bbox: Tuple[float, float, float, float]):
    region = MagicMock()
    region.page = page
    region.bbox = bbox
    region.x0, region.top, region.x1, region.bottom = bbox
    region.width = bbox[2] - bbox[0]
    region.height = bbox[3] - bbox[1]
    region.has_polygon = False
    region.polygon = [
        (bbox[0], bbox[1]),
        (bbox[2], bbox[1]),
        (bbox[2], bbox[3]),
        (bbox[0], bbox[3]),
    ]
    region.is_point_inside = lambda cx, cy: (bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3])
    return region


def _make_element(page, bbox: Tuple[float, float, float, float]):
    elem = MagicMock()
    elem.page = page
    elem.bbox = bbox
    elem.x0, elem.top, elem.x1, elem.bottom = bbox
    elem.width = bbox[2] - bbox[0]
    elem.height = bbox[3] - bbox[1]
    elem.has_polygon = False
    return elem


# ===========================================================================
# 1. stack_images tests
# ===========================================================================


class TestStackImages:
    def test_empty_returns_none(self):
        assert stack_images([]) is None

    def test_single_image_returned_as_is(self):
        img = Image.new("RGB", (50, 30), (255, 0, 0))
        result = stack_images([img])
        assert result is img

    def test_vertical_dimensions(self):
        a = Image.new("RGB", (100, 40), (255, 0, 0))
        b = Image.new("RGB", (80, 60), (0, 255, 0))
        result = stack_images([a, b], direction="vertical", gap=10)
        assert result is not None
        assert result.width == 100  # max width
        assert result.height == 40 + 60 + 10  # heights + gap

    def test_horizontal_dimensions(self):
        a = Image.new("RGB", (100, 40), (255, 0, 0))
        b = Image.new("RGB", (80, 60), (0, 255, 0))
        result = stack_images([a, b], direction="horizontal", gap=5)
        assert result is not None
        assert result.width == 100 + 80 + 5
        assert result.height == 60  # max height

    def test_vertical_with_separator(self):
        a = Image.new("RGB", (50, 20))
        b = Image.new("RGB", (50, 20))
        result = stack_images(
            [a, b],
            direction="vertical",
            gap=0,
            separator_color=(255, 0, 0),
            separator_thickness=4,
        )
        assert result is not None
        assert result.height == 20 + 20 + 4

    def test_horizontal_with_separator(self):
        a = Image.new("RGB", (30, 50))
        b = Image.new("RGB", (30, 50))
        result = stack_images(
            [a, b],
            direction="horizontal",
            gap=2,
            separator_color=(0, 0, 255),
            separator_thickness=3,
        )
        assert result is not None
        assert result.width == 30 + 30 + 2 + 3

    def test_invalid_direction(self):
        img = Image.new("RGB", (10, 10))
        with pytest.raises(ValueError, match="Invalid direction"):
            stack_images([img, img], direction="diagonal")

    def test_three_images_vertical(self):
        imgs = [Image.new("RGB", (40, 20)) for _ in range(3)]
        result = stack_images(imgs, direction="vertical", gap=5)
        assert result is not None
        assert result.height == 20 * 3 + 5 * 2
        assert result.width == 40


# ===========================================================================
# 2. FlowElement helper tests
# ===========================================================================


class TestFindStartSegment:
    """Tests for FlowElement._find_start_segment()."""

    def test_center_inside_segment(self):
        page = _make_page()
        seg = _make_region(page, (0, 0, 100, 100))
        elem = _make_element(page, (40, 40, 60, 60))

        flow = MagicMock()
        flow.segments = [seg]
        fe = FlowElement(elem, flow)
        assert fe._find_start_segment() == 0

    def test_element_on_different_page(self):
        page1 = _make_page(0)
        page2 = _make_page(1)
        seg = _make_region(page1, (0, 0, 100, 100))
        elem = _make_element(page2, (10, 10, 20, 20))

        flow = MagicMock()
        flow.segments = [seg]
        fe = FlowElement(elem, flow)
        assert fe._find_start_segment() == -1

    def test_bbox_overlap_fallback(self):
        """When center is outside but bboxes overlap, fall back to overlap detection."""
        page = _make_page()
        # Segment in top-left quadrant
        seg = _make_region(page, (0, 0, 50, 50))
        # Element overlapping but with center outside (center at 55, 55)
        elem = _make_element(page, (45, 45, 65, 65))
        # Center is (55, 55), outside of (0,0,50,50)
        # But bboxes overlap

        flow = MagicMock()
        flow.segments = [seg]
        fe = FlowElement(elem, flow)
        assert fe._find_start_segment() == 0

    def test_multiple_segments_picks_first_center_match(self):
        page = _make_page()
        seg0 = _make_region(page, (0, 0, 100, 100))
        seg1 = _make_region(page, (0, 100, 100, 200))
        elem = _make_element(page, (40, 140, 60, 160))

        flow = MagicMock()
        flow.segments = [seg0, seg1]
        fe = FlowElement(elem, flow)
        assert fe._find_start_segment() == 1


class TestResolveCrossSize:
    def test_absolute_takes_priority(self):
        page = _make_page()
        elem = _make_element(page, (10, 20, 110, 70))
        flow = MagicMock()
        fe = FlowElement(elem, flow)

        result = fe._resolve_cross_size("below", cross_size_ratio=2.0, cross_size_absolute=42.0)
        assert result == 42.0

    def test_ratio_for_vertical_direction(self):
        page = _make_page()
        elem = _make_element(page, (10, 20, 110, 70))  # width=100, height=50
        flow = MagicMock()
        fe = FlowElement(elem, flow)

        result = fe._resolve_cross_size("below", cross_size_ratio=2.0, cross_size_absolute=None)
        assert result == 200.0  # width * ratio for above/below

    def test_ratio_for_horizontal_direction(self):
        page = _make_page()
        elem = _make_element(page, (10, 20, 110, 70))  # width=100, height=50
        flow = MagicMock()
        fe = FlowElement(elem, flow)

        result = fe._resolve_cross_size("right", cross_size_ratio=2.0, cross_size_absolute=None)
        assert result == 100.0  # height * ratio for left/right

    def test_default_left_right_uses_element_height(self):
        page = _make_page()
        elem = _make_element(page, (10, 20, 110, 70))
        flow = MagicMock()
        fe = FlowElement(elem, flow)

        result = fe._resolve_cross_size("left", cross_size_ratio=None, cross_size_absolute=None)
        assert result == 50.0  # element height

    def test_default_above_below_uses_full(self):
        page = _make_page()
        elem = _make_element(page, (10, 20, 110, 70))
        flow = MagicMock()
        fe = FlowElement(elem, flow)

        result = fe._resolve_cross_size("above", cross_size_ratio=None, cross_size_absolute=None)
        assert result == "full"


class TestBuildSegmentIterator:
    def test_below_forward(self):
        rng, is_fwd = FlowElement._build_segment_iterator("below", 1, 5)
        assert list(rng) == [1, 2, 3, 4]
        assert is_fwd is True

    def test_above_backward(self):
        rng, is_fwd = FlowElement._build_segment_iterator("above", 2, 5)
        assert list(rng) == [2, 1, 0]
        assert is_fwd is False

    def test_right_forward(self):
        rng, is_fwd = FlowElement._build_segment_iterator("right", 0, 3)
        assert list(rng) == [0, 1, 2]
        assert is_fwd is True

    def test_left_backward(self):
        rng, is_fwd = FlowElement._build_segment_iterator("left", 2, 3)
        assert list(rng) == [2, 1, 0]
        assert is_fwd is False

    def test_invalid_direction(self):
        with pytest.raises(ValueError, match="Invalid direction"):
            FlowElement._build_segment_iterator("diagonal", 0, 3)


class TestClipToBudget:
    def test_under_budget_no_clip(self):
        region = MagicMock()
        region.height = 30.0
        region.width = 30.0
        clipped, remaining = FlowElement._clip_to_budget(region, 100.0, "below", True)
        assert clipped is region
        assert remaining == pytest.approx(70.0)

    def test_over_budget_clips_vertical_forward(self):
        region = MagicMock()
        region.height = 100.0
        region.width = 50.0
        region.top = 10.0
        region.bottom = 110.0

        clipped_result = MagicMock()
        region.clip = MagicMock(return_value=clipped_result)

        result, remaining = FlowElement._clip_to_budget(region, 40.0, "below", True)
        assert result is clipped_result
        assert remaining == pytest.approx(0.0)
        region.clip.assert_called_once_with(bottom=50.0, top=None)

    def test_exact_budget(self):
        region = MagicMock()
        region.height = 50.0
        region.width = 50.0
        clipped, remaining = FlowElement._clip_to_budget(region, 50.0, "above", False)
        assert clipped is region
        assert remaining == pytest.approx(0.0)


# ===========================================================================
# 3. FlowRegion property tests (is_empty, parts, map_parts, bbox)
# ===========================================================================


class TestFlowRegionProperties:
    def _make_flow_region(self, regions, flow=None):
        if flow is None:
            flow = MagicMock()
            flow._context = None
        return FlowRegion(flow=flow, constituent_regions=regions)

    def test_is_empty_true(self):
        fr = self._make_flow_region([])
        assert fr.is_empty is True

    def test_is_empty_false(self):
        page = _make_page()
        reg = _make_region(page, (0, 0, 100, 100))
        fr = self._make_flow_region([reg])
        assert fr.is_empty is False

    def test_parts_alias(self):
        page = _make_page()
        regs = [_make_region(page, (0, 0, 50, 50)), _make_region(page, (50, 0, 100, 50))]
        fr = self._make_flow_region(regs)
        assert fr.parts is fr.constituent_regions

    def test_map_parts(self):
        page = _make_page()
        regs = [_make_region(page, (0, 0, 50, 50)), _make_region(page, (50, 0, 100, 50))]
        fr = self._make_flow_region(regs)
        widths = fr.map_parts(lambda r: r.width)
        assert widths == [50.0, 50.0]


class TestFlowRegionBbox:
    def _make_flow_region(self, regions, flow=None):
        if flow is None:
            flow = MagicMock()
            flow._context = None
        return FlowRegion(flow=flow, constituent_regions=regions)

    def test_single_page_bbox_valid(self):
        page = _make_page()
        r1 = _make_region(page, (10, 20, 100, 200))
        r2 = _make_region(page, (50, 100, 150, 300))
        fr = self._make_flow_region([r1, r2])
        bbox = fr.bbox
        assert bbox is not None
        assert bbox == (10, 20, 150, 300)

    def test_multi_page_bbox_still_returns_merged(self):
        """Multi-page bbox returns a merged bbox (useful for sorting), not None."""
        page1 = _make_page(0)
        page2 = _make_page(1)
        r1 = _make_region(page1, (0, 0, 100, 100))
        r2 = _make_region(page2, (10, 10, 200, 200))
        fr = self._make_flow_region([r1, r2])
        bbox = fr.bbox
        assert bbox is not None
        assert bbox == (0, 0, 200, 200)

    def test_multi_page_coordinate_properties_work(self):
        """Coordinate properties should work even for multi-page FlowRegions."""
        page1 = _make_page(0)
        page2 = _make_page(1)
        r1 = _make_region(page1, (5, 10, 100, 100))
        r2 = _make_region(page2, (0, 0, 200, 200))
        fr = self._make_flow_region([r1, r2])
        assert fr.x0 == 0
        assert fr.top == 0
        assert fr.x1 == 200
        assert fr.bottom == 200

    def test_empty_bbox_returns_none(self):
        fr = self._make_flow_region([])
        assert fr.bbox is None
        with pytest.raises(ValueError, match="no bounding box"):
            _ = fr.x0


# ===========================================================================
# 4. Deleted __getattr__ — verify explicit methods still work
# ===========================================================================


class TestFlowRegionNoGetattr:
    """After __getattr__ removal, accessing unknown attributes should raise normally."""

    def test_unknown_attribute_raises(self):
        flow = MagicMock()
        flow._context = None
        fr = FlowRegion(flow=flow, constituent_regions=[])
        with pytest.raises(AttributeError):
            _ = fr.nonexistent_attr_xyz

    def test_explicit_methods_exist(self):
        """Verify that previously blocklisted methods are still reachable."""
        flow = MagicMock()
        flow._context = None
        fr = FlowRegion(flow=flow, constituent_regions=[])
        # These should all be callable (not raise AttributeError)
        assert callable(getattr(fr, "above", None))
        assert callable(getattr(fr, "below", None))
        assert callable(getattr(fr, "left", None))
        assert callable(getattr(fr, "right", None))
        assert callable(getattr(fr, "to_region", None))


# ===========================================================================
# 5. Integration: _flow_direction via real PDF
# ===========================================================================


@pytest.mark.parametrize("pdf_path", ["pdfs/multicolumn.pdf"])
def test_flow_direction_still_works_end_to_end(pdf_path):
    """Ensure the refactored _flow_direction produces the same results as before."""
    pdf = PDF(pdf_path)
    page = pdf.pages[0]
    col_width = page.width / 3
    columns = [page.region(left=i * col_width, width=col_width) for i in range(3)]
    flow = Flow(columns, arrangement="vertical")

    bold = flow.find("text:bold")
    assert bold is not None, "Should find at least one bold element"

    region = bold.below(until="text:bold")
    assert len(region.constituent_regions) >= 1
    # The region should have valid text
    text = region.extract_text()
    assert len(text) > 0
