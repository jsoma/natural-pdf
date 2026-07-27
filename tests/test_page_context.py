import inspect
from pathlib import Path

from natural_pdf import PDF
from natural_pdf.analyzers.guides.helpers import _resolve_single_page
from natural_pdf.core.interfaces import HasSinglePage, SupportsBBox, SupportsGeometry
from natural_pdf.elements.base import extract_bbox
from natural_pdf.services._shape_detection_impl import get_image_for_detection
from natural_pdf.utils.page_context import resolve_page_context


def test_region_preserves_dynamic_bbox_for_context_resolution():
    pdf = PDF(Path("pdfs/needs-ocr.pdf"))
    try:
        page = pdf.pages[0]
        region = page.region(0, 100, 200, 300)

        assert inspect.getattr_static(region, "bbox") is not None
        assert isinstance(region, HasSinglePage)
        assert isinstance(region, SupportsBBox)
        assert isinstance(region, SupportsGeometry)

        assert extract_bbox(region) == region.bbox
        resolved_page, bounds = resolve_page_context(region)

        assert resolved_page is page
        assert bounds == region.bbox
        assert _resolve_single_page(region) is page
    finally:
        pdf.close()


def test_shape_detection_uses_region_bounds_for_detection_image():
    pdf = PDF(Path("pdfs/needs-ocr.pdf"))
    try:
        page = pdf.pages[0]
        region = page.region(0, 100, 200, 300)

        image, _scale_factor, origin, resolved_page = get_image_for_detection(region, 72)

        assert image.shape == (200, 200, 3)
        assert origin == (0.0, 100.0)
        assert resolved_page is page
    finally:
        pdf.close()
