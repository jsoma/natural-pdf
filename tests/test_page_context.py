import inspect
from pathlib import Path

from natural_pdf import PDF
from natural_pdf.analyzers.guides.helpers import _resolve_single_page
from natural_pdf.core.interfaces import HasSinglePage, SupportsBBox, SupportsGeometry
from natural_pdf.elements.base import extract_bbox
from natural_pdf.services.shape_detection_service import _shape_proxy_factory
from natural_pdf.utils.page_context import resolve_page_context


def test_shape_proxy_preserves_dynamic_region_bbox_for_context_resolution():
    pdf = PDF(Path("pdfs/needs-ocr.pdf"))
    try:
        page = pdf.pages[0]
        region = page.region(0, 100, 200, 300)
        proxy = _shape_proxy_factory(region)

        assert inspect.getattr_static(proxy, "bbox") is not None
        assert isinstance(proxy, HasSinglePage)
        assert isinstance(proxy, SupportsBBox)
        assert isinstance(proxy, SupportsGeometry)

        assert extract_bbox(proxy) == region.bbox
        resolved_page, bounds = resolve_page_context(proxy)

        assert resolved_page is page
        assert bounds == region.bbox
        assert _resolve_single_page(proxy) is page
    finally:
        pdf.close()


def test_shape_proxy_uses_region_bounds_for_detection_image():
    pdf = PDF(Path("pdfs/needs-ocr.pdf"))
    try:
        page = pdf.pages[0]
        region = page.region(0, 100, 200, 300)
        proxy = _shape_proxy_factory(region)

        image, _scale_factor, origin, resolved_page = proxy._get_image_for_detection(72)

        assert image.shape == (200, 200, 3)
        assert origin == (0.0, 100.0)
        assert resolved_page is page
    finally:
        pdf.close()
