from types import SimpleNamespace

import pytest

from natural_pdf.core.page_collection import PageCollection
from natural_pdf.core.pdf import PDF
from natural_pdf.core.pdf_collection import PDFCollection
from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.flows.flow import Flow


@pytest.mark.parametrize("legacy", [True, False, None, 1])
def test_aggregate_ocr_hosts_reject_legacy_replace_values_before_work(legacy):
    hosts = [
        lambda: PageCollection([]).apply_ocr(replace=legacy),
        lambda: PDF.apply_ocr(object.__new__(PDF), replace=legacy),
        lambda: PDFCollection([]).apply_ocr(replace=legacy),
        lambda: Flow.apply_ocr(object.__new__(Flow), replace=legacy),
        lambda: ElementCollection([]).apply_ocr(replace=legacy),
    ]

    for invoke in hosts:
        with pytest.raises(TypeError, match="replace must be one of"):
            invoke()


def test_page_collection_normalizes_replace_before_forwarding():
    calls = []
    page = SimpleNamespace(_execute_ocr_request=lambda request: calls.append(request))

    result = PageCollection([page]).apply_ocr(replace=" OCR ")

    assert result.pages == [page]
    assert calls[0].replace == "ocr"


def test_element_collection_forwards_normalized_replace_mode():
    calls = []
    element = SimpleNamespace(_execute_ocr_request=lambda request: calls.append(request))

    result = ElementCollection([element]).apply_ocr(replace=" NONE ")

    assert result[0] is element
    assert len(calls) == 1
    assert calls[0].replace == "none"
