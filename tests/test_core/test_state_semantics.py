from __future__ import annotations

from pathlib import Path

import pytest

from natural_pdf import PDF


def _two_page_pdf_bytes() -> bytes:
    return b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 5 0 R /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >>
endobj
4 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 6 0 R /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >>
endobj
5 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
72 720 Td
(Page 1 Test) Tj
ET
endstream
endobj
6 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
72 720 Td
(Page 2 Test) Tj
ET
endstream
endobj
xref
0 7
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000264 00000 n
0000000413 00000 n
0000000507 00000 n
trailer
<< /Size 7 /Root 1 0 R >>
startxref
601
%%EOF"""


def test_exclusion_changes_do_not_drop_synthetic_ocr_elements(practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]

    before_words = len(page.words)
    created = page.create_text_elements_from_ocr(
        [{"text": "TEST", "bbox": (10, 10, 40, 30), "confidence": 0.9}],
        scale_x=1.0,
        scale_y=1.0,
    )
    assert len(created) == 1
    assert len(page.words) == before_words + 1

    page.add_exclusion(page.create_region(0, 0, 50, 50), label="mask")
    after_add = [
        word
        for word in page.words
        if getattr(word, "source", None) == "ocr" and getattr(word, "text", None) == "TEST"
    ]
    assert len(after_add) == 1
    assert len(page.words) == before_words + 1

    page.clear_exclusions()
    after_clear = [
        word
        for word in page.words
        if getattr(word, "source", None) == "ocr" and getattr(word, "text", None) == "TEST"
    ]
    assert len(after_clear) == 1
    assert len(page.words) == before_words + 1


def test_unlabeled_pdf_exclusions_do_not_duplicate(practice_pdf_fresh):
    pdf = practice_pdf_fresh
    pdf.add_exclusion(lambda page: page.create_region(0, 0, 100, 100))

    page = pdf.pages[0]
    regions = page._get_exclusion_regions(include_callable=True)

    assert len(page._exclusions) == 0
    assert len(regions) == 1


def test_empty_selector_results_preserve_originating_context(practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]

    page_result = page.find_all('text:contains("__definitely_missing__")')
    region_result = page.create_region(0, 0, 20, 20).find_all(
        'text:contains("__definitely_missing__")'
    )

    assert len(page_result) == 0
    assert len(region_result) == 0
    assert page_result._context is page._context
    assert region_result._context is page._context
    assert page_result.services.selector is page.services.selector


def test_empty_pdf_and_page_slices_preserve_context(practice_pdf_fresh):
    pdf = practice_pdf_fresh

    empty_pdf_slice = pdf[:0]
    empty_page_slice = pdf.pages[:0]

    assert len(empty_pdf_slice) == 0
    assert len(empty_page_slice) == 0
    assert empty_pdf_slice._context is pdf._context
    assert empty_page_slice._context is pdf._context


def test_remove_regions_counts_logical_region_once(practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]
    region = page.create_region(0, 0, 50, 50)
    page.add_region(region, name="header")

    removed = page.remove_regions(
        source="named", predicate=lambda current: current.name == "header"
    )

    assert removed == 1
    assert "header" not in page._regions["named"]
    assert all(getattr(current, "name", None) != "header" for current in page._element_mgr.regions)


def test_close_is_partial_but_blocks_live_backing_operations(tmp_path: Path):
    pdf_path = tmp_path / "two-pages.pdf"
    pdf_path.write_bytes(_two_page_pdf_bytes())

    pdf = PDF(str(pdf_path))
    page = pdf.pages[0]
    cached_word_count = len(page.words)

    pdf.close()

    assert page.number == 1
    assert len(page.words) == cached_word_count

    with pytest.raises(RuntimeError, match="closed"):
        _ = pdf.pages[1]

    with pytest.raises(RuntimeError, match="closed"):
        page.apply_ocr(engine="easyocr")
