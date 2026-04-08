from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest


def test_or_selectors_preserve_branch_local_post_semantics(practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]

    first_text = page.find_all("text:first")
    last_text = page.find_all("text:last")
    combined = page.find_all("text:first|text:last")

    assert first_text and last_text and combined
    assert [id(el) for el in combined.elements] == [
        id(first_text.elements[0]),
        id(last_text.elements[0]),
    ]


def test_or_selectors_preserve_branch_local_relational_semantics(practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]

    separate = page.find_all("text:above(text:last)")
    combined = page.find_all("text:above(text:last)|text:last")

    assert separate and combined
    separate_ids = {id(el) for el in separate.elements}
    combined_ids = {id(el) for el in combined.elements}
    assert separate_ids.issubset(combined_ids)
    assert id(page.find_all("text:last").elements[0]) in combined_ids


def test_region_find_all_honors_closest_semantics(practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]
    region = page.create_region(0, 0, page.width, page.height)

    page_results = page.find_all('text:closest("Durham@0.4")')
    region_results = region.find_all('text:closest("Durham@0.4")')

    assert [el.text for el in region_results.elements] == [el.text for el in page_results.elements]


def test_region_find_all_honors_ocr_semantics(practice_pdf_fresh):
    if importlib.util.find_spec("rapidfuzz") is None:
        pytest.skip("rapidfuzz is required for :ocr() selector matching")

    page = practice_pdf_fresh.pages[0]
    created = page.create_text_elements_from_ocr(
        [
            {
                "bbox": (10, 10, 60, 20),
                "text": "Durharn",
                "confidence": 0.99,
            }
        ],
        scale_x=1.0,
        scale_y=1.0,
    )
    assert created

    region = page.create_region(0, 0, page.width, page.height)
    matches = region.find_all('text:ocr("Durham@0.4")')

    assert any(match.text == "Durharn" for match in matches.elements)


def test_flow_geometry_errors_fail_closed(practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]
    bad_region = page.create_region(0, 0, page.width, page.height)

    def _boom(_element):
        raise RuntimeError("boom")

    bad_region.intersects = _boom
    flow = SimpleNamespace(segments=[bad_region])

    results = page.services.selector._find_all_flow(flow, selector="text")

    assert len(results) == 0
