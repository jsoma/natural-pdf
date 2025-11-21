from __future__ import annotations

from typing import Dict, List

import pytest

from tests.realpdf.snapshot_utils import (
    DEFAULT_SNAPSHOT_PATH,
    REAL_PDF_CASES,
    RealPDFCase,
    load_snapshots,
)

pytestmark = pytest.mark.realpdf

SNAPSHOTS = load_snapshots(DEFAULT_SNAPSHOT_PATH)


@pytest.fixture(params=REAL_PDF_CASES, ids=lambda case: case.name)
def page_case(request):
    case: RealPDFCase = request.param
    from natural_pdf import PDF

    pdf = PDF(str(case.absolute_path))
    try:
        page = pdf.pages[case.page_index]
        manager = page._element_mgr  # noqa: SLF001 - intentional test access
        yield case, pdf, page, manager
    finally:
        pdf.close()


def _counts(manager) -> Dict[str, int]:
    return {
        "chars": len(manager.chars),
        "words": len(manager.words),
        "rects": len(manager.rects),
        "lines": len(manager.lines),
        "images": len(manager.images),
    }


def test_load_elements_idempotent(page_case):
    case, _, _, manager = page_case
    manager.load_elements()
    first_counts = _counts(manager)
    manager.load_elements()
    second_counts = _counts(manager)
    assert first_counts == second_counts
    expected = SNAPSHOTS[case.name]["elements"]
    for key, counts in first_counts.items():
        assert counts == expected[key]["count"]


def test_accessor_matches_get_elements(page_case):
    _, _, _, manager = page_case
    manager.load_elements()
    for key, attr in [
        ("chars", manager.chars),
        ("words", manager.words),
        ("rects", manager.rects),
        ("lines", manager.lines),
        ("images", manager.images),
    ]:
        gateway = manager.get_elements(key)
        assert attr == gateway


def test_get_all_elements_union(page_case):
    _, _, _, manager = page_case
    manager.load_elements()
    lists: List[List] = [manager.chars, manager.words, manager.rects, manager.lines, manager.images]
    manual_total = sum(len(items) for items in lists)
    union_total = len(manager.get_all_elements())
    assert union_total >= max(len(items) for items in lists)
    assert union_total <= manual_total + len(manager.regions)


def test_remove_ocr_elements_is_noop_without_ocr(page_case):
    _, _, _, manager = page_case
    manager.load_elements()
    removed = manager.remove_ocr_elements()
    assert removed == 0
    removed_words = manager.remove_elements_by_source("words", "ocr")
    removed_chars = manager.remove_elements_by_source("chars", "ocr")
    assert removed_words == 0
    assert removed_chars == 0


def test_invalidate_cache_and_reload(page_case):
    case, _, _, manager = page_case
    manager.load_elements()
    before = _counts(manager)
    manager.invalidate_cache()
    manager.load_elements()
    after = _counts(manager)
    expected_counts = {k: SNAPSHOTS[case.name]["elements"][k]["count"] for k in before}
    assert before == expected_counts
    assert after == expected_counts


def test_create_text_elements_from_ocr_round_trip(page_case):
    _, _, page, manager = page_case
    manager.load_elements()
    before_words = len(manager.words)
    ocr_payload = [
        {"text": "TEST", "bbox": (10, 10, 40, 30), "confidence": 0.9},
    ]
    added = manager.create_text_elements_from_ocr(ocr_payload, scale_x=1.0, scale_y=1.0)
    assert len(added) == 1
    assert len(manager.words) == before_words + 1
    removed = manager.remove_ocr_elements()
    assert removed >= 1
    assert len(manager.words) == before_words


def test_has_elements_flag(page_case):
    case, _, _, manager = page_case
    manager.load_elements()
    expected = SNAPSHOTS[case.name]["elements"]
    tracked_keys = ("words", "rects", "lines", "regions")
    expect_true = any(expected[key]["count"] > 0 for key in tracked_keys if key in expected)
    assert manager.has_elements() == expect_true
