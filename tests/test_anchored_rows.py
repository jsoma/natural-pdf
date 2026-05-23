from __future__ import annotations

import json
from pathlib import Path

import pytest

import natural_pdf as npdf
from natural_pdf.core.page import Page
from natural_pdf.core.page_collection import PageCollection
from natural_pdf.elements.element_collection import ElementCollection


class FakeText:
    def __init__(self, text, bbox, page=None):
        self._text = text
        self.bbox = bbox
        self.x0, self.top, self.x1, self.bottom = bbox
        self.page = page

    def extract_text(self):
        return self._text


class FakePage:
    def __init__(self, number):
        self.number = number
        self._context = None
        self._items = {}

    def add(self, selector, *items):
        self._items.setdefault(selector, []).extend(items)
        for item in items:
            item.page = self

    def find_all(self, selector, **_kwargs):
        return ElementCollection(self._items.get(selector, []))

    def extract_anchored_rows(self, *args, **kwargs):
        return Page.extract_anchored_rows(self, *args, **kwargs)


def test_page_extract_anchored_rows_collects_same_row_text_to_the_right():
    page = FakePage(1)
    anchor = FakeText("12", (20, 100, 35, 110))
    first = FakeText("Hello", (80, 101, 105, 111))
    second = FakeText("world", (110, 100, 140, 110))
    title = FakeText("Title", (80, 75, 110, 85))
    left_note = FakeText("left", (5, 101, 18, 111))
    page.add("line_numbers", anchor)
    page.add("text", anchor, second, title, first, left_note)

    rows = page.extract_anchored_rows("line_numbers", content_selector="text")

    assert len(rows) == 1
    assert rows[0].anchor is anchor
    assert rows[0].text == "Hello world"
    assert list(rows[0].elements) == [first, second]
    assert rows[0].bbox == (20.0, 100.0, 140.0, 111.0)
    assert rows[0].page_number == 1


def test_page_extract_anchored_rows_can_collect_both_sides_without_anchor_text():
    page = FakePage(1)
    anchor = FakeText("ID-7", (50, 100, 70, 110))
    left = FakeText("left", (10, 100, 30, 110))
    right = FakeText("right", (80, 100, 110, 110))
    page.add("ids", anchor)
    page.add("text", anchor, right, left)

    rows = page.extract_anchored_rows("ids", side="both")

    assert rows[0].text == "left right"
    assert list(rows[0].elements) == [left, right]


def test_page_extract_anchored_rows_accepts_apply_exclusions_selector_option():
    page = FakePage(1)
    anchor = FakeText("1", (20, 100, 30, 110))
    text = FakeText("body", (80, 100, 110, 110))
    page.add("line_numbers", anchor)
    page.add("text", anchor, text)

    rows = page.extract_anchored_rows(
        "line_numbers",
        content_selector="text",
        apply_exclusions=False,
    )

    assert [row.text for row in rows] == ["body"]


def test_page_collection_extract_anchored_rows_preserves_page_order():
    page1 = FakePage(1)
    page2 = FakePage(2)
    anchor1 = FakeText("1", (20, 100, 30, 110))
    anchor2 = FakeText("2", (20, 120, 30, 130))
    text1 = FakeText("first", (80, 100, 110, 110))
    text2 = FakeText("second", (80, 120, 120, 130))
    page1.add("line_numbers", anchor1)
    page1.add("text", anchor1, text1)
    page2.add("line_numbers", anchor2)
    page2.add("text", anchor2, text2)

    rows = PageCollection([page1, page2]).extract_anchored_rows("line_numbers")

    assert [row.text for row in rows] == ["first", "second"]
    assert [row.page_number for row in rows] == [1, 2]


def test_extract_anchored_rows_matches_checked_georgia_margin_line_gold():
    pdf_path = Path("bad-pdfs/submissions/20252026-236232.pdf")
    expected_path = Path(
        "experiments/llm_agent_harness_workspace/tasks/real_georgia_markup_diff_lines/expected.json"
    )
    if not pdf_path.exists() or not expected_path.exists():
        pytest.skip("Georgia real-PDF gold is not available in this checkout")

    pdf = npdf.PDF(str(pdf_path))
    try:
        expected = json.loads(expected_path.read_text())
        rows = pdf.pages.extract_anchored_rows(
            lambda page: [
                element
                for element in page.find_all("text")
                if element.extract_text().strip().isdigit() and element.x1 < 70
            ],
            side="right",
            y_tolerance=4,
        )
        clean_text = [row.text for row in rows if row.text.strip()]
    finally:
        pdf.close()

    assert len(rows) == expected["excluded_line_number_count"] == 181
    assert clean_text == expected["clean_text"]
