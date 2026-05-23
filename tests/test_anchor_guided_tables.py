from __future__ import annotations

import re
from pathlib import Path

import pytest

import natural_pdf as npdf
from natural_pdf.analyzers.guides import Guides

USE_OF_FORCE_PDF = Path("bad-pdfs/submissions/use-of-force-raw.pdf")
CASE_NUMBER_RE = re.compile(r"\d{2}-\d{4}-\d+")


def _text(element) -> str:
    return element.extract_text().strip()


def _mid_y(element) -> float:
    return (element.top + element.bottom) / 2


def _use_of_force_page():
    if not USE_OF_FORCE_PDF.exists():
        pytest.skip("Use-of-force real PDF is not available in this checkout")
    pdf = npdf.PDF(str(USE_OF_FORCE_PDF))
    return pdf, pdf.pages[0]


def _case_header_and_row_anchors(page):
    texts = list(page.find_all("text"))
    header = page.find('text:contains("CASE #")')
    assert header is not None

    row_anchors = sorted(
        [
            text
            for text in texts
            if CASE_NUMBER_RE.fullmatch(_text(text))
            and text.top > header.top
            and abs(text.x0 - header.x0) <= 8
        ],
        key=lambda text: text.top,
    )
    return texts, header, row_anchors


def _case_number_anchors(page):
    return sorted(
        [
            text
            for text in page.find_all("text")
            if CASE_NUMBER_RE.fullmatch(_text(text)) and abs(text.x0 - 180) <= 8
        ],
        key=lambda text: text.top,
    )


def _assert_row_values(row, expected: dict[str, str]) -> None:
    values = row.to_dict()
    assert {key: values.get(key) for key in expected} == expected


def test_case_number_anchors_cover_all_use_of_force_pages():
    pdf, _page = _use_of_force_page()
    try:
        anchors_by_page = [_case_number_anchors(page) for page in pdf.pages]
        rows = pdf.pages.extract_anchored_rows(
            _case_number_anchors,
            side="both",
            y_tolerance=1.2,
        )
    finally:
        pdf.close()

    assert [len(anchors) for anchors in anchors_by_page] == [298, 299, 299, 74]
    assert len(rows) == 970
    assert [_text(rows[0].anchor), _text(rows[-1].anchor)] == [
        "23-2016-18335",
        "23-2019-10327",
    ]
    assert [rows[0].page_number, rows[-1].page_number] == [1, 4]
    assert "PFEIFER, TIM 12/2/2016 Dec-02-2016 22:30 WEST" in rows[0].text


def test_case_number_anchors_recover_use_of_force_same_row_fields():
    pdf, page = _use_of_force_page()
    try:
        _texts, _header, row_anchors = _case_header_and_row_anchors(page)

        rows = page.extract_anchored_rows(
            row_anchors[:5],
            side="both",
            y_tolerance=1.2,
        )
    finally:
        pdf.close()

    assert [_text(row.anchor) for row in rows[:5]] == [
        "23-2016-18335",
        "23-2016-18348",
        "23-2016-18571",
        "23-2016-18335",
        "23-2016-18818",
    ]
    assert "PFEIFER, TIM 12/2/2016 Dec-02-2016 22:30 WEST" in rows[0].text
    assert "SOLLERS, CECIL ARTHUR" in rows[0].text
    assert "NICHOLSON, DUSTIN 12/3/2016 Dec-03-2016 6:45 WEST" in rows[1].text
    assert "JENNINGS, ERIK 12/7/2016 Dec-07-2016 13:05 EAST" in rows[2].text
    assert "BATES, JUSTIN 12/13/2016 Dec-02-2016 22:30 WEST" in rows[3].text


def test_header_guides_and_case_anchors_extract_use_of_force_sample():
    pdf, page = _use_of_force_page()
    try:
        texts, header, row_anchors = _case_header_and_row_anchors(page)
        header_row = sorted(
            [text for text in texts if abs(_mid_y(text) - _mid_y(header)) <= 1.1 and _text(text)],
            key=lambda text: text.x0,
        )

        guides = Guides(page)
        guides.vertical.from_headers(header_row, method="min_crossings")
        verticals_before = [round(float(value), 2) for value in guides.vertical]
        guides.vertical.snap_to_whitespace(
            min_gap=2,
            detection_method="text",
            on_no_snap="ignore",
        )
        verticals_after = [round(float(value), 2) for value in guides.vertical]
        guides.horizontal.from_content(
            [header, *row_anchors[:12]],
            align="between",
            outer=True,
        )
        table = guides.extract_table(
            cell_extract="words",
            cell_overlap="center",
            header="first",
            include_outer_boundaries=True,
        )
        df = table.to_df().dropna(how="all")
    finally:
        pdf.close()

    moved_verticals = sum(
        1 for before, after in zip(verticals_before, verticals_after) if abs(before - after) > 0.25
    )

    assert [_text(header) for header in header_row[:6]] == [
        "OFFICER",
        "DATE OF REPORT",
        "DATE OF FORCE",
        "TIME OF FORCE",
        "PRECINCT",
        "CASE #",
    ]
    assert len(row_anchors) == 298
    assert moved_verticals == 14
    assert df.shape == (12, 25)
    assert list(df.columns[:6]) == [
        "OFFICER",
        "DATE OF REPORT",
        "DATE OF FORCE",
        "TIME OF FORCE",
        "PRECINCT",
        "CASE #",
    ]
    _assert_row_values(
        df.iloc[0],
        {
            "OFFICER": "PFEIFER, TIM",
            "DATE OF REPORT": "12/2/2016",
            "DATE OF FORCE": "Dec-02-2016",
            "TIME OF FORCE": "22:30",
            "PRECINCT": "WEST",
            "CASE #": "23-2016-18335",
            "SUBJECT PERSON": "SOLLERS, CECIL ARTHUR",
        },
    )
    _assert_row_values(
        df.iloc[1],
        {
            "OFFICER": "NICHOLSON, DUSTIN",
            "DATE OF REPORT": "12/3/2016",
            "DATE OF FORCE": "Dec-03-2016",
            "TIME OF FORCE": "6:45",
            "PRECINCT": "WEST",
            "CASE #": "23-2016-18348",
        },
    )


def test_header_guides_and_dynamic_case_anchors_extract_all_use_of_force_pages():
    pdf, page = _use_of_force_page()
    try:
        texts, header, _row_anchors = _case_header_and_row_anchors(page)
        header_row = sorted(
            [text for text in texts if abs(_mid_y(text) - _mid_y(header)) <= 1.1 and _text(text)],
            key=lambda text: text.x0,
        )

        def page_case_anchors(target_page):
            markers = [header] if target_page.number == page.number else []
            markers.extend(
                sorted(
                    [
                        text
                        for text in target_page.find_all("text")
                        if CASE_NUMBER_RE.fullmatch(_text(text)) and abs(text.x0 - header.x0) <= 8
                    ],
                    key=lambda text: text.top,
                )
            )
            return markers

        guides = Guides(page)
        guides.vertical.from_headers(header_row, method="min_crossings")
        guides.vertical.snap_to_whitespace(
            min_gap=2,
            detection_method="text",
            on_no_snap="ignore",
        )
        guides.horizontal.from_content(
            page_case_anchors,
            align="between",
            outer=True,
        )
        table = guides.extract_table(
            pdf.pages,
            cell_extract="words",
            cell_overlap="center",
            header="first",
            include_outer_boundaries=True,
            show_progress=False,
        )
        df = table.to_df().dropna(how="all")
    finally:
        pdf.close()

    assert df.shape == (970, 25)
    assert list(df.columns[:6]) == [
        "OFFICER",
        "DATE OF REPORT",
        "DATE OF FORCE",
        "TIME OF FORCE",
        "PRECINCT",
        "CASE #",
    ]
    _assert_row_values(
        df.iloc[0],
        {
            "OFFICER": "PFEIFER, TIM",
            "DATE OF REPORT": "12/2/2016",
            "DATE OF FORCE": "Dec-02-2016",
            "TIME OF FORCE": "22:30",
            "PRECINCT": "WEST",
            "CASE #": "23-2016-18335",
        },
    )
    _assert_row_values(
        df.iloc[-1],
        {
            "OFFICER": "GIBSON, AARON",
            "DATE OF REPORT": "8/1/2019",
            "DATE OF FORCE": "Jun-24-2019",
            "TIME OF FORCE": "21:55",
            "PRECINCT": "WEST",
            "CASE #": "23-2019-10327",
        },
    )


def test_header_guides_vertical_only_extract_all_use_of_force_pages():
    pdf, page = _use_of_force_page()
    try:
        headers = page.find_all("text[y0=min()]")

        guides = Guides(page)
        guides.vertical.from_headers(headers)
        df = guides.extract_table(pdf.pages).to_df()
    finally:
        pdf.close()

    assert headers.extract_each_text()[:8] == [
        "OFFICER",
        "DATE OF REPORT",
        "DATE OF FORCE",
        "TIME OF FORCE",
        "PRECINCT",
        "CASE #",
        "SUBJECT PERSON",
        "SEX",
    ]
    assert df.shape == (970, 25)
    assert list(df.columns[:6]) == [
        "OFFICER",
        "DATE OF REPORT",
        "DATE OF FORCE",
        "TIME OF FORCE",
        "PRECINCT",
        "CASE #",
    ]
    _assert_row_values(
        df.iloc[3],
        {
            "OFFICER": "BATES, JUSTIN",
            "DATE OF REPORT": "12/13/2016",
            "DATE OF FORCE": "Dec-02-2016",
            "TIME OF FORCE": "22:30",
            "PRECINCT": "WEST",
            "CASE #": "23-2016-18335",
        },
    )
    _assert_row_values(
        df.iloc[-1],
        {
            "OFFICER": "GIBSON, AARON",
            "DATE OF REPORT": "8/1/2019",
            "DATE OF FORCE": "Jun-24-2019",
            "TIME OF FORCE": "21:55",
            "PRECINCT": "WEST",
            "CASE #": "23-2019-10327",
        },
    )


def test_header_and_row_anchor_helper_extracts_use_of_force_sample():
    pdf, page = _use_of_force_page()
    try:
        texts, header, row_anchors = _case_header_and_row_anchors(page)
        header_row = sorted(
            [text for text in texts if abs(_mid_y(text) - _mid_y(header)) <= 1.1 and _text(text)],
            key=lambda text: text.x0,
        )

        guides = Guides(page).from_headers_and_row_anchors(
            header_row,
            row_anchors[:2],
            header_anchor=header,
        )
        table = guides.extract_table(
            cell_extract="words",
            cell_overlap="center",
            header="first",
            include_outer_boundaries=True,
        )
        df = table.to_df().dropna(how="all")
    finally:
        pdf.close()

    assert df.shape == (2, 25)
    assert list(df.columns[:6]) == [
        "OFFICER",
        "DATE OF REPORT",
        "DATE OF FORCE",
        "TIME OF FORCE",
        "PRECINCT",
        "CASE #",
    ]
    _assert_row_values(
        df.iloc[0],
        {
            "OFFICER": "PFEIFER, TIM",
            "DATE OF REPORT": "12/2/2016",
            "DATE OF FORCE": "Dec-02-2016",
            "TIME OF FORCE": "22:30",
            "PRECINCT": "WEST",
            "CASE #": "23-2016-18335",
        },
    )


def test_page_extract_table_guided_wraps_header_and_row_anchor_pattern():
    pdf, page = _use_of_force_page()
    try:
        texts, header, row_anchors = _case_header_and_row_anchors(page)
        header_row = sorted(
            [text for text in texts if abs(_mid_y(text) - _mid_y(header)) <= 1.1 and _text(text)],
            key=lambda text: text.x0,
        )

        table = page.extract_table_guided(
            header_row,
            row_anchors[:2],
        )
        df = table.to_df().dropna(how="all")
    finally:
        pdf.close()

    assert df.shape == (2, 25)
    assert list(df.columns[:6]) == [
        "OFFICER",
        "DATE OF REPORT",
        "DATE OF FORCE",
        "TIME OF FORCE",
        "PRECINCT",
        "CASE #",
    ]
    _assert_row_values(
        df.iloc[0],
        {
            "OFFICER": "PFEIFER, TIM",
            "DATE OF REPORT": "12/2/2016",
            "DATE OF FORCE": "Dec-02-2016",
            "TIME OF FORCE": "22:30",
            "PRECINCT": "WEST",
            "CASE #": "23-2016-18335",
        },
    )


def test_page_extract_table_guided_infers_header_anchor_from_string_headers():
    pdf, page = _use_of_force_page()
    try:
        texts, header, row_anchors = _case_header_and_row_anchors(page)
        header_row = sorted(
            [text for text in texts if abs(_mid_y(text) - _mid_y(header)) <= 1.1 and _text(text)],
            key=lambda text: text.x0,
        )
        header_names = [_text(text) for text in header_row]

        table = page.extract_table_guided(
            header_names,
            row_anchors[:2],
        )
        df = table.to_df().dropna(how="all")
    finally:
        pdf.close()

    assert df.shape == (2, 25)
    assert list(df.columns[:6]) == [
        "OFFICER",
        "DATE OF REPORT",
        "DATE OF FORCE",
        "TIME OF FORCE",
        "PRECINCT",
        "CASE #",
    ]
    _assert_row_values(
        df.iloc[0],
        {
            "OFFICER": "PFEIFER, TIM",
            "DATE OF REPORT": "12/2/2016",
            "DATE OF FORCE": "Dec-02-2016",
            "TIME OF FORCE": "22:30",
            "PRECINCT": "WEST",
            "CASE #": "23-2016-18335",
        },
    )
