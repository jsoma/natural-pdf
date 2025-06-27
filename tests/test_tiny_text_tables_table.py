import os
from pathlib import Path

import pytest

from natural_pdf import PDF


def _get_test_pdf_path() -> str:
    root = Path(__file__).resolve().parents[1]
    pdf_path = root / "pdfs" / "tiny-text-tables.pdf"
    if not pdf_path.exists():
        pytest.skip("tiny-text-tables.pdf not found – skipping table test")
    return str(pdf_path)


# Dial combos for pdfplumber table extraction
DIALS = [
    {
        "vertical_strategy": "text",
        "horizontal_strategy": "text",
        "snap_tolerance": 1,
        "join_tolerance": 1,
        # "min_words_vertical": 1,
        # "min_words_horizontal": 1,
    },
]


@pytest.mark.parametrize("table_settings", DIALS)
def test_tiny_font_table_extraction(table_settings):
    pdf_path = _get_test_pdf_path()
    pdf = PDF(pdf_path)  # auto text tolerance enabled
    page = pdf.pages[0]

    # Inject page-level text tolerances into settings so the extractor is consistent
    xt = page._config.get("x_tolerance")
    yt = page._config.get("y_tolerance")
    if xt is not None:
        table_settings = dict(table_settings, text_x_tolerance=xt)
    if yt is not None:
        table_settings = dict(table_settings, text_y_tolerance=yt)

    table = page.extract_table(method="pdfplumber", table_settings=table_settings)

    assert table, f"No table rows found with settings {table_settings}"
    # Expect at least 100 data rows in the tiny-text sample
    assert len(table) > 100, f"Too few rows ({len(table)}) extracted with {table_settings}"

    # Basic content sanity – look for the surname that appears in the first row
    flat_text = " ".join(" ".join(row) for row in table).upper()
    assert "PFEIFER" in flat_text, "Surname 'PFEIFER' missing in extracted table"
