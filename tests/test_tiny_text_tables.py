import os
from pathlib import Path

import pytest

from natural_pdf import PDF


def _get_test_pdf_path() -> str:
    """Return absolute path to the tiny-font sample PDF bundled with the repo."""
    root = Path(__file__).resolve().parents[1]
    pdf_path = root / "pdfs" / "tiny-text-tables.pdf"
    if not pdf_path.exists():
        pytest.skip("Test PDF 'tiny-text-tables.pdf' is not available.")
    return str(pdf_path)


# Column spacing needs a tighter x_tolerance as well.  Explore combos.
DIALS = [
    {"y_tolerance": 1, "x_tolerance": 0.5},
    {"y_tolerance": 1, "x_tolerance": 0.3},
]


@pytest.mark.parametrize("dial_config", DIALS, ids=[f"dial_{i}" for i, _ in enumerate(DIALS, 1)])
def test_tiny_font_text_extract(dial_config):
    """Ensure that tiny 2-pt text is extracted as words, not a single jumble string."""
    pdf_path = _get_test_pdf_path()

    pdf = PDF(pdf_path, text_tolerance=dial_config, auto_text_tolerance=False)

    page = pdf.pages[0]
    extracted = page.extract_text()

    assert extracted, "No text was extracted from the sample page."

    upper = extracted.upper()

    # Basic sanity: correct row appears.
    assert (
        "PFEIFER" in upper
    ), f"Expected 'PFEIFER' not found with dial {dial_config}. Extracted snippet: {extracted[:200]}"

    # Column gap check: we should see a space between the date formats in the first data row.
    first_row_snippet = extracted.split("\n")[1][:120]
    assert (
        "12/2/2016 " in extracted
    ), f"Column spacing missing with dial {dial_config}. Snippet: {first_row_snippet}"
