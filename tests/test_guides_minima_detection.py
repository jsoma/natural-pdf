"""Test minima-based gap detection for snap_to_whitespace."""

import pytest

import natural_pdf as npdf
from natural_pdf import Guides


@pytest.fixture
def election_pdf():
    """Load the election results PDF."""
    pdf = npdf.PDF("pdfs/0500000US42001.pdf")
    yield pdf
    pdf.close()


def test_snap_to_whitespace_handles_variable_text_lengths(election_pdf):
    """
    Test that snap_to_whitespace correctly finds gaps even when one row
    extends much further than others (the KENYATTA problem).
    """
    page = election_pdf.pages[1]

    # KENYATTA extends to x=124, other rows end around x=89-101
    kenyatta = page.find("text:contains(KENYATTA)")
    assert kenyatta is not None
    assert kenyatta.x1 > 120  # Verify it extends far

    # Create table area including KENYATTA
    table_area = (
        kenyatta.below(until="text:contains(Contest Totals)", include_source=True).clip().trim()
    )

    guides = Guides(table_area)
    guides.vertical.divide(2)

    initial_guide = guides.vertical.data[1]  # Middle guide
    guides.vertical.snap_to_whitespace()
    snapped_guide = guides.vertical.data[1]

    # The guide should snap to the actual gap (after x=124, before x=185)
    # NOT stay in the text area or snap to a false gap within KENYATTA
    assert snapped_guide > 124, f"Guide should be after KENYATTA text (x=124), got {snapped_guide}"
    assert (
        snapped_guide < 185
    ), f"Guide should be before numbers column (x=185), got {snapped_guide}"


def test_snap_to_whitespace_finds_center_of_gap(election_pdf):
    """Test that snapped guides are centered in the gap."""
    page = election_pdf.pages[1]

    kenyatta = page.find("text:contains(KENYATTA)")
    table_area = (
        kenyatta.below(until="text:contains(Contest Totals)", include_source=True).clip().trim()
    )

    guides = Guides(table_area)
    guides.vertical.divide(2)
    guides.vertical.snap_to_whitespace()

    snapped_guide = guides.vertical.data[1]

    # Gap is roughly 124-185, center is around 154
    gap_center = (124 + 185) / 2
    assert (
        abs(snapped_guide - gap_center) < 10
    ), f"Guide should be near gap center (~{gap_center}), got {snapped_guide}"


def test_snap_to_whitespace_extracts_correct_table(election_pdf):
    """Test that snapped guides produce correct table extraction."""
    page = election_pdf.pages[1]

    kenyatta = page.find("text:contains(KENYATTA)")
    table_area = (
        kenyatta.below(until="text:contains(Contest Totals)", include_source=True).clip().trim()
    )

    guides = Guides(table_area)
    guides.vertical.divide(2)
    guides.vertical.snap_to_whitespace()
    guides.horizontal.from_lines(outer=True)

    df = guides.extract_table().to_df()

    # Check first row contains KENYATTA (not split incorrectly)
    first_row_text = str(df.columns[0])
    assert (
        "KENYATTA" in first_row_text
    ), f"First column header should contain KENYATTA, got: {first_row_text}"

    # Check numbers are in second column
    second_col = df.iloc[:, 1] if len(df.columns) > 1 else df.columns[1]
    # The header row should have a number (128 for KENYATTA)
    assert "128" in str(df.columns[1]) or any("128" in str(v) for v in df.iloc[:, 1].values)
