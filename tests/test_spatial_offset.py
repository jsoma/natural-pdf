"""Test spatial methods with small offset to ensure boundary elements are included."""

import pytest

from natural_pdf import PDF


def test_below_includes_close_text():
    """Test that below() includes text elements very close to the boundary.

    This was a real issue where text 1.5px below a boundary was excluded
    due to the 1px offset, causing missing rows in table extraction.
    """
    # This PDF has text very close to element boundaries
    pdf = PDF("https://www.pak-ks.org/desk/inc/media/EB62887E-EDF3-4CE4-B4D5-DEC69D53A2EF.pdf")
    page = pdf.pages[0]

    # Find the yellow header rectangle
    yellow_rect = page.find("rect[fill~=yellow]")
    assert yellow_rect is not None

    # Get region below
    region_below = yellow_rect.below()

    # The offset should be small (0.1px, not 1px)
    offset = region_below.top - yellow_rect.bottom
    assert offset < 0.2, f"Offset {offset} is too large, should be ~0.1"

    # Find text that's very close to the boundary
    text_nr01 = page.find("text:contains(nr.01)")
    assert text_nr01 is not None

    # Text should be in the region
    assert text_nr01.top >= region_below.top, "Text should be included in region below"

    # Extract table - should include the first row
    table = region_below.extract_table()
    assert table is not None

    # Convert to dataframe and check content
    df = table.to_df(header=None)
    assert len(df) > 0, "Table should have rows"

    # Check that first row contains "nr.01"
    first_row_text = " ".join(str(cell) for cell in df.iloc[0].values)
    assert "nr.01" in first_row_text, "First row should contain 'nr.01'"


def test_spatial_methods_offset():
    """Test that all spatial methods use small offset."""
    pdf = PDF("https://www.pak-ks.org/desk/inc/media/EB62887E-EDF3-4CE4-B4D5-DEC69D53A2EF.pdf")
    page = pdf.pages[0]

    # Find an element to test with
    elem = page.find("rect[fill~=yellow]")

    # Test all directions
    below_region = elem.below()
    above_region = elem.above()
    left_region = elem.left()
    right_region = elem.right()

    # All offsets should be small (0.1, not 1)
    assert abs(below_region.top - elem.bottom) < 0.2
    assert abs(elem.top - above_region.bottom) < 0.2
    assert abs(elem.x0 - left_region.x1) < 0.2
    assert abs(right_region.x0 - elem.x1) < 0.2


def test_table_extraction_completeness():
    """Test that table extraction includes all expected rows."""
    pdf = PDF("https://www.pak-ks.org/desk/inc/media/EB62887E-EDF3-4CE4-B4D5-DEC69D53A2EF.pdf")
    page = pdf.pages[0]

    # Extract table from region below header
    header = page.find("rect[fill~=yellow]")
    table = header.below().extract_table()
    df = table.to_df(header=None)

    # Should have all numbered items (1-8 based on the visible content)
    row_numbers = []
    for _, row in df.iterrows():
        first_cell = str(row.iloc[0])
        if first_cell.isdigit():
            row_numbers.append(int(first_cell))

    # Check we have consecutive numbers starting from 1
    assert 1 in row_numbers, "Should include row 1"
    assert min(row_numbers) == 1, "First numbered row should be 1"
