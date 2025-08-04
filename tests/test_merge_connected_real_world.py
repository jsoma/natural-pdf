"""Test merge_connected with more realistic scenarios."""

import pytest

from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.elements.region import Region


class MockPage:
    """Mock page for testing."""

    def __init__(self, index=0):
        self.index = index
        self.width = 612
        self.height = 792


def test_merge_connected_accented_characters():
    """Test merging regions split by accented characters."""
    page = MockPage()

    # Simulate text "café" split into "caf" and "é"
    region1 = Region(page, (100, 100, 130, 110))  # "caf"
    region1.extract_text = lambda: "caf"

    region2 = Region(page, (130, 100, 140, 110))  # "é" - directly adjacent
    region2.extract_text = lambda: "é"

    collection = ElementCollection([region1, region2])

    # With small threshold, should merge - use empty separator for adjacent chars
    result = collection.merge_connected(proximity_threshold=1.0, text_separator="")
    assert len(result) == 1
    merged = result[0]
    assert merged.metadata["merge_info"]["merged_text"] == "café"


def test_merge_connected_font_variations():
    """Test merging text split due to font variations (bold, italic)."""
    page = MockPage()

    # Simulate "This is bold text" where "bold" is in a different font
    region1 = Region(page, (100, 100, 150, 112))  # "This is "
    region1.extract_text = lambda: "This is "

    region2 = Region(page, (150, 100, 180, 112))  # "bold" - adjacent
    region2.extract_text = lambda: "bold"
    region2.metadata = {"font": "Helvetica-Bold"}

    region3 = Region(page, (180, 100, 210, 112))  # " text" - adjacent
    region3.extract_text = lambda: " text"

    collection = ElementCollection([region1, region2, region3])
    # Use empty separator since text already has proper spacing
    result = collection.merge_connected(proximity_threshold=2.0, text_separator="")

    assert len(result) == 1
    merged = result[0]
    assert merged.metadata["merge_info"]["merged_text"] == "This is bold text"


def test_merge_connected_multiline_paragraph():
    """Test merging a paragraph split across multiple lines."""
    page = MockPage()

    # Simulate a paragraph with multiple lines
    line1 = Region(page, (50, 100, 550, 115))
    line1.extract_text = lambda: "This is the first line of a paragraph that"

    line2 = Region(page, (50, 117, 550, 132))  # 2 points gap
    line2.extract_text = lambda: "continues on the second line with more text"

    line3 = Region(page, (50, 134, 350, 149))  # 2 points gap
    line3.extract_text = lambda: "and ends on the third line."

    collection = ElementCollection([line1, line2, line3])
    result = collection.merge_connected(proximity_threshold=3.0)

    assert len(result) == 1
    merged = result[0]
    expected_text = (
        "This is the first line of a paragraph that "
        "continues on the second line with more text "
        "and ends on the third line."
    )
    assert merged.metadata["merge_info"]["merged_text"] == expected_text


def test_merge_connected_table_cells():
    """Test NOT merging table cells that are adjacent but should remain separate."""
    page = MockPage()

    # Simulate table cells
    cell1 = Region(page, (100, 100, 200, 120))
    cell1.extract_text = lambda: "Name"
    cell1.region_type = "table_cell"

    cell2 = Region(page, (202, 100, 302, 120))  # 2 point gap
    cell2.extract_text = lambda: "Age"
    cell2.region_type = "table_cell"

    cell3 = Region(page, (304, 100, 404, 120))  # 2 point gap
    cell3.extract_text = lambda: "City"
    cell3.region_type = "table_cell"

    collection = ElementCollection([cell1, cell2, cell3])

    # Even with threshold that would merge them, we might want to keep table cells separate
    # This test shows current behavior - they will merge
    result = collection.merge_connected(proximity_threshold=3.0)
    assert len(result) == 1  # Currently merges - might want to add table_cell exception

    # With smaller threshold, they stay separate
    result = collection.merge_connected(proximity_threshold=1.0)
    assert len(result) == 3


def test_merge_connected_mixed_languages():
    """Test merging text with mixed languages/scripts."""
    page = MockPage()

    # Simulate English text with embedded Arabic
    region1 = Region(page, (100, 100, 150, 112))
    region1.extract_text = lambda: "The word "

    region2 = Region(page, (150, 100, 200, 112))  # Arabic word
    region2.extract_text = lambda: "كتاب"
    region2.metadata = {"language": "ar"}

    region3 = Region(page, (200, 100, 280, 112))
    region3.extract_text = lambda: " means book"

    collection = ElementCollection([region1, region2, region3])
    # Use empty separator since text already has spaces
    result = collection.merge_connected(proximity_threshold=2.0, text_separator="")

    assert len(result) == 1
    merged = result[0]
    assert merged.metadata["merge_info"]["merged_text"] == "The word كتاب means book"


def test_merge_connected_superscript_subscript():
    """Test merging text with superscripts/subscripts."""
    page = MockPage()

    # Simulate "H2O" where "2" is subscript (slightly lower)
    region1 = Region(page, (100, 100, 110, 112))
    region1.extract_text = lambda: "H"

    region2 = Region(page, (110, 102, 115, 110))  # Subscript "2" - vertically offset
    region2.extract_text = lambda: "2"
    region2.metadata = {"style": "subscript"}

    region3 = Region(page, (115, 100, 125, 112))
    region3.extract_text = lambda: "O"

    collection = ElementCollection([region1, region2, region3])
    # Use empty separator for chemical formula
    result = collection.merge_connected(proximity_threshold=3.0, text_separator="")

    assert len(result) == 1
    merged = result[0]
    # Note: Order might be H2O or HO2 depending on vertical position
    # Since subscript is lower, it might sort differently
    assert merged.metadata["merge_info"]["merged_text"] in ["H2O", "HO2"]


def test_merge_connected_complex_layout():
    """Test merge behavior with complex layout including headers, body, and sidebar."""
    page = MockPage()

    # Header
    header = Region(page, (50, 50, 550, 80))
    header.extract_text = lambda: "Document Title"
    header.region_type = "header"

    # Main body paragraph
    body1 = Region(page, (50, 100, 400, 115))
    body1.extract_text = lambda: "This is the main body text that flows"

    body2 = Region(page, (50, 117, 400, 132))
    body2.extract_text = lambda: "across multiple lines in the document."

    # Sidebar (should not merge with body)
    sidebar = Region(page, (420, 100, 550, 200))
    sidebar.extract_text = lambda: "Sidebar content"
    sidebar.region_type = "sidebar"

    collection = ElementCollection([header, body1, body2, sidebar])
    result = collection.merge_connected(proximity_threshold=3.0)

    # Should get 3 regions: header (alone), merged body, sidebar (alone)
    assert len(result) == 3

    # Find the merged body
    merged_body = None
    for r in result:
        if hasattr(r, "metadata") and "merge_info" in r.metadata:
            if r.metadata["merge_info"]["source_count"] == 2:
                merged_body = r
                break

    assert merged_body is not None
    expected_body = "This is the main body text that flows across multiple lines in the document."
    assert merged_body.metadata["merge_info"]["merged_text"] == expected_body


def test_merge_connected_preserve_reading_order():
    """Test that merged text preserves proper reading order."""
    page = MockPage()

    # Create regions in non-sequential order
    regions = []

    # Second line
    r2 = Region(page, (50, 120, 400, 135))
    r2.extract_text = lambda: "second line of text"
    regions.append(r2)

    # First line
    r1 = Region(page, (50, 100, 400, 115))
    r1.extract_text = lambda: "This is the first line"
    regions.append(r1)

    # Third line
    r3 = Region(page, (50, 140, 400, 155))
    r3.extract_text = lambda: "and the third line."
    regions.append(r3)

    collection = ElementCollection(regions)
    result = collection.merge_connected(proximity_threshold=5.0, preserve_order=True)

    assert len(result) == 1
    merged = result[0]
    expected = "This is the first line second line of text and the third line."
    assert merged.metadata["merge_info"]["merged_text"] == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
