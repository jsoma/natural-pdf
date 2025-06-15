import pytest


def test_find_all_text(practice_pdf):
    """Tests finding all text elements on a page."""
    page = practice_pdf.pages[0]

    # Find all text elements
    text_elements = page.find_all("text")

    # Assertions
    assert len(text_elements) > 0, "Should find text elements on the page"

    # Check element properties
    for element in text_elements:
        assert hasattr(element, "text"), "Text element should have 'text' attribute"
        assert element.text, "Text should not be empty"
        assert hasattr(element, "fontname"), "Text element should have 'fontname' attribute"
        assert hasattr(element, "size"), "Text element should have 'size' attribute"
        assert hasattr(element, "bbox"), "Text element should have 'bbox' property"
        assert len(element.bbox) == 4, "Bbox should have 4 coordinates (x0,y0,x1,y1)"


def test_find_text_with_content(practice_pdf):
    """Tests finding text elements containing specific content."""
    page = practice_pdf.pages[0]

    # Find text elements containing "Report"
    elements = page.find_all('text:contains("Jungle")')

    # Assertions
    assert len(elements) > 0, "Should find elements containing 'Jungle'"

    # Check that each found element contains the word
    for element in elements:
        assert "Jungle" in element.text, "Element text should contain 'Jungle'"


def test_find_all_lines(practice_pdf):
    """Tests finding line elements on a page."""
    page = practice_pdf.pages[0]

    # Find all line elements
    lines = page.find_all("line")

    # Assertions - not all PDFs have lines, so we can't assert length > 0
    for line in lines:
        assert hasattr(line, "bbox"), "Line should have 'bbox' attribute"
        assert len(line.bbox) == 4, "Line bbox should have 4 coordinates"
        assert hasattr(line, "type"), "Line should have 'type' attribute"


def test_find_horizontal_lines(atlanta_pdf):
    """Tests finding horizontal line elements."""
    page = atlanta_pdf.pages[0]

    # Find horizontal lines
    horizontal_lines = page.find_all("line:horizontal")

    # Assertions
    assert len(horizontal_lines) > 0, "Should find horizontal lines in Atlanta PDF"

    # Check that lines are actually horizontal (y0 ~= y1)
    for line in horizontal_lines:
        assert hasattr(line, "bbox")
        # Check that y coordinates are similar (horizontal line)
        assert (
            abs(line.bbox[1] - line.bbox[3]) < 1
        ), "Horizontal line should have similar y coordinates"
        # Check that x coordinates differ (has width)
        assert (
            abs(line.bbox[0] - line.bbox[2]) > 1
        ), "Horizontal line should have different x coordinates"


def test_find_vertical_lines(atlanta_pdf):
    """Tests finding vertical line elements."""
    page = atlanta_pdf.pages[0]

    # Find vertical lines
    vertical_lines = page.find_all("line:vertical")

    # No assertion on count as not all PDFs have vertical lines

    # Check that found lines are vertical (x0 ~= x1)
    for line in vertical_lines:
        assert hasattr(line, "bbox")
        assert line.type == "vertical", "Line should be vertical"
        # For vertical lines, x coordinates should be similar
        assert (
            abs(line.bbox[0] - line.bbox[2]) < 1
        ), "Vertical line should have similar x coordinates"


def test_find_rectangles(practice_pdf):
    """Tests finding rectangle elements."""
    page = practice_pdf.pages[0]

    # Find rectangles
    rectangles = page.find_all("rect")

    # Check rectangle properties
    for rect in rectangles:
        assert hasattr(rect, "bbox"), "Rectangle should have 'bbox' attribute"
        assert len(rect.bbox) == 4, "Rectangle bbox should have 4 coordinates"
        assert rect.bbox[2] > rect.bbox[0], "Rectangle width should be positive"
        assert rect.bbox[3] > rect.bbox[1], "Rectangle height should be positive"


def test_find_with_multiple_conditions(practice_pdf):
    """Tests finding elements with multiple selector conditions."""
    page = practice_pdf.pages[0]

    # Find text with specific font size
    large_text = page.find_all("text[size>=12]")

    # Assertions - font sizes vary by PDF, so we just check the filtering works
    for text in large_text:
        assert text.size >= 12, "Found text should have size >= 12"


def test_find_with_xpath_like_selector(practice_pdf):
    """Tests finding elements with more complex selectors."""
    page = practice_pdf.pages[0]

    # Find bold text containing "Report"
    # (Note: not all PDFs have 'bold' metadata, so this might find nothing)
    elements = page.find_all('text[bold=true]:contains("Report")')

    # Skip specific assertions if no elements found
    if elements:
        for element in elements:
            assert "Report" in element.text, "Element should contain 'Report'"
            assert getattr(element, "bold", False), "Element should be bold"


def test_element_collection_operations(practice_pdf):
    """Tests operations on ElementCollection objects."""
    page = practice_pdf.pages[0]

    # Get a collection
    all_text = page.find_all("text")

    # Test slicing
    first_three = all_text[:3]
    assert len(first_three) <= 3, "Sliced collection should have at most 3 elements"

    # Test filtering
    def is_short_text(elem):
        return len(elem.text) < 10

    short_texts = all_text.filter(is_short_text)

    for elem in short_texts:
        assert len(elem.text) < 10, "Filtered elements should have text length < 10"


def test_nearest_element(practice_pdf):
    """Tests finding the nearest element to another element."""
    page = practice_pdf.pages[0]

    # Find a reference element
    elements = page.find_all("text")

    # Skip test if not enough elements
    if len(elements) < 2:
        pytest.skip("Not enough text elements to test nearest functionality")

    # Get a reference element
    reference = elements[0]

    # Find nearest text element to the reference
    nearest = reference.nearest("text")

    assert nearest is not None, "Should find a nearest text element"
