"""Test enhanced expand() method functionality."""


def test_expand_with_boolean_values(practice_pdf):
    """Test expanding to page edges with boolean True."""
    page = practice_pdf.pages[0]

    # Find an element in the middle of the page
    element = page.find('text:contains("Statute")')
    assert element is not None

    # Test expanding to full width
    expanded = element.expand(left=True, right=True)
    assert expanded.x0 == 0
    assert expanded.x1 == page.width
    assert expanded.top == element.top
    assert expanded.bottom == element.bottom

    # Test expanding to full height
    expanded = element.expand(top=True, bottom=True)
    assert expanded.x0 == element.x0
    assert expanded.x1 == element.x1
    assert expanded.top == 0
    assert expanded.bottom == page.height

    # Test expanding to all edges
    expanded = element.expand(left=True, right=True, top=True, bottom=True)
    assert expanded.x0 == 0
    assert expanded.x1 == page.width
    assert expanded.top == 0
    assert expanded.bottom == page.height


def test_expand_with_numeric_values(practice_pdf):
    """Test expanding with fixed pixel amounts."""
    page = practice_pdf.pages[0]

    element = page.find('text:contains("Statute")')
    assert element is not None

    # Test expanding by fixed amounts
    expanded = element.expand(left=10, right=20, top=5, bottom=15)
    assert expanded.x0 == element.x0 - 10
    assert expanded.x1 == element.x1 + 20
    assert expanded.top == element.top - 5
    assert expanded.bottom == element.bottom + 15

    # Test uniform expansion
    expanded = element.expand(25)
    assert expanded.x0 == element.x0 - 25
    assert expanded.x1 == element.x1 + 25
    assert expanded.top == element.top - 25
    assert expanded.bottom == element.bottom + 25


def test_expand_with_selectors(practice_pdf):
    """Test expanding until specific elements using selectors."""
    page = practice_pdf.pages[0]

    # Find "Statute" element
    statute = page.find('text:contains("Statute")')
    assert statute is not None

    # Find "Repeat?" element to the right
    repeat = page.find('text:contains("Repeat?")')
    assert repeat is not None
    assert repeat.x0 > statute.x1  # Verify it's to the right

    # Test expanding right until "Repeat?" (excluding)
    expanded = statute.expand(right='text:contains("Repeat?")')
    assert expanded.x0 == statute.x0
    assert expanded.x1 == repeat.x0 - 0.01  # Should stop just before "Repeat?" with default offset
    assert expanded.top == statute.top
    assert expanded.bottom == statute.bottom

    # Test expanding right until "Repeat?" (including)
    expanded_inclusive = statute.expand(right='+text:contains("Repeat?")')
    assert expanded_inclusive.x0 == statute.x0
    assert expanded_inclusive.x1 == repeat.x1  # Should include "Repeat?"
    assert expanded_inclusive.top == statute.top
    assert expanded_inclusive.bottom == statute.bottom


def test_expand_with_selectors_not_found(practice_pdf):
    """Test behavior when selector doesn't match any elements."""
    page = practice_pdf.pages[0]

    element = page.find('text:contains("Statute")')
    assert element is not None

    # Test with selector that won't match anything
    expanded = element.expand(right='text:contains("NonExistentText")')
    assert expanded.x0 == element.x0
    assert expanded.x1 == page.width  # Should expand to page edge when selector not found
    assert expanded.top == element.top
    assert expanded.bottom == element.bottom


def test_expand_mixed_parameters(practice_pdf):
    """Test combining different types of expansion parameters."""
    page = practice_pdf.pages[0]

    element = page.find('text:contains("Statute")')
    assert element is not None

    # Mix boolean, numeric, and selector
    expanded = element.expand(
        left=True,  # To page edge
        right='text:contains("Repeat?")',  # Until element
        top=10,  # Fixed pixels
        bottom=20,  # Fixed pixels
    )

    assert expanded.x0 == 0  # Left edge of page
    assert expanded.top == element.top - 10
    assert expanded.bottom == element.bottom + 20

    # The right edge should be at "Repeat?" if found
    repeat = page.find('text:contains("Repeat?")')
    if repeat and repeat.x0 > element.x1:
        assert expanded.x1 == repeat.x0 - 0.01  # With default offset


def test_expand_with_factors(practice_pdf):
    """Test expand with width and height factors."""
    page = practice_pdf.pages[0]

    element = page.find('text:contains("Statute")')
    assert element is not None

    # Test with factors after selector expansion
    expanded = element.expand(right='text:contains("Repeat?")', width_factor=0.8)

    # Verify the width was scaled
    repeat = page.find('text:contains("Repeat?")')
    if repeat and repeat.x0 > element.x1:
        expected_width = (repeat.x0 - element.x0) * 0.8
        actual_width = expanded.x1 - expanded.x0
        assert abs(actual_width - expected_width) < 1  # Allow small rounding differences


def test_expand_directional_filtering(practice_pdf):
    """Test that expand only considers elements in the correct direction."""
    page = practice_pdf.pages[0]

    # Find an element with text both above and below
    element = page.find('text:contains("Address")')
    if element is None:
        # Fallback to another common element
        element = page.find('text:contains("Date")')
    assert element is not None

    # Find all elements containing certain text
    all_text_elements = page.find_all("text")

    # Expand in different directions and verify it doesn't pick up elements from wrong direction
    expanded_right = element.expand(right="text")
    expanded_left = element.expand(left="text")

    # The expansions should be different if there are elements on both sides
    if expanded_right.x1 != element.x1 and expanded_left.x0 != element.x0:
        assert expanded_right.x1 != expanded_left.x0


def test_expand_zero_values(practice_pdf):
    """Test that zero values don't change the element."""
    page = practice_pdf.pages[0]

    element = page.find('text:contains("Statute")')
    assert element is not None

    # Expand with explicit zeros (default behavior)
    expanded = element.expand(left=0, right=0, top=0, bottom=0)
    assert expanded.x0 == element.x0
    assert expanded.x1 == element.x1
    assert expanded.top == element.top
    assert expanded.bottom == element.bottom


def test_expand_on_region(practice_pdf):
    """Test that expand works on regions as well as elements."""
    page = practice_pdf.pages[0]

    # Create a region
    element = page.find('text:contains("Statute")')
    region = element.expand(10)  # Create initial region

    # Now expand the region further
    expanded = region.expand(right=True, bottom=20)
    assert expanded.x0 == region.x0
    assert expanded.x1 == page.width
    assert expanded.top == region.top
    assert expanded.bottom == region.bottom + 20
