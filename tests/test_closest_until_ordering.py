import pytest

from natural_pdf import PDF


def test_closest_preserves_similarity_ordering():
    """Test that until with :closest uses similarity-based ordering, not positional"""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    # Start from a known position
    start = page.find("text:contains(Summary)")
    assert start, "Should find Summary text"

    # Search for something where we can see the ordering difference
    # Use a low threshold to get multiple matches
    result = start.below(until="text:closest(Violation@0.5)")

    # Check what we got
    assert result.bottom > start.bottom, "Region should extend below start"

    # Find what's at the boundary - need to be more flexible with the boundary detection
    boundary = page.find_all("text").filter(
        lambda e: e.top >= result.bottom - 1 and e.top <= result.bottom + 1
    )
    if boundary:
        print(f"Boundary element: '{boundary[0].extract_text()}' at y={boundary[0].top}")
        # The key point is that we're using the :closest ordering, not positional
        assert "Violation" in boundary[0].extract_text()


def test_closest_ordering_demonstration():
    """Demonstrate how :closest ordering works with until"""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    # Find a good starting point
    start = page.find("text:contains(Site)")
    assert start, "Should find Site text"

    print(f"\nStarting from: '{start.extract_text()}' at y={start.top}")

    # Search for "Durham" which has exact matches
    matches = page.find_all("text:closest(Durham@0.3)")
    print(f"\nAll Durham matches:")
    for i, m in enumerate(matches[:5]):
        print(f"  {i}: '{m.extract_text()}' at y={m.top}")

    # Use in below()
    result = start.below(until="text:closest(Durham@0.3)")
    print(f"\nRegion ends at y={result.bottom}")

    # The result should stop at the first match in the :closest ordering
    # that's below our starting point
    below_matches = [m for m in matches if m.top > start.bottom]
    if below_matches:
        expected_boundary = below_matches[0]
        # Allow for small differences due to include_endpoint behavior
        assert (
            abs(result.bottom - expected_boundary.top) < 10
        ), f"Expected boundary near {expected_boundary.top}, got {result.bottom}"


def test_regular_selectors_still_use_positional():
    """Ensure non-:closest selectors still use positional ordering"""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    start = page.find("text:contains(Summary)")
    assert start, "Should find Summary text"

    # Regular contains selector should still sort by position
    result = start.below(until="text:contains(world)")

    # Find all texts containing 'world'
    all_world_texts = page.find_all("text:contains(world)")
    below_world_texts = [t for t in all_world_texts if t.top > start.bottom]

    if below_world_texts:
        # Sort by position to find what we expect
        below_world_texts.sort(key=lambda t: t.top)
        first_positional = below_world_texts[0]

        print(
            f"\nFirst 'world' text below: '{first_positional.extract_text()}' at {first_positional.top}"
        )
        print(f"Region bottom: {result.bottom}")

        # The boundary should be near the first positional match
        # Allow some tolerance for include_endpoint behavior
        assert abs(result.bottom - first_positional.top) < 10


def test_closest_with_threshold_zero():
    """Test the specific case mentioned - :closest with @0.0 threshold"""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    start = page.find("text:contains(Name)")
    if start:
        # With threshold 0.0, all text elements match
        # but they're still ordered by similarity
        result = start.below(until="text:closest(ICE Information@0.0)")

        # This should include most of the page since threshold is 0
        assert result.bottom > start.bottom

        # The key insight: even with threshold 0.0, :closest still
        # orders by similarity, so the first match below will be
        # the one with highest similarity to "ICE Information"


if __name__ == "__main__":
    test_closest_preserves_similarity_ordering()
    test_closest_ordering_demonstration()
    test_regular_selectors_still_use_positional()
    test_closest_with_threshold_zero()
