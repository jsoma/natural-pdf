import pytest

from natural_pdf import PDF


def test_compare_before_after_fix():
    """Compare behavior with and without the fix"""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    # Find a good starting point
    start = page.find("text:contains(Jungle)")
    if not start:
        start = page.find("text")

    print(f"\nStarting from: '{start.extract_text()}' at y={start.top}")

    # Use a search term that will have clear quality differences
    # Let's search for "Violation" which should match "Violations" well
    search_term = "Violation"

    # First, show all matches with quality scores
    all_matches = page.find_all(f"text:closest({search_term}@0.5)")
    print(f"\n=== All matches for '{search_term}' ===")

    from difflib import SequenceMatcher

    for i, match in enumerate(all_matches[:10]):
        text = match.extract_text()
        sim = SequenceMatcher(None, search_term.lower(), text.lower()).ratio()
        print(f"{i}: '{text}' at y={match.top} (similarity: {sim:.3f})")

    # Find matches below our starting point
    below_matches = [m for m in all_matches if m.top > start.bottom]
    print(f"\n=== Matches below starting point ===")
    for i, match in enumerate(below_matches[:5]):
        text = match.extract_text()
        sim = SequenceMatcher(None, search_term.lower(), text.lower()).ratio()
        print(f"{i}: '{text}' at y={match.top} (similarity: {sim:.3f})")

    # Test with .below()
    print(f"\n=== Testing .below(until='text:closest({search_term}@0.5)') ===")
    result = start.below(until=f"text:closest({search_term}@0.5)")
    print(f"Region ends at y={result.bottom}")

    # What's at the boundary?
    boundary = page.find_all("text").filter(lambda e: abs(e.top - result.bottom) < 1)
    if boundary:
        boundary_text = boundary[0].extract_text()
        boundary_sim = SequenceMatcher(None, search_term.lower(), boundary_text.lower()).ratio()
        print(
            f"Boundary element: '{boundary_text}' at y={boundary[0].top} (similarity: {boundary_sim:.3f})"
        )

        # Is this the best match or just the first positional match?
        if below_matches:
            first_positional = min(below_matches, key=lambda m: m.top)
            first_pos_text = first_positional.extract_text()
            first_pos_sim = SequenceMatcher(
                None, search_term.lower(), first_pos_text.lower()
            ).ratio()

            if first_positional.top != boundary[0].top:
                print(
                    f"\nFirst positional match: '{first_pos_text}' at y={first_positional.top} (similarity: {first_pos_sim:.3f})"
                )
                print("The boundary is NOT the first positional match - fix is working!")
            else:
                print(f"\nBoundary IS the first positional match")


def test_with_perfect_match():
    """Test with a search that has a perfect match"""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    start = page.find("text")
    print(f"\n=== Testing with perfect match ===")
    print(f"Starting from: '{start.extract_text()}' at y={start.top}")

    # Search for "Chicago" which should have an exact match
    search_term = "Chicago"

    matches = page.find_all(f"text:closest({search_term}@0.5)")
    print(f"\nMatches for '{search_term}':")

    from difflib import SequenceMatcher

    for i, match in enumerate(matches[:5]):
        text = match.extract_text()
        sim = SequenceMatcher(None, search_term.lower(), text.lower()).ratio()
        print(f"{i}: '{text}' at y={match.top} (similarity: {sim:.3f})")

    # Test with below
    result = start.below(until=f"text:closest({search_term}@0.5)")
    print(f"\nRegion ends at y={result.bottom}")

    # Check boundary
    boundary = page.find_all("text").filter(lambda e: abs(e.top - result.bottom) < 1)
    if boundary:
        print(f"Boundary: '{boundary[0].extract_text()}'")


if __name__ == "__main__":
    test_compare_before_after_fix()
    test_with_perfect_match()
