import pytest

from natural_pdf import PDF


def test_closest_until_debug():
    """Debug test to understand the closest/until behavior"""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    # Find a starting element
    start = page.find("text:contains(Name)")
    if not start:
        start = page.find("text")

    print(f"\n=== Starting from: '{start.extract_text()}' at y={start.top} ===")

    # Look for Durham matches
    search_term = "Durham"

    # Get all matches with closest
    all_matches = page.find_all(f"text:closest({search_term}@0.5)")
    print(f"\n=== All matches for '{search_term}' ===")
    for i, match in enumerate(all_matches):
        print(f"{i}: '{match.extract_text()}' at y={match.top}")

    # Now let's manually check what happens in the below() method
    # First, let's see which elements are considered "below" our starting point
    below_elements = [m for m in all_matches if m.top < start.top]  # PDF coords
    print(f"\n=== Elements below starting point ===")
    for i, elem in enumerate(below_elements):
        print(f"{i}: '{elem.extract_text()}' at y={elem.top}")

    # Test the actual below() call
    print(f"\n=== Testing .below(until='text:closest({search_term}@0.5)') ===")
    result = start.below(until=f"text:closest({search_term}@0.5)")
    print(f"Result region: from y={result.top} to y={result.bottom}")

    # Let's also test with a different search that might have more variation
    print("\n\n=== Testing with 'Information' search ===")
    info_matches = page.find_all("text:closest(Information@0.3)")
    print(f"Found {len(info_matches)} matches")
    for i, match in enumerate(info_matches[:5]):
        print(f"{i}: '{match.extract_text()}' at y={match.top}")

    if info_matches:
        result2 = start.below(until="text:closest(Information@0.3)")
        print(f"\nRegion ends at y={result2.bottom}")

        # Find what's at the boundary
        boundary = [m for m in info_matches if abs(m.top - result2.bottom) < 1]
        if boundary:
            print(f"Boundary element: '{boundary[0].extract_text()}'")


def test_closest_sorting():
    """Test that closest selector properly sorts by similarity"""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    # Test with a word that might have partial matches
    search_term = "Inspection"

    print(f"\n=== Testing closest sorting for '{search_term}' ===")

    # Get matches with different thresholds to see ordering
    for threshold in [0.3, 0.5, 0.7]:
        matches = page.find_all(f"text:closest({search_term}@{threshold})")
        print(f"\nThreshold {threshold}: {len(matches)} matches")
        for i, match in enumerate(matches[:5]):
            text = match.extract_text()
            # Calculate actual similarity to verify ordering
            from difflib import SequenceMatcher

            similarity = SequenceMatcher(None, search_term.lower(), text.lower()).ratio()
            print(f"  {i}: '{text}' (similarity: {similarity:.3f})")


if __name__ == "__main__":
    test_closest_until_debug()
    test_closest_sorting()
