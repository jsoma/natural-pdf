import pytest

from natural_pdf import PDF


def test_closest_until_should_use_best_match():
    """Test that until='text:closest()' selects the best match, not just the first positional match"""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    # Let's find a starting point and look for text below it
    # We'll search for text that might have multiple matches with varying quality

    # First, let's see what's on the page
    print("\n=== Page content preview ===")
    for i, text in enumerate(page.find_all("text")[:20]):
        print(f"{i}: '{text.extract_text()}' at y={text.top}")

    # Find a starting element
    start = page.find("text:contains(Name)")
    if not start:
        # Try another starting point
        start = page.find("text")

    print(f"\n=== Starting from: '{start.extract_text()}' at y={start.top} ===")

    # Look for text below with fuzzy matching
    # Use a low threshold to see all potential matches
    search_term = "Durham"

    # First, let's see what matches we get with closest
    all_matches = page.find_all(f"text:closest({search_term}@0.5)")
    print(f"\n=== All matches for '{search_term}' with threshold 0.5 ===")
    for match in all_matches[:10]:
        print(f"  '{match.extract_text()}' at y={match.top}")

    # Now test the below() with until using closest
    print(f"\n=== Testing .below(until='text:closest({search_term}@0.5)') ===")

    # This should find the best match below, not just the first one
    result = start.below(until=f"text:closest({search_term}@0.5)")

    print(f"Result region: from y={result.top} to y={result.bottom}")

    # Let's see what text is at the boundary
    boundary_texts = page.find_all("text").filter(lambda t: abs(t.top - result.bottom) < 1)
    if boundary_texts:
        print(f"Text at boundary: '{boundary_texts[0].extract_text()}'")

    # Now let's verify if this is indeed the best match
    # Find all potential matches below our starting point
    below_matches = [
        m for m in all_matches if m.top < start.top
    ]  # Note: in PDF coords, lower y is below
    if below_matches:
        # The best match should be the first in all_matches that's also below
        best_match = below_matches[0]
        print(f"\nBest match below should be: '{best_match.extract_text()}' at y={best_match.top}")

        # Check if the region ends at the best match
        if abs(result.bottom - best_match.top) > 1:
            print(
                f"WARNING: Region ends at y={result.bottom}, but best match is at y={best_match.top}"
            )
            print("This suggests it's using positional proximity rather than match quality!")


def test_closest_until_with_threshold():
    """Test how threshold affects the until boundary selection"""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    start = page.find("text")
    search_term = "City"

    print(f"\n=== Testing different thresholds for '{search_term}' ===")

    # Test with different thresholds
    for threshold in [0.3, 0.5, 0.7, 0.9]:
        matches = page.find_all(f"text:closest({search_term}@{threshold})")
        print(f"\nThreshold {threshold}: {len(matches)} matches")
        if matches and len(matches) <= 5:
            for m in matches:
                print(f"  '{m.extract_text()}'")

        if matches:
            result = start.below(until=f"text:closest({search_term}@{threshold})")
            print(f"  Region ends at y={result.bottom}")


def test_closest_until_real_world_case():
    """Test the specific case mentioned in the issue"""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    # Look for something that might have "ICE Information" or similar
    print("\n=== Looking for text containing 'Information' ===")
    info_texts = page.find_all("text:contains(Information)")
    for text in info_texts:
        print(f"Found: '{text.extract_text()}' at y={text.top}")

    # Test with closest matching
    print("\n=== Testing with closest matching ===")
    start = page.find("text")
    if start:
        # Try to find something like "ICE Information" with fuzzy matching
        result = start.below(until="text:closest(Information@0.5)")
        print(f"Region from y={result.top} to y={result.bottom}")

        # Check what's at the boundary
        boundary = page.find_all("text").filter(lambda t: abs(t.top - result.bottom) < 1)
        if boundary:
            print(f"Boundary text: '{boundary[0].extract_text()}'")


if __name__ == "__main__":
    test_closest_until_should_use_best_match()
    test_closest_until_with_threshold()
    test_closest_until_real_world_case()
