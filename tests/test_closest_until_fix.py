import pytest

from natural_pdf import PDF


def test_closest_preserves_quality_ordering():
    """Test that until with :closest preserves quality-based ordering"""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    # Start from the top of the page
    start = page.find("text")
    print(f"\nStarting from: '{start.extract_text()}' at y={start.top}")

    # Search for something that will have multiple matches with varying quality
    # Let's search for "meat" which should match "Meatpacking" perfectly
    search_term = "meat"

    # First, see all matches with their quality
    all_matches = page.find_all(f"text:closest({search_term}@0.3)")
    print(f"\n=== All matches for '{search_term}' with threshold 0.3 ===")

    # Calculate actual similarities
    from difflib import SequenceMatcher

    matches_with_scores = []
    for match in all_matches:
        text = match.extract_text()
        similarity = SequenceMatcher(None, search_term.lower(), text.lower()).ratio()
        matches_with_scores.append((match, text, similarity))

    # Sort by similarity to show quality ordering
    matches_with_scores.sort(key=lambda x: x[2], reverse=True)

    for i, (match, text, sim) in enumerate(matches_with_scores[:10]):
        print(f"{i}: '{text}' at y={match.top} (similarity: {sim:.3f})")

    # Find the best match that's below our starting point
    below_matches = [(m, t, s) for m, t, s in matches_with_scores if m.top > start.bottom]
    if below_matches:
        best_below = below_matches[0]
        print(
            f"\nBest match below start: '{best_below[1]}' at y={best_below[0].top} (similarity: {best_below[2]:.3f})"
        )

    # Now test with .below()
    print(f"\n=== Testing .below(until='text:closest({search_term}@0.3)') ===")
    result = start.below(until=f"text:closest({search_term}@0.3)")
    print(f"Region ends at y={result.bottom}")

    # Find what's at the boundary
    boundary_elems = page.find_all("text").filter(lambda e: abs(e.top - result.bottom) < 1)
    if boundary_elems:
        boundary_text = boundary_elems[0].extract_text()
        boundary_sim = SequenceMatcher(None, search_term.lower(), boundary_text.lower()).ratio()
        print(f"Boundary element: '{boundary_text}' (similarity: {boundary_sim:.3f})")

        # Check if this is the best match
        if below_matches and abs(best_below[0].top - result.bottom) > 1:
            print(f"\nWARNING: Not using best match!")
            print(f"  Best match: '{best_below[1]}' at y={best_below[0].top}")
            print(f"  Used: '{boundary_text}' at y={boundary_elems[0].top}")


def test_specific_use_case():
    """Test the specific use case from the issue"""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    # Find "Name" as mentioned in the issue
    name_elem = page.find("text:contains(Name)")
    if not name_elem:
        print("Could not find 'Name' element")
        return

    print(f"\n=== Testing specific use case ===")
    print(f"Starting from: '{name_elem.extract_text()}' at y={name_elem.top}")

    # Look for ICE Information or similar
    search_terms = ["ICE Information", "Information", "ICE"]

    for search_term in search_terms:
        print(f"\n--- Searching for '{search_term}' ---")

        # See what matches we get
        matches = page.find_all(f"text:closest({search_term}@0.0)")
        if len(matches) > 0:
            print(f"Found {len(matches)} matches")
            # Show matches below our starting point
            below = [m for m in matches if m.top > name_elem.bottom]
            for i, match in enumerate(below[:5]):
                from difflib import SequenceMatcher

                sim = SequenceMatcher(
                    None, search_term.lower(), match.extract_text().lower()
                ).ratio()
                print(f"  {i}: '{match.extract_text()}' at y={match.top} (sim: {sim:.3f})")

            if below:
                # Test with below()
                result = name_elem.below(until=f"text:closest({search_term}@0.0)")
                print(f"Region ends at y={result.bottom}")

                # What's at the boundary?
                boundary = page.find_all("text").filter(lambda e: abs(e.top - result.bottom) < 1)
                if boundary:
                    print(f"Boundary: '{boundary[0].extract_text()}'")


if __name__ == "__main__":
    test_closest_preserves_quality_ordering()
    test_specific_use_case()
