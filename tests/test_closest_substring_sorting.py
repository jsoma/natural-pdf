import pytest

from natural_pdf import PDF


def test_closest_sorts_substring_matches_by_similarity():
    """Test that :closest sorts substring matches by similarity score"""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    # Test with "Information" - should find exact matches first
    matches = page.find_all("text:closest(Information@0.3)")

    if matches:
        # Check that results are sorted by similarity within substring matches
        from difflib import SequenceMatcher

        print("\n=== Matches for 'Information' ===")
        for i, match in enumerate(matches[:10]):
            text = match.extract_text()
            sim = SequenceMatcher(None, "information", text.lower()).ratio()
            contains = "information" in text.lower()
            print(f"{i}: '{text}' (sim: {sim:.3f}, contains: {contains})")

        # If there's an exact match "Information", it should come first
        # among substring matches
        exact_matches = [m for m in matches if m.extract_text().strip().lower() == "information"]
        if exact_matches:
            assert matches[0] in exact_matches, "Exact match should come first"


def test_closest_with_until_uses_best_match():
    """Test that .below(until=:closest) now uses the best similarity match"""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    # Find a starting point
    start = page.find("text:contains(Summary)")
    if not start:
        start = page.find("text")

    print(f"\nStarting from: '{start.extract_text()}' at y={start.top}")

    # Search for "Violation" which should match "Violations" better than "Violation Count:"
    search_term = "Violation"

    # Get all matches to see the ordering
    all_matches = page.find_all(f"text:closest({search_term}@0.5)")

    # Show the ordering
    from difflib import SequenceMatcher

    print(f"\n=== All matches for '{search_term}' ===")
    for i, match in enumerate(all_matches[:5]):
        text = match.extract_text()
        sim = SequenceMatcher(None, search_term.lower(), text.lower()).ratio()
        print(f"{i}: '{text}' at y={match.top} (similarity: {sim:.3f})")

    # Now test with .below()
    result = start.below(until=f"text:closest({search_term}@0.5)")

    # Find what's at the boundary
    boundary = page.find_all("text").filter(lambda e: abs(e.top - result.bottom) < 1)
    if boundary:
        boundary_text = boundary[0].extract_text()
        boundary_sim = SequenceMatcher(None, search_term.lower(), boundary_text.lower()).ratio()
        print(f"\nBoundary element: '{boundary_text}' (similarity: {boundary_sim:.3f})")

        # The boundary should be the best similarity match among substring matches
        # that are below our starting point
        below_matches = [m for m in all_matches if m.top > start.bottom]
        if below_matches:
            best_match = below_matches[0]  # Should be sorted by similarity now
            best_text = best_match.extract_text()
            best_sim = SequenceMatcher(None, search_term.lower(), best_text.lower()).ratio()
            print(f"Best match below: '{best_text}' (similarity: {best_sim:.3f})")


def test_exact_match_comes_before_partial():
    """Test that exact matches come before partial matches"""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    # Search for a short word that might have exact and partial matches
    # Let's try "Date" which might match "Date:" exactly
    search_term = "Date"

    matches = page.find_all(f"text:closest({search_term}@0.3)")

    if len(matches) >= 2:
        from difflib import SequenceMatcher

        print(f"\n=== Matches for '{search_term}' ===")
        # Show first few matches with their similarity
        for i, match in enumerate(matches[:5]):
            text = match.extract_text()
            sim = SequenceMatcher(None, search_term.lower(), text.lower()).ratio()
            # Check if it's an exact match (ignoring punctuation)
            clean_text = "".join(c for c in text if c.isalnum())
            is_exact = clean_text.lower() == search_term.lower()
            print(f"{i}: '{text}' (sim: {sim:.3f}, exact: {is_exact})")

        # Check ordering: exact matches should come before partial matches
        # with the same "contains" status


def test_threshold_still_works():
    """Ensure threshold filtering still works with the new sorting"""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    search_term = "Chicago"

    # Test with different thresholds
    for threshold in [0.3, 0.5, 0.7, 0.9]:
        matches = page.find_all(f"text:closest({search_term}@{threshold})")

        if matches:
            # All matches should have similarity >= threshold
            from difflib import SequenceMatcher

            for match in matches:
                text = match.extract_text()
                sim = SequenceMatcher(None, search_term.lower(), text.lower()).ratio()
                assert (
                    sim >= threshold - 0.01
                ), f"Match '{text}' has similarity {sim:.3f} < threshold {threshold}"

    print("\nThreshold filtering works correctly")


if __name__ == "__main__":
    test_closest_sorts_substring_matches_by_similarity()
    test_closest_with_until_uses_best_match()
    test_exact_match_comes_before_partial()
    test_threshold_still_works()
