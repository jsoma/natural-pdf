#!/usr/bin/env python3
"""Test the exclusion recursion fix."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import natural_pdf as npdf


def test_exclusion_with_find():
    """Test that exclusions using find() don't cause infinite recursion."""
    # Create a simple test PDF
    pdf = npdf.PDF("pdfs/sections.pdf")
    page = pdf.pages[0]

    # This should NOT cause infinite recursion anymore
    # Use safer lambdas that handle None returns
    page.add_exclusion(
        lambda p: (
            p.find("text:contains('Section')").above()
            if p.find("text:contains('Section')")
            else None
        )
    )
    page.add_exclusion(lambda p: p.find("text").expand() if p.find("text") else None)

    # Try to extract text - this should work without recursion
    text = page.extract_text()
    print(f"Successfully extracted {len(text)} characters")

    # Try finding elements - this should also work
    elements = page.find_all("text")
    print(f"Found {len(elements)} text elements after applying exclusions")

    # Test with ElementCollection return
    page.add_exclusion(lambda p: p.find_all("text:contains('Header')"))
    text2 = page.extract_text()
    print(f"Successfully extracted {len(text2)} characters with ElementCollection exclusion")

    print("✅ All tests passed - no infinite recursion!")


if __name__ == "__main__":
    test_exclusion_with_find()
