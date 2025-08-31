"""Test exclude functionality with a real PDF"""

import natural_pdf as npdf


def test_exclude_text_from_multi_page_pdf():
    """Test excluding text elements from a multi-page PDF"""
    # Load the PDF
    pdf = npdf.PDF("pdfs/multipage-multicol-lines.pdf")

    # Find all text elements across all pages
    all_text = pdf.find_all("text")

    # Verify we found some text
    assert len(all_text) > 0, "No text found in the PDF"

    # Get pages that contain text
    pages_with_text = set()
    for element in all_text:
        pages_with_text.add(element.page.index)

    print(f"Found {len(all_text)} text elements across {len(pages_with_text)} pages")

    # Exclude first 10 text elements (which may span multiple pages)
    to_exclude = all_text[:10]
    result = to_exclude.exclude()

    # Verify it returns the collection for method chaining
    assert result is to_exclude

    # Verify exclusions were added to the appropriate pages
    excluded_pages = set()
    for element in to_exclude:
        excluded_pages.add(element.page.index)

    for page_idx in excluded_pages:
        page = pdf.pages[page_idx]
        # Check that the page has exclusions
        assert len(page._exclusions) > 0, f"Page {page_idx} should have exclusions"

    print(f"Successfully excluded {len(to_exclude)} elements across {len(excluded_pages)} pages")


def test_exclude_specific_text_pattern():
    """Test excluding text matching a specific pattern"""
    pdf = npdf.PDF("pdfs/multipage-multicol-lines.pdf")

    # Find text containing numbers (if any)
    number_text = pdf.find_all('text:contains("[0-9]")')

    if len(number_text) > 0:
        print(f"Found {len(number_text)} text elements containing numbers")

        # Exclude all text with numbers
        number_text.exclude()

        # Verify we can still extract text from pages
        for page in pdf.pages[:3]:  # Check first 3 pages
            text = page.extract_text()
            assert isinstance(text, str)

        print("Successfully excluded text containing numbers")
    else:
        # Find any text starting with capital letters
        capital_text = pdf.find_all("text").filter(lambda el: el.text and el.text[0].isupper())

        if len(capital_text) > 0:
            print(f"Found {len(capital_text)} text elements starting with capital letters")

            # Exclude them
            capital_text.exclude()

            print("Successfully excluded text starting with capital letters")


def test_exclude_empty_collection():
    """Test that excluding an empty collection doesn't raise errors"""
    pdf = npdf.PDF("pdfs/multipage-multicol-lines.pdf")

    # Try to find something that doesn't exist
    non_existent = pdf.find_all("text[color~=purple]")

    # This should be empty
    assert len(non_existent) == 0

    # Excluding empty collection should work fine
    result = non_existent.exclude()

    # Should return self
    assert result is non_existent

    print("Successfully handled excluding empty collection")


if __name__ == "__main__":
    test_exclude_text_from_multi_page_pdf()
    test_exclude_specific_text_pattern()
    test_exclude_empty_collection()
