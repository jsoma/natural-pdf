"""Test to reproduce the slicing exclusion issue."""

from pathlib import Path

import natural_pdf as npdf


def test_slice_exclusion_issue():
    """Test that exclusions are properly applied when accessing pages through slices."""
    # Use any existing test PDF
    test_pdfs = list(Path("tests/samples").glob("*.pdf"))
    if not test_pdfs:
        print("No test PDFs found, creating minimal example...")
        # Use user's example path
        pdf_path = "/var/folders/96/px7q37_d3l59nvr2mcytqn8r0000gn/T/tmpeahpfakw/tmpqd_3yj2a.pdf"
        if not Path(pdf_path).exists():
            print(f"Cannot find PDF at {pdf_path}")
            return
    else:
        pdf_path = test_pdfs[0]

    # Load PDF
    pdf = npdf.PDF(pdf_path)

    # Add exclusion lambda
    pdf.add_exclusion(lambda page: page.find_all('text:contains("2779 Records Found")'))

    print("\n=== Testing Exclusion Application ===\n")

    # Test 1: Direct access via pdf.pages[-1]
    print("Test 1: Direct access via pdf.pages[-1]")
    print(f"Exclusions exist: {bool(pdf._exclusions)}")
    text1 = pdf.pages[-1].extract_text()
    print(f"Text contains '2779 Records Found': {'2779 Records Found' in text1}")
    print(f"Full text: {repr(text1)}")

    # Test 2: Access via sliced collection
    print("\n\nTest 2: Access via sliced collection")
    pages = pdf.pages[:2]  # Get all pages via slice
    print(f"Type of pages: {type(pages)}")
    print(f"Type of pages.pages: {type(pages.pages)}")

    # Check the last page from slice
    last_page = pages[-1]
    print(f"\nLast page number: {last_page.number}")
    print(f"Last page has pdf reference: {hasattr(last_page, 'pdf')}")
    print(
        f"Last page.pdf._exclusions: {last_page.pdf._exclusions if hasattr(last_page, 'pdf') else 'N/A'}"
    )
    print(f"Last page._exclusions: {last_page._exclusions}")

    text2 = last_page.extract_text()
    print(f"\nText contains '2779 Records Found': {'2779 Records Found' in text2}")
    print(f"Full text: {repr(text2)}")

    # Test 3: Direct comparison
    print("\n\nTest 3: Direct comparison")
    print(f"Are they the same page object? {pdf.pages[-1] is pages[-1]}")
    print(f"Page index match? {pdf.pages[-1].index == pages[-1].index}")

    # Test 4: Check what happens with debug
    print("\n\nTest 4: Debug exclusion evaluation")
    text3 = pages[-1].extract_text(debug_exclusions=True)

    # Test 5: Check exclusion regions
    print("\n\nTest 5: Get exclusion regions")
    regions = pages[-1]._get_exclusion_regions(debug=True)
    print(f"Number of exclusion regions: {len(regions)}")

    # Assertions
    assert "2779 Records Found" not in text1, "Direct access should exclude '2779 Records Found'"
    assert (
        "2779 Records Found" not in text2
    ), "Sliced access should also exclude '2779 Records Found'"


if __name__ == "__main__":
    test_slice_exclusion_issue()
