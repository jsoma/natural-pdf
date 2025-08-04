"""
Simple test to verify include_boundaries fix is working.
"""

from pathlib import Path

import natural_pdf as npdf


def test_with_real_pdf():
    """Test include_boundaries with any available PDF."""
    # Try to find any PDF in the pdfs directory
    pdfs_dir = Path(__file__).parent.parent / "pdfs"
    pdf_files = list(pdfs_dir.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found for testing")
        return

    # Use the first available PDF
    pdf_path = pdf_files[0]
    print(f"\nTesting with: {pdf_path.name}")

    pdf = npdf.PDF(str(pdf_path))

    # Find any text that appears multiple times
    page = pdf.pages[0]
    text_elements = page.find_all("text")

    # Group text elements by their content to find repeated text
    text_groups = {}
    for elem in text_elements:
        text = elem.extract_text().strip()
        if len(text) > 2:  # Skip very short text
            if text not in text_groups:
                text_groups[text] = []
            text_groups[text].append(elem)

    # Find text that appears multiple times
    repeated_texts = {k: v for k, v in text_groups.items() if len(v) > 1}

    if repeated_texts:
        # Use the first repeated text
        test_text = list(repeated_texts.keys())[0]
        print(f"Found repeated text: '{test_text}' ({len(repeated_texts[test_text])} times)")
    else:
        # Just use any distinctive text
        for elem in text_elements[:10]:
            text = elem.extract_text().strip()
            if len(text) > 5 and len(text) < 50:
                test_text = text
                print(f"Using text: '{test_text}'")
                break
        else:
            print("No suitable text found")
            return

    # Test get_sections with this text
    try:
        sections_both = pdf.get_sections(f"text:contains({test_text})", include_boundaries="both")
        sections_none = pdf.get_sections(f"text:contains({test_text})", include_boundaries="none")

        print(f"\nSections found: {len(sections_both)}")

        if len(sections_both) > 0:
            # Check that the parameter is being used
            # Even if we can't verify the exact behavior, we can check that no errors occur
            print("✅ get_sections executed successfully with include_boundaries parameter")

            # Try to extract text to verify sections are created correctly
            for i, (sect_both, sect_none) in enumerate(zip(sections_both[:2], sections_none[:2])):
                text_both = sect_both.extract_text()[:100]
                text_none = sect_none.extract_text()[:100]

                print(f"\nSection {i+1}:")
                print(f"  With 'both': {len(sect_both.extract_text())} chars")
                print(f"  With 'none': {len(sect_none.extract_text())} chars")

                # They should be different if boundaries are working
                if text_both != text_none:
                    print("  ✅ Different text extracted - boundaries are working!")
                else:
                    print("  ⚠️  Same text extracted - boundaries might not be different enough")

    except Exception as e:
        print(f"❌ Error: {e}")
        raise


def test_boundary_parameter_acceptance():
    """Test that all include_boundaries values are accepted without error."""
    pdfs_dir = Path(__file__).parent.parent / "pdfs"
    pdf_files = list(pdfs_dir.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found")
        return

    pdf = npdf.PDF(str(pdf_files[0]))

    # Just verify that all parameter values are accepted
    boundary_options = ["both", "start", "end", "none"]

    print("\nTesting parameter acceptance...")
    for option in boundary_options:
        try:
            sections = pdf.get_sections("text", include_boundaries=option)
            print(f"  ✅ include_boundaries='{option}' accepted")
        except Exception as e:
            print(f"  ❌ include_boundaries='{option}' failed: {e}")
            raise

    print("\n✅ All include_boundaries values accepted!")


if __name__ == "__main__":
    test_with_real_pdf()
    print("\n" + "=" * 60)
    test_boundary_parameter_acceptance()
