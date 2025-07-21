#!/usr/bin/env python3
"""Test pdf.add_exclusion() when lambda returns ElementCollection."""

from pathlib import Path

import pytest

from natural_pdf import PDF


def find_test_pdf():
    """Find a test PDF."""
    project_root = Path(__file__).parent.parent
    pdf_path = project_root / "pdfs/practice.pdf"
    if not pdf_path.exists():
        # Try alternative location
        pdf_path = (
            project_root
            / "bad-pdfs/submissions/Doc 06 - Approved Expenses 07.01.2022-06.30.2023 Marketplace Transactions - REDACTED.pdf"
        )
    return pdf_path if pdf_path.exists() else None


@pytest.mark.skipif(find_test_pdf() is None, reason="No test PDF file found")
def test_pdf_add_exclusion_with_element_list_lambda():
    """Test that pdf.add_exclusion() works when lambda returns a list of elements."""

    pdf_path = find_test_pdf()
    pdf = PDF(pdf_path)
    page = pdf[0]

    # Find some text elements to use
    text_elements = page.find_all("text")
    if len(text_elements) < 3:
        pytest.skip("Not enough text elements for test")

    # Add exclusion using lambda that returns a Python list of elements
    pdf.add_exclusion(
        lambda page: list(page.find_all("text")[:2]),  # Returns Python list
        label="first_two_elements",
    )

    # Check that exclusions were applied
    with_exclusions = page.find_all("text", apply_exclusions=True)
    without_exclusions = page.find_all("text", apply_exclusions=False)

    assert len(with_exclusions) < len(
        without_exclusions
    ), "List exclusion from lambda should reduce element count"


@pytest.mark.skipif(find_test_pdf() is None, reason="No test PDF file found")
def test_pdf_add_exclusion_with_elementcollection_lambda():
    """Test that pdf.add_exclusion() works when lambda returns ElementCollection."""

    pdf_path = find_test_pdf()
    pdf = PDF(pdf_path)

    # Access first page to initialize it
    page = pdf[0]

    # Find some text elements to use
    text_elements = page.find_all("text")
    if len(text_elements) < 3:
        pytest.skip("Not enough text elements for test")

    # Get initial text count for comparison
    initial_text_count = len(page.find_all("text"))
    print(f"Initial text elements: {initial_text_count}")

    # Add exclusion using lambda that returns ElementCollection
    # This should work but currently doesn't
    pdf.add_exclusion(
        lambda page: page.find_all("text").filter(lambda elem: len(elem.text.strip()) > 20)[:2],
        label="long_text_exclusion",
    )

    # Check that exclusions were actually applied
    # We need to test this by checking if text extraction respects the exclusion
    remaining_text_elements = page.find_all("text", apply_exclusions=True)
    excluded_text_elements = page.find_all("text", apply_exclusions=False)

    print(f"With exclusions: {len(remaining_text_elements)}")
    print(f"Without exclusions: {len(excluded_text_elements)}")

    # The exclusion should have been applied, so we should have fewer elements
    # when apply_exclusions=True compared to apply_exclusions=False
    assert len(remaining_text_elements) < len(
        excluded_text_elements
    ), "ElementCollection exclusion from lambda should reduce element count"


@pytest.mark.skipif(find_test_pdf() is None, reason="No test PDF file found")
def test_page_add_exclusion_with_elementcollection_lambda():
    """Test that page.add_exclusion() works when lambda returns ElementCollection."""

    pdf_path = find_test_pdf()
    pdf = PDF(pdf_path)
    page = pdf[0]

    # Find some text elements
    text_elements = page.find_all("text")
    if len(text_elements) < 2:
        pytest.skip("Not enough text elements for test")

    # Add exclusion using lambda that returns ElementCollection
    # This might already work at the page level
    page.add_exclusion(
        lambda p: p.find_all("text").filter(lambda elem: "test" in elem.text.lower())[:1],
        label="test_text_exclusion",
    )

    # Test that exclusion is applied
    with_exclusions = page.find_all("text", apply_exclusions=True)
    without_exclusions = page.find_all("text", apply_exclusions=False)

    print(f"Page level - With exclusions: {len(with_exclusions)}")
    print(f"Page level - Without exclusions: {len(without_exclusions)}")

    # Should have same or fewer elements with exclusions applied
    assert len(with_exclusions) <= len(without_exclusions)


if __name__ == "__main__":
    # Run simple tests
    pdf_path = find_test_pdf()
    if pdf_path:
        print(f"Testing with PDF: {pdf_path}")

        pdf = PDF(pdf_path)
        page = pdf[0]

        # Test lambda that returns ElementCollection
        print("\n1. Testing PDF level add_exclusion with ElementCollection lambda...")
        try:
            pdf.add_exclusion(
                lambda page: page.find_all("text")[:2],  # Returns ElementCollection
                label="first_two_elements",
            )
            print("   ✅ pdf.add_exclusion() accepted ElementCollection lambda")
        except Exception as e:
            print(f"   ❌ pdf.add_exclusion() failed: {e}")

        print("\n2. Testing page level add_exclusion with ElementCollection lambda...")
        try:
            page.add_exclusion(
                lambda p: p.find_all("text")[:1], label="first_element"  # Returns ElementCollection
            )
            print("   ✅ page.add_exclusion() accepted ElementCollection lambda")
        except Exception as e:
            print(f"   ❌ page.add_exclusion() failed: {e}")

        # Test if exclusions actually work
        print("\n3. Testing exclusion application...")
        with_exclusions = page.find_all("text", apply_exclusions=True)
        without_exclusions = page.find_all("text", apply_exclusions=False)

        print(f"   With exclusions applied: {len(with_exclusions)} elements")
        print(f"   Without exclusions applied: {len(without_exclusions)} elements")

        if len(with_exclusions) < len(without_exclusions):
            print("   ✅ Exclusions are being applied!")
        else:
            print("   ⚠️  Exclusions may not be working as expected")

    else:
        print("❌ No test PDF found")

    # Run pytest
    pytest.main([__file__, "-v"])
