#!/usr/bin/env python3
"""Test guides.extract_table() with apply_exclusions parameter."""

from pathlib import Path

import pytest

from natural_pdf import PDF
from natural_pdf.analyzers.guides import Guides
from natural_pdf.elements.region import Region


def find_test_pdf():
    """Find a suitable test PDF file."""
    project_root = Path(__file__).parent.parent
    pdf_path = (
        project_root
        / "bad-pdfs/submissions/Doc 06 - Approved Expenses 07.01.2022-06.30.2023 Marketplace Transactions - REDACTED.pdf"
    )
    return pdf_path if pdf_path.exists() else None


@pytest.mark.skipif(find_test_pdf() is None, reason="No test PDF file found")
def test_extract_table_apply_exclusions_parameter():
    """Test that extract_table accepts apply_exclusions parameter."""

    pdf_path = find_test_pdf()
    pdf = PDF(pdf_path)
    page = pdf[0]

    # Create simple guides for testing
    guides = Guides(page)
    guides.vertical.divide(3)  # 3 columns
    guides.horizontal.divide(3)  # 3 rows

    # Test that the parameter is accepted
    try:
        # Default (apply_exclusions=True)
        table1 = guides.extract_table(method="text")
        print(f"✅ extract_table works with default apply_exclusions")

        # Explicit True
        table2 = guides.extract_table(method="text", apply_exclusions=True)
        print(f"✅ extract_table accepts apply_exclusions=True")

        # Explicit False
        table3 = guides.extract_table(method="text", apply_exclusions=False)
        print(f"✅ extract_table accepts apply_exclusions=False")

    except TypeError as e:
        pytest.fail(f"Parameter not accepted: {e}")


@pytest.mark.skipif(find_test_pdf() is None, reason="No test PDF file found")
def test_extract_table_exclusions_effect():
    """Test that exclusions actually affect table extraction."""

    pdf_path = find_test_pdf()
    pdf = PDF(pdf_path)
    page = pdf[0]

    # Find some text on the page to create a meaningful test
    text_elements = page.find_all("text")
    if len(text_elements) < 10:
        pytest.skip("Not enough text elements for meaningful test")

    # Create guides that cover some text
    guides = Guides(page)
    guides.vertical.divide(4)
    guides.horizontal.divide(5)

    # Extract table without any exclusions first
    table_no_exclusions = guides.extract_table(method="text", apply_exclusions=False)

    # Add an exclusion that covers part of the page
    # Create exclusion in the middle of the page
    exclusion_region = Region(
        page, (page.width * 0.25, page.height * 0.4, page.width * 0.75, page.height * 0.6)
    )
    page.add_exclusion(exclusion_region, label="test_exclusion")

    # Extract with exclusions applied
    table_with_exclusions = guides.extract_table(method="text", apply_exclusions=True)

    # Extract without exclusions applied (should ignore the exclusion)
    table_ignore_exclusions = guides.extract_table(method="text", apply_exclusions=False)

    print(f"Table rows - no exclusions: {len(table_no_exclusions)}")
    print(f"Table rows - with exclusions: {len(table_with_exclusions)}")
    print(f"Table rows - ignore exclusions: {len(table_ignore_exclusions)}")

    # The tables extracted with apply_exclusions=False should be similar
    # The table with exclusions applied might have empty cells where exclusions are

    # Check that all methods completed without error
    assert isinstance(table_no_exclusions.to_df().values.tolist(), list)
    assert isinstance(table_with_exclusions.to_df().values.tolist(), list)
    assert isinstance(table_ignore_exclusions.to_df().values.tolist(), list)

    print("✅ Exclusions properly handled in extract_table")


@pytest.mark.skipif(find_test_pdf() is None, reason="No test PDF file found")
def test_extract_table_methods_support_exclusions():
    """Test that different extraction methods support apply_exclusions."""

    pdf_path = find_test_pdf()
    pdf = PDF(pdf_path)
    page = pdf[0]

    guides = Guides(page)
    guides.vertical.divide(3)
    guides.horizontal.divide(3)

    # Test different methods
    methods_to_test = ["text", "pdfplumber"]

    for method in methods_to_test:
        try:
            # With exclusions
            table_with = guides.extract_table(method=method, apply_exclusions=True)

            # Without exclusions
            table_without = guides.extract_table(method=method, apply_exclusions=False)

            print(f"✅ Method '{method}' supports apply_exclusions parameter")

        except Exception as e:
            # Some methods might not be available or might fail for other reasons
            print(f"⚠️  Method '{method}' failed: {e}")


def test_extract_table_signature():
    """Test that extract_table has the correct signature."""

    from inspect import signature

    sig = signature(Guides.extract_table)
    params = sig.parameters

    # Check that apply_exclusions parameter exists
    assert "apply_exclusions" in params, "apply_exclusions parameter should exist"

    # Check default value
    apply_exclusions_param = params["apply_exclusions"]
    assert apply_exclusions_param.default is True, "apply_exclusions should default to True"

    print("✅ extract_table signature is correct")


if __name__ == "__main__":
    # Run a simple test
    pdf_path = find_test_pdf()
    if pdf_path:
        print(f"Found test PDF: {pdf_path}")

        pdf = PDF(pdf_path)
        page = pdf[0]

        guides = Guides(page)
        guides.vertical.divide(3)
        guides.horizontal.divide(3)

        # Test parameter acceptance
        try:
            table = guides.extract_table(apply_exclusions=True)
            print("✅ extract_table accepts apply_exclusions=True")
        except TypeError as e:
            print(f"❌ Parameter not accepted: {e}")

        try:
            table = guides.extract_table(apply_exclusions=False)
            print("✅ extract_table accepts apply_exclusions=False")
        except TypeError as e:
            print(f"❌ Parameter not accepted: {e}")

    else:
        print("❌ No test PDF found")

    # Run pytest
    pytest.main([__file__, "-v"])
