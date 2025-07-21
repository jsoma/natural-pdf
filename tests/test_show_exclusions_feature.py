#!/usr/bin/env python3
"""Test the exclusions parameter in show() method."""

from pathlib import Path

import pytest

from natural_pdf import PDF


def find_test_pdf():
    """Find a test PDF."""
    project_root = Path(__file__).parent.parent
    pdf_path = project_root / "pdfs/practice.pdf"
    if not pdf_path.exists():
        pdf_path = (
            project_root
            / "bad-pdfs/submissions/Doc 06 - Approved Expenses 07.01.2022-06.30.2023 Marketplace Transactions - REDACTED.pdf"
        )
    return pdf_path if pdf_path.exists() else None


@pytest.mark.skipif(find_test_pdf() is None, reason="No test PDF file found")
def test_show_exclusions_parameter():
    """Test that show(exclusions='red') visualizes exclusions."""

    pdf_path = find_test_pdf()
    pdf = PDF(pdf_path)
    page = pdf[0]

    # Add some exclusions
    page.add_exclusion(page.region(0, 0, page.width, 50), label="header")
    page.add_exclusion(page.region(0, page.height - 50, page.width, page.height), label="footer")

    # Test that exclusions parameter works
    result = page.show(exclusions="red", limit=1)
    assert result is not None, "show() should return an image when exclusions are shown"

    # Test with different color
    result2 = page.show(exclusions="blue", limit=1)
    assert result2 is not None, "show() should work with different exclusion colors"

    # Test boolean value (True defaults to red)
    result3 = page.show(exclusions=True, limit=1)
    assert result3 is not None, "show() should work with exclusions=True"


@pytest.mark.skipif(find_test_pdf() is None, reason="No test PDF file found")
def test_pdf_show_exclusions():
    """Test that PDF-level show() also supports exclusions parameter."""

    pdf_path = find_test_pdf()
    pdf = PDF(pdf_path)

    # Add exclusions at PDF level
    pdf.add_exclusion(lambda page: page.region(0, 0, page.width, 30), label="pdf_header")

    # Test that PDF show() passes through exclusions parameter
    result = pdf.show(exclusions="green", limit=2)
    assert result is not None, "PDF show() should support exclusions parameter"


def demo_exclusions_visualization():
    """Demonstrate the exclusions visualization feature."""

    pdf_path = find_test_pdf()
    if not pdf_path:
        print("No test PDF found")
        return

    print(f"📄 Using PDF: {pdf_path.name}")

    pdf = PDF(pdf_path)
    page = pdf[0]

    # Add various exclusions
    print("\n1️⃣  Adding exclusion zones...")

    # Header exclusion
    page.add_exclusion(page.region(0, 0, page.width, 60), label="header")

    # Footer exclusion
    page.add_exclusion(page.region(0, page.height - 60, page.width, page.height), label="footer")

    # Side margin exclusion
    page.add_exclusion(page.region(0, 0, 50, page.height), label="left_margin")

    print("   ✅ Added 3 exclusion zones")

    print("\n2️⃣  Visualizing exclusions...")

    # Show with red exclusions
    print("   🔴 Showing exclusions in red:")
    img1 = page.show(exclusions="red", limit=1)
    if img1:
        print("      ✅ Successfully displayed with red exclusions")

    # Show with blue exclusions
    print("   🔵 Showing exclusions in blue:")
    img2 = page.show(exclusions="blue", limit=1)
    if img2:
        print("      ✅ Successfully displayed with blue exclusions")

    # Show with default (True)
    print("   ⚪ Showing exclusions with default color:")
    img3 = page.show(exclusions=True, limit=1)
    if img3:
        print("      ✅ Successfully displayed with default exclusions")

    print("\n3️⃣  Usage examples:")
    print("   page.show(exclusions='red')     # Show exclusions in red")
    print("   page.show(exclusions='blue')    # Show exclusions in blue")
    print("   page.show(exclusions=True)      # Show exclusions in default color (red)")
    print("   page.show()                     # Don't show exclusions")
    print("\n   pdf.show(exclusions='green')    # Works at PDF level too!")


if __name__ == "__main__":
    # Run demo
    demo_exclusions_visualization()

    # Run tests
    print("\n" + "=" * 70)
    print("Running tests...")
    pytest.main([__file__, "-v"])
