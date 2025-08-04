#!/usr/bin/env python3
"""Test that exclusions work correctly with sliced page collections."""

from pathlib import Path

import pytest

from natural_pdf import PDF


def get_test_pdf_path():
    """Find a test PDF file."""
    project_root = Path(__file__).parent.parent

    # Try multiple possible locations
    possible_paths = [
        project_root / "pdfs/practice.pdf",
        project_root / "bad-pdfs/submissions/nara_book1_sample.pdf",
        project_root
        / "bad-pdfs/submissions/K046682-111320-OPA-LEA-Database-Install_1-Requestor-Copy---sample.pdf",
    ]

    for path in possible_paths:
        if path.exists():
            return path

    # If no test PDF found, skip the test
    pytest.skip("No test PDF found")


def test_page_specific_exclusions_preserved_in_slices():
    """Test that page-specific exclusions are preserved when accessing pages through slices."""
    pdf_path = get_test_pdf_path()
    pdf = PDF(pdf_path)

    # Add a page-specific exclusion
    page0 = pdf[0]
    page0.add_exclusion(page0.region(0, 100, page0.width, 150), label="page0-specific")

    # Access the same page through a slice
    sliced_page = pdf.pages[0:1][0]

    # The sliced page should have the same exclusions
    direct_exclusions = page0._get_exclusion_regions(include_callable=True)
    sliced_exclusions = sliced_page._get_exclusion_regions(include_callable=True)

    assert len(direct_exclusions) == len(
        sliced_exclusions
    ), f"Direct page has {len(direct_exclusions)} exclusions, sliced has {len(sliced_exclusions)}"

    # Check that the page-specific exclusion is present
    direct_labels = {exc.label for exc in direct_exclusions}
    sliced_labels = {exc.label for exc in sliced_exclusions}

    assert "page0-specific" in direct_labels
    assert "page0-specific" in sliced_labels


def test_pdf_and_page_exclusions_in_slices():
    """Test that both PDF-level and page-specific exclusions work in sliced collections."""
    pdf_path = get_test_pdf_path()
    pdf = PDF(pdf_path)

    # Add PDF-level exclusion
    pdf.add_exclusion(lambda page: page.region(0, 0, page.width, 50), label="pdf-header")

    # Add page-specific exclusion
    page0 = pdf[0]
    page0.add_exclusion(
        page0.region(0, page0.height - 50, page0.width, page0.height), label="page0-footer"
    )

    # Test different access patterns
    test_cases = [
        ("Direct page", page0),
        ("Via pages[0]", pdf.pages[0]),
        ("Via slice[0]", pdf.pages[0:1][0]),
        ("Via negative index", pdf.pages[-len(pdf.pages)]),  # First page via negative index
    ]

    for desc, page in test_cases:
        exclusions = page._get_exclusion_regions(include_callable=True)
        labels = {exc.label for exc in exclusions}

        assert len(exclusions) == 2, f"{desc}: Expected 2 exclusions, got {len(exclusions)}"
        assert "pdf-header" in labels, f"{desc}: Missing pdf-header exclusion"
        assert "page0-footer" in labels, f"{desc}: Missing page0-footer exclusion"


def test_exclusions_no_duplication():
    """Test that exclusions are not duplicated when accessing pages through slices."""
    pdf_path = get_test_pdf_path()
    pdf = PDF(pdf_path)

    # Add PDF-level exclusion
    pdf.add_exclusion(lambda page: page.region(0, 0, page.width, 50), label="pdf-header")

    # Add page-specific exclusion
    page0 = pdf[0]
    page0.add_exclusion(page0.region(0, 100, page0.width, 150), label="page0-specific")

    # Access through slice multiple times
    sliced1 = pdf.pages[0:1][0]
    sliced2 = pdf.pages[0:2][0]
    sliced3 = pdf.pages[:1][0]

    for desc, page in [("sliced1", sliced1), ("sliced2", sliced2), ("sliced3", sliced3)]:
        exclusions = page._get_exclusion_regions(include_callable=True)

        # Should have exactly 2 exclusions, not duplicated
        assert len(exclusions) == 2, f"{desc}: Expected 2 exclusions, got {len(exclusions)}"

        # Count occurrences of each label
        label_counts = {}
        for exc in exclusions:
            label_counts[exc.label] = label_counts.get(exc.label, 0) + 1

        for label, count in label_counts.items():
            assert count == 1, f"{desc}: Label '{label}' appears {count} times (should be 1)"


def test_exclusions_visualization_in_slices():
    """Test that exclusions can be visualized in sliced collections."""
    pdf_path = get_test_pdf_path()
    pdf = PDF(pdf_path)

    # Add exclusions
    pdf.add_exclusion(lambda page: page.region(0, 0, page.width, 50), label="pdf-header")
    page0 = pdf[0]
    page0.add_exclusion(
        page0.region(0, page0.height - 50, page0.width, page0.height), label="page0-footer"
    )

    # Test visualization doesn't raise errors
    try:
        # Full collection
        pdf.pages.show(limit=1, exclusions="red")

        # Sliced collection
        pdf.pages[0:1].show(exclusions="blue")

        # Single page from slice
        pdf.pages[0:1][0].show(exclusions="green")

        # Via negative index
        pdf.pages[-1].show(exclusions="red")

    except Exception as e:
        pytest.fail(f"Exclusion visualization failed: {e}")


if __name__ == "__main__":
    # Run tests
    test_page_specific_exclusions_preserved_in_slices()
    print("✅ test_page_specific_exclusions_preserved_in_slices passed")

    test_pdf_and_page_exclusions_in_slices()
    print("✅ test_pdf_and_page_exclusions_in_slices passed")

    test_exclusions_no_duplication()
    print("✅ test_exclusions_no_duplication passed")

    test_exclusions_visualization_in_slices()
    print("✅ test_exclusions_visualization_in_slices passed")

    print("\nAll tests passed!")
