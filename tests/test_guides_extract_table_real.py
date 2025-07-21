#!/usr/bin/env python3
"""Test guides.extract_table() with a real PDF file."""

from pathlib import Path

import pytest

from natural_pdf import PDF
from natural_pdf.analyzers.guides import Guides


# Find a test PDF file
def find_test_pdf():
    """Find a suitable test PDF file."""
    project_root = Path(__file__).parent.parent

    # Look for PDFs in common locations
    pdf_paths = [
        project_root / "pdfs" / "practice.pdf",
        project_root / "test-files" / "practice.pdf",
        project_root / "tests" / "practice.pdf",
    ]

    # Also check bad-pdfs directory for any PDF
    bad_pdfs_dir = project_root / "bad-pdfs" / "submissions"
    if bad_pdfs_dir.exists():
        pdf_files = list(bad_pdfs_dir.glob("*.pdf"))[:3]  # Just use first 3
        pdf_paths.extend(pdf_files)

    for pdf_path in pdf_paths:
        if pdf_path.exists():
            return pdf_path

    return None


@pytest.mark.skipif(find_test_pdf() is None, reason="No test PDF file found")
def test_guides_extract_table_real_pdf():
    """Test guides.extract_table() with a real PDF file."""

    pdf_path = find_test_pdf()
    print(f"Using test PDF: {pdf_path}")

    # Load the PDF
    pdf = PDF(pdf_path)
    page = pdf[0]  # Use first page

    print(f"Page dimensions: {page.width} x {page.height}")
    print(f"Text elements on page: {len(page.find_all('text'))}")

    # Create guides from detected lines
    guides = Guides(page)

    # Add vertical guides from lines with threshold
    guides.vertical.from_lines(threshold=0.3)
    print(f"Found {len(guides.vertical)} vertical guides: {guides.vertical[:5]}...")

    # Add horizontal guides from lines
    guides.horizontal.from_lines()
    print(f"Found {len(guides.horizontal)} horizontal guides: {guides.horizontal[:5]}...")

    # Skip test if we don't have enough guides for a table
    if len(guides.vertical) < 2 or len(guides.horizontal) < 2:
        pytest.skip(
            f"Not enough guides found (v={len(guides.vertical)}, h={len(guides.horizontal)}) to form a table"
        )

    print("Testing guides.extract_table()...")

    # Test the extract_table method
    try:
        table_result = guides.extract_table(
            include_outer_boundaries=True, method="text"  # Use text method as it's most reliable
        )

        print(f"✅ Successfully extracted table!")
        print(f"Table shape: {len(table_result)} rows")

        if len(table_result) > 0:
            print(f"First row: {table_result[0]}")
        if len(table_result) > 1:
            print(f"Second row: {table_result[1]}")

        # Verify we got a TableResult object
        assert hasattr(table_result, "to_df"), "Should return TableResult object"
        assert hasattr(table_result, "__len__"), "Should be sequence-like"

        # Try converting to DataFrame (should work with keep_blank parameter)
        if len(table_result) > 0:
            df = table_result.to_df()  # Uses new keep_blank=False default
            print(f"DataFrame shape: {df.shape}")
            print(f"DataFrame columns: {list(df.columns)}")

    except Exception as e:
        print(f"❌ extract_table failed: {e}")
        # Still consider this a pass if the method exists and executes properly
        # (the PDF might just not have a good table structure)
        if "No table region was created" in str(e):
            pytest.skip(f"PDF doesn't have extractable table structure: {e}")
        else:
            raise  # Re-raise unexpected errors

    # Verify cleanup: check that no temporary regions remain
    remaining_temp_regions = [
        r for r in page._element_mgr.regions if getattr(r, "source", None) == "guides_temp"
    ]

    assert (
        len(remaining_temp_regions) == 0
    ), f"Found {len(remaining_temp_regions)} temporary regions that weren't cleaned up"
    print("✅ All temporary regions cleaned up successfully")


def test_guides_extract_table_parameters():
    """Test that guides.extract_table() accepts all the expected parameters."""

    pdf_path = find_test_pdf()
    if pdf_path is None:
        pytest.skip("No test PDF file found")

    pdf = PDF(pdf_path)
    page = pdf[0]

    # Create simple guides
    guides = Guides(verticals=[100, 200, 300], horizontals=[100, 150, 200], context=page)

    # Test that method accepts all parameters without error
    try:
        table_result = guides.extract_table(
            target=page,
            source="test_source",
            cell_padding=1.0,
            include_outer_boundaries=True,
            method="text",
            table_settings={"explicit_vertical_lines": [], "explicit_horizontal_lines": []},
            use_ocr=False,
            ocr_config=None,
            text_options={"snap_tolerance": 5},
            cell_extraction_func=None,
            show_progress=False,
            content_filter=None,
            multi_page="auto",
        )
        print("✅ All parameters accepted")

    except ValueError as e:
        if "No table region was created" in str(e):
            print("✅ Method executed properly (no table structure found)")
        else:
            raise

    # Verify cleanup of custom source
    remaining_temp_regions = [
        r for r in page._element_mgr.regions if getattr(r, "source", None) == "test_source"
    ]

    assert len(remaining_temp_regions) == 0, "Custom source regions not cleaned up"


@pytest.mark.skipif(find_test_pdf() is None, reason="No test PDF file found")
def test_guides_extract_table_workflow_comparison():
    """Compare the new extract_table() method with the traditional workflow."""

    pdf_path = find_test_pdf()
    pdf = PDF(pdf_path)
    page = pdf[0]

    # Create guides
    guides = Guides(page)
    guides.vertical.from_lines(threshold=0.3)
    guides.horizontal.from_lines()

    if len(guides.vertical) < 2 or len(guides.horizontal) < 2:
        pytest.skip("Not enough guides for comparison test")

    try:
        # Method 1: New extract_table() method
        print("Testing new extract_table() method...")
        table_result_new = guides.extract_table(method="text")

        # Method 2: Traditional workflow
        print("Testing traditional build_grid + extract workflow...")
        grid_result = guides.build_grid(include_outer_boundaries=True)
        table_region = grid_result["regions"]["table"]

        if table_region is None:
            pytest.skip("No table region created in traditional workflow")

        table_result_traditional = table_region.extract_table(method="text")

        # Clean up the traditional workflow regions
        for region_type in ["table", "rows", "columns", "cells"]:
            regions_to_clean = grid_result["regions"].get(region_type, [])
            if not isinstance(regions_to_clean, list):
                regions_to_clean = [regions_to_clean] if regions_to_clean else []

            for region in regions_to_clean:
                if region:
                    page._element_mgr.remove_element(region, element_type="regions")

        # Compare results
        print(f"New method rows: {len(table_result_new)}")
        print(f"Traditional method rows: {len(table_result_traditional)}")

        # Results should be similar (might not be identical due to different region boundaries)
        assert (
            abs(len(table_result_new) - len(table_result_traditional)) <= 1
        ), "Results should be similar"

        print("✅ Both methods produced similar results")

    except Exception as e:
        if "No table region was created" in str(e):
            pytest.skip(f"Table extraction not possible with this PDF: {e}")
        else:
            raise


if __name__ == "__main__":
    # Run a simple test to check if method works
    pdf_path = find_test_pdf()
    if pdf_path:
        print(f"Found test PDF: {pdf_path}")
        print("Running basic test...")

        pdf = PDF(pdf_path)
        page = pdf[0]

        guides = Guides(page)
        guides.vertical.from_lines(threshold=0.3)
        guides.horizontal.from_lines()

        print(f"Guides: {len(guides.vertical)} vertical, {len(guides.horizontal)} horizontal")

        if len(guides.vertical) >= 2 and len(guides.horizontal) >= 2:
            try:
                table_result = guides.extract_table()
                print(f"✅ Success! Extracted {len(table_result)} rows")
            except Exception as e:
                print(f"❌ Failed: {e}")
        else:
            print("❌ Not enough guides for table extraction")
    else:
        print("❌ No test PDF found")

    # Run pytest
    pytest.main([__file__, "-v"])
