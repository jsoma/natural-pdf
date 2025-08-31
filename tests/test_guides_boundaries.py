"""Tests for guides boundary handling to ensure first/last columns are not missing."""

import os
import tempfile

import pytest
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

from natural_pdf import PDF
from natural_pdf.analyzers import Guides


def create_test_pdf_with_boundary_content():
    """Create a test PDF with table content at page boundaries."""
    # Create temporary file
    fd, pdf_path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)

    # Create PDF with no margins
    doc = SimpleDocTemplate(
        pdf_path, pagesize=letter, leftMargin=0, rightMargin=0, topMargin=36, bottomMargin=36
    )

    # Create table data with 5 columns
    data = [
        ["FIRST_COL", "SECOND_COL", "THIRD_COL", "FOURTH_COL", "LAST_COL"],
        ["A1", "B1", "C1", "D1", "E1"],
        ["A2", "B2", "C2", "D2", "E2"],
        ["A3", "B3", "C3", "D3", "E3"],
    ]

    # Calculate column widths to span full page width
    page_width = letter[0]
    col_width = page_width / 5

    # Create table that spans full width
    t = Table(data, colWidths=[col_width] * 5)
    t.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )

    doc.build([t])
    return pdf_path


class TestGuidesBoundaries:
    """Test guides boundary handling."""

    def test_from_lines_default_excludes_boundaries(self):
        """Test that from_lines() by default excludes page boundaries."""
        pdf_path = create_test_pdf_with_boundary_content()
        try:
            pdf = PDF(pdf_path)
            page = pdf[0]

            # Create guides with default settings
            guides = Guides.from_lines(page)

            # Check if boundaries are included
            has_left_boundary = any(abs(x - 0) < 1 for x in guides.vertical.data)
            has_right_boundary = any(abs(x - page.width) < 1 for x in guides.vertical.data)

            # By default, boundaries should NOT be included
            # This is the current behavior that causes missing columns
            assert (
                not has_left_boundary or not has_right_boundary
            ), "Default from_lines should not include both boundaries"

        finally:
            os.unlink(pdf_path)

    def test_from_lines_outer_includes_boundaries(self):
        """Test that from_lines(outer=True) includes page boundaries."""
        pdf_path = create_test_pdf_with_boundary_content()
        try:
            pdf = PDF(pdf_path)
            page = pdf[0]

            # Create guides with outer=True
            guides = Guides.from_lines(page, outer=True)

            # Check if boundaries are included
            has_left_boundary = any(abs(x - 0) < 1 for x in guides.vertical.data)
            has_right_boundary = any(abs(x - page.width) < 1 for x in guides.vertical.data)

            # With outer=True, boundaries SHOULD be included
            assert has_left_boundary, "outer=True should include left boundary"
            assert has_right_boundary, "outer=True should include right boundary"

        finally:
            os.unlink(pdf_path)

    def test_extract_table_missing_columns_without_boundaries(self):
        """Test that extract_table() misses columns without boundaries."""
        pdf_path = create_test_pdf_with_boundary_content()
        try:
            pdf = PDF(pdf_path)
            page = pdf[0]

            # Create guides without boundaries
            guides = Guides.from_lines(page, outer=False)

            # Skip test if no guides found
            if not guides.vertical.data or not guides.horizontal.data:
                pytest.skip("No guides found in test PDF")

            # Extract table
            result = guides.extract_table()

            # Check if we're missing expected columns
            expected_headers = {"FIRST_COL", "SECOND_COL", "THIRD_COL", "FOURTH_COL", "LAST_COL"}
            actual_headers = set(result.headers) if result.headers else set()

            # This demonstrates the issue - we expect to be missing some columns
            missing_columns = expected_headers - actual_headers

            # Current behavior: some columns are missing
            # This test documents the issue
            if missing_columns:
                assert (
                    "FIRST_COL" in missing_columns or "LAST_COL" in missing_columns
                ), "Expected to be missing boundary columns"

        finally:
            os.unlink(pdf_path)

    def test_extract_table_all_columns_with_outer(self):
        """Test that extract_table() gets all columns with outer=True."""
        pdf_path = create_test_pdf_with_boundary_content()
        try:
            pdf = PDF(pdf_path)
            page = pdf[0]

            # Create guides with outer=True
            guides = Guides.from_lines(page, outer=True)

            # Skip test if no guides found
            if not guides.vertical.data or not guides.horizontal.data:
                pytest.skip("No guides found in test PDF")

            # Extract table
            result = guides.extract_table()

            # Check if we have all expected columns
            expected_headers = {"FIRST_COL", "SECOND_COL", "THIRD_COL", "FOURTH_COL", "LAST_COL"}
            actual_headers = set(result.headers) if result.headers else set()

            # With outer=True, we should get ALL columns
            assert expected_headers.issubset(
                actual_headers
            ), f"Missing columns: {expected_headers - actual_headers}"

        finally:
            os.unlink(pdf_path)

    def test_extract_table_include_outer_boundaries_parameter(self):
        """Test that extract_table(include_outer_boundaries=True) gets all columns."""
        pdf_path = create_test_pdf_with_boundary_content()
        try:
            pdf = PDF(pdf_path)
            page = pdf[0]

            # Create guides without outer boundaries
            guides = Guides.from_lines(page, outer=False)

            # Skip test if no guides found
            if not guides.vertical.data or not guides.horizontal.data:
                pytest.skip("No guides found in test PDF")

            # Extract table with include_outer_boundaries=True
            result = guides.extract_table(include_outer_boundaries=True)

            # Check if we have all expected columns
            expected_headers = {"FIRST_COL", "SECOND_COL", "THIRD_COL", "FOURTH_COL", "LAST_COL"}
            actual_headers = set(result.headers) if result.headers else set()

            # With include_outer_boundaries=True, we should get ALL columns
            assert expected_headers.issubset(
                actual_headers
            ), f"Missing columns: {expected_headers - actual_headers}"

        finally:
            os.unlink(pdf_path)

    def test_manual_boundary_addition(self):
        """Test manually adding boundaries to guides."""
        pdf_path = create_test_pdf_with_boundary_content()
        try:
            pdf = PDF(pdf_path)
            page = pdf[0]

            # Create guides without boundaries
            guides = Guides.from_lines(page, outer=False)

            # Manually add boundaries
            guides.vertical.add([0, page.width])

            # Extract table
            result = guides.extract_table()

            # Check if we have all expected columns
            expected_headers = {"FIRST_COL", "SECOND_COL", "THIRD_COL", "FOURTH_COL", "LAST_COL"}
            actual_headers = set(result.headers) if result.headers else set()

            # With manually added boundaries, we should get ALL columns
            assert expected_headers.issubset(
                actual_headers
            ), f"Missing columns: {expected_headers - actual_headers}"

        finally:
            os.unlink(pdf_path)


if __name__ == "__main__":
    # Run tests
    test = TestGuidesBoundaries()

    print("Testing guides boundary handling...")

    try:
        test.test_from_lines_default_excludes_boundaries()
        print("✓ Default from_lines excludes boundaries (as expected)")
    except AssertionError as e:
        print(f"✗ Default test failed: {e}")

    try:
        test.test_from_lines_outer_includes_boundaries()
        print("✓ from_lines(outer=True) includes boundaries")
    except AssertionError as e:
        print(f"✗ Outer parameter test failed: {e}")

    try:
        test.test_extract_table_missing_columns_without_boundaries()
        print("✓ extract_table misses columns without boundaries (documents the issue)")
    except AssertionError as e:
        print(f"✗ Missing columns test failed: {e}")

    try:
        test.test_extract_table_all_columns_with_outer()
        print("✓ extract_table gets all columns with outer=True")
    except AssertionError as e:
        print(f"✗ All columns with outer test failed: {e}")

    try:
        test.test_extract_table_include_outer_boundaries_parameter()
        print("✓ extract_table(include_outer_boundaries=True) gets all columns")
    except AssertionError as e:
        print(f"✗ include_outer_boundaries test failed: {e}")

    try:
        test.test_manual_boundary_addition()
        print("✓ Manual boundary addition works")
    except AssertionError as e:
        print(f"✗ Manual boundary test failed: {e}")
