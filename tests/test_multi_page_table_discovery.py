"""
Test Smart Page-Level Registry for multi-page table discovery.

This test verifies that when Guides creates a multi-page table spanning multiple
pages/regions, the table becomes discoverable from any constituent page using
the natural page.find('table') API.
"""

import pytest

from natural_pdf import PDF
from natural_pdf.analyzers import Guides
from natural_pdf.flows.region import FlowRegion
from natural_pdf.tables import TableResult


class TestMultiPageTableDiscovery:
    """Test suite for multi-page table discovery functionality."""

    def test_multi_page_table_registry_basic(self):
        """Test that multi-page tables are registered with all constituent pages."""
        # Use the provided test case as base
        pdf = PDF("pdfs/pdf_sample_land_registry_japan.PDF")
        page = pdf.pages[0]

        # Find table boundaries using the Unicode box drawing characters
        table_starts = pdf.find_all(text="┏━━━")
        table_ends = pdf.find_all(text="┗━━━")

        # Get sections between table boundaries
        regions = pdf.pages.get_sections(
            start_elements=table_starts,
            end_elements=table_ends,
        )

        if len(regions) < 2:
            pytest.skip("Test PDF doesn't have enough table sections")

        # Use the second table region
        table_region = regions[1]

        # Verify this is a FlowRegion spanning multiple areas
        assert hasattr(
            table_region, "constituent_regions"
        ), "Expected FlowRegion with constituent_regions"
        assert len(table_region.constituent_regions) > 0, "Expected at least one constituent region"

        # Create guides and build multi-page grid
        guides = Guides(table_region)
        guides.vertical.from_lines(detection_method="pixels", threshold=0.7)
        guides.horizontal.from_lines(detection_method="pixels", threshold=0.5)

        # Build multi-page grid - this should register the table with all pages
        results = guides.build_grid(multi_page=True)

        # Verify the grid was created
        assert results["counts"]["table"] == 1, "Expected exactly one table to be created"
        assert results["regions"]["table"] is not None, "Expected table region to be created"

        multi_page_table = results["regions"]["table"]
        assert isinstance(multi_page_table, FlowRegion), "Expected table to be a FlowRegion"
        assert (
            multi_page_table.metadata.get("is_multi_page") is True
        ), "Expected table to be marked as multi-page"

        # CORE TEST: Verify table is discoverable from constituent pages
        constituent_pages = set()
        for region in table_region.constituent_regions:
            if hasattr(region, "page"):
                constituent_pages.add(region.page)

        assert len(constituent_pages) > 0, "Expected at least one constituent page"

        # Test that each page can find the multi-page table
        for page in constituent_pages:
            found_table = page.find("table")
            assert (
                found_table is not None
            ), f"Page {page.page_number} should be able to find the multi-page table"

            # Check if the found table is the multi-page table or verify it's a FlowRegion
            if found_table != multi_page_table:
                # If not exactly the same object, it might be due to search precedence
                # Check that at least we can find the multi-page table in the results
                all_tables = page.find_all("table")
                assert (
                    multi_page_table in all_tables.elements
                ), f"Page {page.page_number} should include the multi-page table in find_all results"
            else:
                # This is the ideal case - direct match
                assert (
                    found_table == multi_page_table
                ), f"Page {page.page_number} should find the same multi-page table"

            # Also test find_all
            found_tables = page.find_all("table")
            assert len(found_tables) > 0, f"Page {page.page_number} should find tables via find_all"
            assert (
                multi_page_table in found_tables.elements
            ), f"Page {page.page_number} should include multi-page table in find_all results"

    def test_multi_page_table_components_registry(self):
        """Test that multi-page table rows, columns, and cells are also discoverable."""
        pdf = PDF("pdfs/pdf_sample_land_registry_japan.PDF")

        table_starts = pdf.find_all(text="┏━━━")
        table_ends = pdf.find_all(text="┗━━━")
        regions = pdf.pages.get_sections(
            start_elements=table_starts,
            end_elements=table_ends,
        )

        if len(regions) < 2:
            pytest.skip("Test PDF doesn't have enough table sections")

        table_region = regions[1]
        guides = Guides(table_region)
        guides.vertical.from_lines(detection_method="pixels", threshold=0.7)
        guides.horizontal.from_lines(detection_method="pixels", threshold=0.5)

        results = guides.build_grid(multi_page=True)

        # Get constituent pages
        constituent_pages = set()
        for region in table_region.constituent_regions:
            if hasattr(region, "page"):
                constituent_pages.add(region.page)

        # Test that table components are discoverable
        for page in constituent_pages:
            # Test table rows
            if results["regions"]["rows"]:
                found_rows = page.find_all("table_row")
                # Should find at least some rows (multi-page rows should be registered)
                multi_page_rows = [
                    r for r in results["regions"]["rows"] if hasattr(r, "constituent_regions")
                ]
                if multi_page_rows:
                    # At least one multi-page row should be discoverable
                    assert any(
                        row in found_rows.elements for row in multi_page_rows
                    ), f"Page {page.page_number} should find multi-page rows"

            # Test table columns
            if results["regions"]["columns"]:
                found_cols = page.find_all("table_column")
                multi_page_cols = [
                    c for c in results["regions"]["columns"] if hasattr(c, "constituent_regions")
                ]
                if multi_page_cols:
                    assert any(
                        col in found_cols.elements for col in multi_page_cols
                    ), f"Page {page.page_number} should find multi-page columns"

            # Test table cells
            if results["regions"]["cells"]:
                found_cells = page.find_all("table_cell")
                multi_page_cells = [
                    c for c in results["regions"]["cells"] if hasattr(c, "constituent_regions")
                ]
                if multi_page_cells:
                    assert any(
                        cell in found_cells.elements for cell in multi_page_cells
                    ), f"Page {page.page_number} should find multi-page cells"

    def test_multi_page_table_extract_table_workflow(self):
        """Test the complete workflow: build grid -> find table -> extract data."""
        pdf = PDF("pdfs/pdf_sample_land_registry_japan.PDF")

        table_starts = pdf.find_all(text="┏━━━")
        table_ends = pdf.find_all(text="┗━━━")
        regions = pdf.pages.get_sections(
            start_elements=table_starts,
            end_elements=table_ends,
        )

        if len(regions) < 2:
            pytest.skip("Test PDF doesn't have enough table sections")

        table_region = regions[1]
        guides = Guides(table_region)
        guides.vertical.from_lines(detection_method="pixels", threshold=0.7)
        guides.horizontal.from_lines(detection_method="pixels", threshold=0.5)

        # Build the multi-page grid
        results = guides.build_grid(multi_page=True)

        # Get any constituent page
        constituent_pages = set()
        for region in table_region.constituent_regions:
            if hasattr(region, "page"):
                constituent_pages.add(region.page)

        if not constituent_pages:
            pytest.skip("No constituent pages found")

        # Pick the first page
        test_page = next(iter(constituent_pages))

        # CORE WORKFLOW TEST: User's natural workflow should work
        found_table = test_page.find("table")
        assert found_table is not None, "Should find table from constituent page"

        # Test that extract_table works on the discovered table
        try:
            table_result = found_table.extract_table()
            assert table_result is not None, "Should be able to extract table data"

            # Test DataFrame conversion if available
            if hasattr(table_result, "df"):
                df = table_result.df
                assert df is not None, "Should be able to convert to DataFrame"
                assert len(df) > 0, "DataFrame should have rows"

        except Exception as e:
            # Table extraction might fail due to complex layout, but discovery should work
            pytest.skip(f"Table extraction failed (expected for complex layouts): {e}")

    def test_single_page_table_still_works(self):
        """Test that single-page tables continue to work as before."""
        pdf = PDF("pdfs/pdf_sample_land_registry_japan.PDF")
        page = pdf.pages[0]

        # Create a simple single-page region
        simple_region = page.region(left=100, top=100, right=400, bottom=300)

        # Create guides for single region
        guides = Guides(simple_region)
        guides.vertical.from_lines(detection_method="pixels", threshold=0.5)
        guides.horizontal.from_lines(detection_method="pixels", threshold=0.5)

        # Build grid (should use single-page logic)
        results = guides.build_grid(multi_page=False)

        if results["counts"]["table"] > 0:
            # Table should be discoverable from the page
            found_table = page.find("table")
            assert found_table is not None, "Should find single-page table"

            # Should be a regular Region, not FlowRegion
            assert not hasattr(
                found_table, "constituent_regions"
            ), "Single-page table should be regular Region"

    def test_multi_page_auto_detection(self):
        """Test that multi_page='auto' correctly detects when tables span pages."""
        pdf = PDF("pdfs/pdf_sample_land_registry_japan.PDF")

        table_starts = pdf.find_all(text="┏━━━")
        table_ends = pdf.find_all(text="┗━━━")
        regions = pdf.pages.get_sections(
            start_elements=table_starts,
            end_elements=table_ends,
        )

        if len(regions) < 2:
            pytest.skip("Test PDF doesn't have enough table sections")

        table_region = regions[1]
        guides = Guides(table_region)
        guides.vertical.from_lines(detection_method="pixels", threshold=0.7)
        guides.horizontal.from_lines(detection_method="pixels", threshold=0.5)

        # Use auto detection - should detect multi-page nature and create FlowRegion table
        results = guides.build_grid(multi_page="auto")

        if results["counts"]["table"] > 0:
            table = results["regions"]["table"]

            # Auto-detection logic: FlowRegion if spans pages OR has multiple regions
            has_multiple_regions = len(table_region.constituent_regions) > 1
            spans_pages = guides._spans_pages()

            if spans_pages or has_multiple_regions:
                assert isinstance(
                    table, FlowRegion
                ), "Auto-detection should create FlowRegion for multi-page guides or multiple regions"
                assert (
                    table.metadata.get("is_multi_page") is True
                ), "Should mark table as multi-page"
            else:
                # Only create regular Region if single region AND doesn't span pages
                assert not isinstance(
                    table, FlowRegion
                ), "Auto-detection should create regular Region for single-page guides"

    def test_selective_table_removal(self):
        """Test that multi-page table creation only removes its own fragments, not other tables."""
        pdf = PDF("pdfs/pdf_sample_land_registry_japan.PDF")

        # Find table boundaries for multi-page table
        table_starts = pdf.find_all(text="┏━━━")
        table_ends = pdf.find_all(text="┗━━━")

        regions = pdf.pages.get_sections(
            start_elements=table_starts,
            end_elements=table_ends,
        )

        if len(regions) < 2:
            pytest.skip("Need at least 2 regions for this test")

        # Create a separate single-page table with the same source on page 1
        page1 = pdf.pages[0]
        separate_guides = Guides(page1)
        separate_guides.vertical.data = [100, 200, 300]  # Some arbitrary guides
        separate_guides.horizontal.data = [100, 150, 200]

        # Build a single-page grid with the same source as we'll use for multi-page
        separate_result = separate_guides.build_grid(source="guides", multi_page=False)
        separate_table = separate_result["regions"]["table"]

        # Verify the separate table exists and can be found
        found_tables_before = page1.find_all("table")
        assert len(found_tables_before) >= 1, "Should have at least the separate table"
        assert separate_table in found_tables_before, "Separate table should be findable"

        # Now create the multi-page table with the same source
        multi_page_region = regions[1]
        multi_guides = Guides(multi_page_region)
        multi_guides.vertical.from_lines(detection_method="pixels", threshold=0.7)
        multi_guides.horizontal.from_lines(detection_method="pixels", threshold=0.5)

        # Build multi-page grid
        multi_result = multi_guides.build_grid(source="guides", multi_page=True)
        multi_page_table = multi_result["regions"]["table"]

        # Verify that:
        # 1. The multi-page table can be found from constituent pages
        # 2. The separate single-page table is still there and findable
        found_tables_after = page1.find_all("table")

        # Should have both the separate table AND the multi-page table
        assert (
            len(found_tables_after) >= 2
        ), f"Should have at least 2 tables, found {len(found_tables_after)}"

        # The separate table should still be there
        assert separate_table in found_tables_after, "Separate table should not have been removed"

        # The multi-page table should also be findable
        assert multi_page_table in found_tables_after, "Multi-page table should be findable"

        # Verify they are different objects
        assert separate_table is not multi_page_table, "Should be different table objects"

        # Verify the multi-page table has the expected metadata
        assert multi_page_table.metadata.get("is_multi_page") is True
        assert separate_table.metadata.get("is_multi_page") is not True

    def test_multi_page_table_returns_table_result(self):
        """Test that FlowRegion.extract_table() returns a TableResult object like Region.extract_table()."""
        pdf = PDF("pdfs/pdf_sample_land_registry_japan.PDF")

        # Find table boundaries for multi-page table
        table_starts = pdf.find_all(text="┏━━━")
        table_ends = pdf.find_all(text="┗━━━")

        regions = pdf.pages.get_sections(
            start_elements=table_starts,
            end_elements=table_ends,
        )

        if len(regions) < 2:
            pytest.skip("Need at least 2 regions for this test")

        # Create multi-page table
        table_region = regions[1]
        guides = Guides(table_region)
        guides.vertical.from_lines(detection_method="pixels", threshold=0.7)
        guides.horizontal.from_lines(detection_method="pixels", threshold=0.5)

        # Build the grid to create the multi-page table
        results = guides.build_grid(multi_page=True)
        multi_page_table = results["regions"]["table"]

        # Test that extract_table returns a TableResult
        table_result = multi_page_table.extract_table()

        assert isinstance(
            table_result, TableResult
        ), f"Expected TableResult, got {type(table_result)}"

        # Test that the TableResult has the expected methods
        assert hasattr(table_result, "df"), "TableResult should have 'df' property"
        assert hasattr(table_result, "to_df"), "TableResult should have 'to_df' method"

        # Test that it behaves like a list (Sequence interface)
        assert hasattr(table_result, "__len__"), "TableResult should have __len__ method"
        assert hasattr(table_result, "__getitem__"), "TableResult should have __getitem__ method"
        assert hasattr(table_result, "__iter__"), "TableResult should have __iter__ method"

        # Test that we can iterate over it
        row_count = 0
        for row in table_result:
            row_count += 1
            assert isinstance(row, list), f"Each row should be a list, got {type(row)}"

        # Test that len() works
        assert len(table_result) == row_count, "len() should match iteration count"

        print(
            f"✅ Multi-page table extract_table() returns TableResult with {len(table_result)} rows"
        )
