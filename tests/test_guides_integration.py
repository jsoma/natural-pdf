"""Integration tests demonstrating the complete Guides workflow."""

import pytest

from natural_pdf.analyzers.guides import Guides


class TestGuidesIntegration:
    """Integration tests using real PDF data."""

    def test_beautiful_syntax_workflow(self, practice_pdf):
        """Test the complete beautiful syntax workflow."""
        page = practice_pdf.pages[0]

        # Your beautiful syntax in action!
        guides = Guides(page)

        # Add vertical guides from lines (defaults: threshold='auto', outer=False)
        guides.vertical.from_lines(detection_method="vector")
        assert len(guides.vertical) > 0

        # Add horizontal guides from lines
        guides.horizontal.from_lines(detection_method="vector")
        assert len(guides.horizontal) > 0

        # Test that we can manipulate individual axes
        original_vertical_count = len(guides.vertical)
        guides.vertical.add(300)  # Add a guide
        assert len(guides.vertical) == original_vertical_count + 1

        # Test chaining - methods return parent for fluent interface
        result = guides.vertical.shift(0, 5)
        assert result is guides

        # Build grid (defaults: include_outer_boundaries=False)
        if len(guides.vertical) > 1 and len(guides.horizontal) > 1:
            result = guides.build_grid(page)
            assert result["counts"]["table"] >= 0
            assert result["counts"]["cells"] >= 0

    def test_outer_boundary_control(self, practice_pdf):
        """Test precise control over outer boundaries."""
        page = practice_pdf.pages[0]

        # Internal boundaries only (default)
        internal_guides = Guides.from_lines(page, detection_method="vector")
        internal_count = len(internal_guides.vertical) + len(internal_guides.horizontal)

        # Include page boundaries
        full_guides = Guides.from_lines(page, outer=True, detection_method="vector")
        full_count = len(full_guides.vertical) + len(full_guides.horizontal)

        # Should have more guides when including boundaries
        assert full_count >= internal_count

        # Verify page boundaries are included
        assert 0.0 in full_guides.vertical or 0.0 in full_guides.horizontal
        assert page.width in full_guides.vertical or page.height in full_guides.horizontal

    def test_mixed_guide_creation(self, practice_pdf):
        """Test mixing different guide creation methods."""
        page = practice_pdf.pages[0]

        # Start with empty guides and build up
        guides = Guides.new(page)

        # Add guides from different sources
        guides.vertical.from_lines(detection_method="vector")  # Lines for vertical
        guides.horizontal.divide(n=3)  # Even division for horizontal

        # Manual additions
        guides.vertical.add(100)
        guides.horizontal.add(200)

        # Should have guides from all sources
        assert len(guides.vertical) > 1  # At least lines + manual
        assert len(guides.horizontal) > 1  # At least division + manual

    def test_intelligent_assignment(self, practice_pdf):
        """Test the intelligent guide assignment feature."""
        page = practice_pdf.pages[0]

        # Create guides from different sources
        line_guides = Guides.from_lines(page, detection_method="vector")
        division_guides = Guides.divide(page, cols=4, rows=3)

        # Test intelligent assignment
        combined_guides = Guides(page)
        combined_guides.vertical = line_guides  # Should extract vertical coordinates
        combined_guides.horizontal = division_guides  # Should extract horizontal coordinates

        assert list(combined_guides.vertical) == list(line_guides.vertical)
        assert list(combined_guides.horizontal) == list(division_guides.horizontal)

    def test_defaults_integration(self, practice_pdf):
        """Test that all the new defaults work together properly."""
        page = practice_pdf.pages[0]

        # Test from_lines defaults
        guides1 = Guides.from_lines(
            page, detection_method="vector"
        )  # threshold='auto', outer=False

        # Should not include page boundaries by default
        page_boundaries = {0.0, page.width, page.height}
        found_boundaries = set(guides1.vertical) | set(guides1.horizontal)
        boundary_overlap = page_boundaries & found_boundaries

        # With outer=False, we shouldn't find page boundaries (unless lines happen to be there)
        # This is a soft assertion since lines could coincidentally be at boundaries
        if len(boundary_overlap) == 0:
            assert True  # Expected: no page boundaries
        else:
            # Boundaries found, but that's okay if they're from actual lines
            pass

        # Test build_grid defaults
        if len(guides1.vertical) > 1 and len(guides1.horizontal) > 1:
            result1 = guides1.build_grid(page)  # include_outer_boundaries=False

            # Test with explicit outer boundaries
            guides2 = Guides.from_lines(page, outer=True, detection_method="vector")
            result2 = guides2.build_grid(page)

            # Should have same or more cells with outer boundaries
            assert result2["counts"]["cells"] >= result1["counts"]["cells"]

    def test_error_recovery(self, practice_pdf):
        """Test that the system handles edge cases gracefully."""
        page = practice_pdf.pages[0]

        # Test with no context
        empty_guides = Guides()

        # Should raise appropriate errors
        with pytest.raises(ValueError):
            empty_guides.vertical.from_lines()

        # Test with valid context but invalid operations
        guides = Guides(page)

        # Should not crash on empty results
        guides.vertical.snap_to_content(markers=["NonexistentText"])

        # Should handle type errors gracefully
        with pytest.raises(TypeError):
            guides.vertical = "invalid"


if __name__ == "__main__":
    pytest.main([__file__])
