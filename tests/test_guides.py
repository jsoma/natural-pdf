"""Tests for the Guides functionality."""

from unittest.mock import Mock

import pytest

from natural_pdf import PDF
from natural_pdf.analyzers.guides import Guides, GuidesList


class TestGuidesInitialization:
    """Test different ways to initialize Guides objects."""

    def test_empty_initialization(self):
        """Test creating empty guides."""
        guides = Guides()
        assert len(guides.vertical) == 0
        assert len(guides.horizontal) == 0
        assert isinstance(guides.vertical, GuidesList)
        assert isinstance(guides.horizontal, GuidesList)

    def test_with_coordinates(self):
        """Test creating guides with explicit coordinates."""
        guides = Guides([100, 200, 300], [50, 150])
        assert list(guides.vertical) == [100, 200, 300]
        assert list(guides.horizontal) == [50, 150]

    def test_shorthand_with_page(self):
        """Test the Guides(page) shorthand syntax."""
        mock_page = Mock()
        mock_page.bbox = (0, 0, 612, 792)

        guides = Guides(mock_page)
        assert guides.context is mock_page
        assert guides.bounds == (0, 0, 612, 792)
        assert len(guides.vertical) == 0
        assert len(guides.horizontal) == 0

    def test_new_factory_method(self):
        """Test the Guides.new() factory method."""
        guides = Guides.new()
        assert len(guides.vertical) == 0
        assert len(guides.horizontal) == 0

        mock_page = Mock()
        mock_page.bbox = (0, 0, 500, 700)
        guides_with_context = Guides.new(mock_page)
        assert guides_with_context.context is mock_page


class TestGuidesFactoryMethods:
    """Test factory methods for creating guides."""

    def test_divide_method(self):
        """Test the divide factory method."""
        mock_page = Mock()
        mock_page.bbox = (0, 0, 600, 800)

        # Test dividing into columns
        guides = Guides.divide(mock_page, cols=3)
        expected_verticals = [0, 200, 400, 600]  # 4 guides for 3 columns
        assert list(guides.vertical) == expected_verticals
        assert len(guides.horizontal) == 0

        # Test dividing into rows
        guides = Guides.divide(mock_page, rows=2)
        expected_horizontals = [0, 400, 800]  # 3 guides for 2 rows
        assert list(guides.horizontal) == expected_horizontals
        assert len(guides.vertical) == 0

    def test_from_lines_default_behavior(self):
        """Test from_lines with default parameters."""
        mock_line1 = Mock()
        mock_line1.is_horizontal = False
        mock_line1.is_vertical = True
        mock_line1.x0, mock_line1.x1 = 150, 152
        mock_line1.top, mock_line1.bottom = 50, 550
        mock_line1.height = 500

        mock_line2 = Mock()
        mock_line2.is_horizontal = True
        mock_line2.is_vertical = False
        mock_line2.x0, mock_line2.x1 = 100, 500
        mock_line2.top, mock_line2.bottom = 200, 202
        mock_line2.width = 400

        # Create a simple class to avoid Mock's hasattr issues
        class MockPage:
            def __init__(self):
                self.bbox = (0, 0, 500, 600)
                self.lines = [mock_line1, mock_line2]

        mock_page = MockPage()

        # Test default behavior (outer=False)
        guides = Guides.from_lines(mock_page, detection_method="vector")
        assert 151.0 in guides.vertical  # Midpoint of vertical line
        assert 201.0 in guides.horizontal  # Midpoint of horizontal line

        # Should NOT include outer boundaries by default
        assert 0.0 not in guides.vertical
        assert 500.0 not in guides.vertical

    def test_from_lines_with_outer_true(self):
        """Test from_lines with outer=True."""
        mock_line = Mock()
        mock_line.is_horizontal = False
        mock_line.is_vertical = True
        mock_line.x0, mock_line.x1 = 150, 152
        mock_line.top, mock_line.bottom = 50, 550
        mock_line.height = 500

        # Create a simple class to avoid Mock's hasattr issues
        class MockPage:
            def __init__(self):
                self.bbox = (0, 0, 500, 600)
                self.lines = [mock_line]

        mock_page = MockPage()

        guides = Guides.from_lines(mock_page, outer=True, detection_method="vector")
        assert 0.0 in guides.vertical  # Left boundary
        assert 151.0 in guides.vertical  # Line midpoint
        assert 500.0 in guides.vertical  # Right boundary

    def test_from_content_default_behavior(self):
        """Test from_content with default parameters."""
        mock_element = Mock()
        mock_element.x0, mock_element.x1 = 100, 200
        mock_element.top, mock_element.bottom = 50, 70

        # Create a simple class to avoid Mock's hasattr issues
        class MockPage:
            def __init__(self):
                self.bbox = (0, 0, 500, 600)
                self.find = Mock(return_value=mock_element)

        mock_page = MockPage()

        # Test default behavior (outer=True, align='left')
        guides = Guides.from_content(mock_page, markers=["Header"])
        mock_page.find.assert_called_with('text:contains("Header")', apply_exclusions=True)

        # Should include the element's left edge and outer boundaries
        assert 0.0 in guides.vertical  # Left boundary (outer=True)
        assert 100.0 in guides.vertical  # Element left edge
        assert 500.0 in guides.vertical  # Right boundary (outer=True)


class TestGuidesListMethods:
    """Test the beautiful GuidesList syntax."""

    def test_guides_list_basic_operations(self):
        """Test that GuidesList acts like a list."""
        guides = Guides([100, 200], [50, 150])

        # Test list-like behavior
        guides.vertical.append(300)
        assert 300 in guides.vertical

        guides.horizontal.extend([250, 350])
        assert 250 in guides.horizontal
        assert 350 in guides.horizontal

        # Test indexing
        assert guides.vertical[0] == 100
        assert len(guides.vertical) == 3

    def test_vertical_from_content(self):
        """Test guides.vertical.from_content() syntax."""
        mock_element = Mock()
        mock_element.x0, mock_element.x1 = 150, 250

        # Create a simple class to avoid Mock's hasattr issues
        class MockPage:
            def __init__(self):
                self.bbox = (0, 0, 500, 600)
                self.find = Mock(return_value=mock_element)

        mock_page = MockPage()

        guides = Guides(mock_page)
        result = guides.vertical.from_content(markers=["Header"], align="left")

        # Should return parent Guides for chaining
        assert result is guides
        # Should have added the element's position
        assert 150.0 in guides.vertical
        # Should have outer boundaries (default outer=True for content)
        assert 0.0 in guides.vertical
        assert 500.0 in guides.vertical

    def test_horizontal_from_lines(self):
        """Test guides.horizontal.from_lines() syntax."""
        mock_line = Mock()
        mock_line.is_horizontal = True
        mock_line.is_vertical = False
        mock_line.x0, mock_line.x1 = 100, 500
        mock_line.top, mock_line.bottom = 200, 202
        mock_line.width = 400

        # Create a simple class to avoid Mock's hasattr issues
        class MockPage:
            def __init__(self):
                self.bbox = (0, 0, 500, 600)
                self.lines = [mock_line]

        mock_page = MockPage()

        guides = Guides(mock_page)
        result = guides.horizontal.from_lines(detection_method="vector")

        # Should return parent Guides for chaining
        assert result is guides
        # Should have added the line's position
        assert 201.0 in guides.horizontal
        # Should NOT have outer boundaries (default outer=False for lines)
        assert 0.0 not in guides.horizontal
        assert 600.0 not in guides.horizontal

    def test_guides_list_manipulation_methods(self):
        """Test manipulation methods on GuidesList."""
        guides = Guides([100, 200, 300], [50, 150, 250])

        # Test add method
        result = guides.vertical.add(400)
        assert result is guides  # Returns parent for chaining
        assert 400 in guides.vertical

        # Test shift method
        original_second = guides.vertical[1]
        guides.vertical.shift(1, 25)
        assert guides.vertical[1] == original_second + 25

        # Test remove_at method
        guides.horizontal.remove_at(0)
        assert 50 not in guides.horizontal
        assert len(guides.horizontal) == 2

        # Test clear_all method
        guides.vertical.clear_all()
        assert len(guides.vertical) == 0

    def test_intelligent_assignment(self):
        """Test that guides.vertical = other_guides works."""
        guides1 = Guides([100, 200], [50, 150])
        guides2 = Guides([300, 400], [250, 350])

        # Test assigning from another Guides object
        guides1.horizontal = guides2
        assert list(guides1.horizontal) == [250, 350]

        # Test assigning a list
        guides1.vertical = [500, 600]
        assert list(guides1.vertical) == [500, 600]


class TestGuidesWithRealPDF:
    """Test Guides with actual PDF data."""

    def test_guides_with_practice_pdf(self, practice_pdf):
        """Test creating guides from the practice PDF."""
        page = practice_pdf.pages[0]

        # Test that we can create guides from the page
        guides = Guides(page)
        assert guides.context is page
        # Page doesn't have bbox, but should have width/height
        assert hasattr(page, "width")
        assert hasattr(page, "height")

        # Test from_lines method
        line_guides = Guides.from_lines(page)
        assert len(line_guides.vertical) > 0 or len(line_guides.horizontal) > 0

        # Test the beautiful syntax
        guides.vertical.from_lines()
        guides.horizontal.from_lines()

        # Should have some guides from the PDF's lines
        total_guides = len(guides.vertical) + len(guides.horizontal)
        assert total_guides > 0

    def test_chaining_methods(self, practice_pdf):
        """Test method chaining with real PDF."""
        page = practice_pdf.pages[0]

        # Test complex chaining - create separate objects to avoid addition issues
        guides = Guides.new(page)
        guides.add_lines(axis="vertical")
        guides.add_lines(axis="horizontal")

        # Should have guides from both axes
        assert len(guides.vertical) > 0 or len(guides.horizontal) > 0

    def test_build_grid_default_behavior(self, practice_pdf):
        """Test that build_grid respects the new defaults."""
        page = practice_pdf.pages[0]

        # Create guides with lines only (no outer boundaries)
        guides = Guides.from_lines(page)  # outer=False by default

        result = None
        if len(guides.vertical) > 1 and len(guides.horizontal) > 1:
            # Test that build_grid doesn't add outer boundaries by default
            result = guides.build_grid(page)  # include_outer_boundaries=False by default

            # Should create some structure
            assert result["counts"]["table"] >= 0
            assert result["counts"]["cells"] >= 0

        # Test with outer boundaries explicitly
        guides_with_outer = Guides.from_lines(page, outer=True)
        if len(guides_with_outer.vertical) > 1 and len(guides_with_outer.horizontal) > 1:
            result_with_outer = guides_with_outer.build_grid(page)

            # Should have more cells when including page boundaries
            baseline_cells = result["counts"]["cells"] if result else 0
            assert result_with_outer["counts"]["cells"] >= baseline_cells


class TestGuidesErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_assignments(self):
        """Test that invalid assignments raise appropriate errors."""
        guides = Guides()

        # Test string assignment (should fail)
        with pytest.raises(TypeError, match="cannot be a string"):
            guides.vertical = "invalid"

        with pytest.raises(TypeError, match="cannot be a string"):
            guides.horizontal = "invalid"

    def test_empty_context_methods(self):
        """Test methods when no context is available."""
        guides = Guides()

        # Should raise ValueError when no context available
        with pytest.raises(ValueError, match="No object provided and no context"):
            guides.vertical.from_lines()

        with pytest.raises(ValueError, match="No object provided and no context"):
            guides.horizontal.from_content(markers=["test"])

    def test_nonexistent_content(self):
        """Test behavior when content markers aren't found."""

        # Create a simple class to avoid Mock's hasattr issues
        class MockPage:
            def __init__(self):
                self.bbox = (0, 0, 500, 600)
                self.find = Mock(return_value=None)  # No element found

        mock_page = MockPage()

        guides = Guides(mock_page)

        # Should not crash, but should warn
        result = guides.vertical.snap_to_content(markers=["NonexistentText"])
        assert result is guides  # Should still return parent for chaining


class TestGuidesDefaults:
    """Test that all the defaults are set correctly."""

    def test_from_lines_defaults(self):
        """Test that from_lines has correct defaults."""
        import inspect

        # Test GuidesList.from_lines defaults
        sig = inspect.signature(GuidesList.from_lines)
        params = sig.parameters
        assert params["threshold"].default == "auto"
        assert params["outer"].default is False

        # Test main Guides.from_lines defaults
        sig2 = inspect.signature(Guides.from_lines)
        params2 = sig2.parameters
        assert params2["threshold"].default == "auto"
        assert params2["outer"].default is False

    def test_from_content_defaults(self):
        """Test that from_content has correct defaults."""
        import inspect

        sig = inspect.signature(Guides.from_content)
        params = sig.parameters
        assert params["outer"].default is True

    def test_build_grid_defaults(self):
        """Test that build_grid has correct defaults."""
        import inspect

        sig = inspect.signature(Guides.build_grid)
        params = sig.parameters
        assert params["include_outer_boundaries"].default is False


def test_markers_parameter_flexibility():
    """Test that markers parameter accepts different input types."""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]
    guides = Guides(page)

    # Test 1: Single selector string
    guides.vertical.from_content(markers='text:contains("Total")')
    initial_count = len(guides.vertical)
    assert initial_count > 0, "Should find guides from single selector"

    # Test 2: Single literal text string
    guides.vertical.clear_all()
    guides.vertical.from_content(markers="Total")
    assert len(guides.vertical) > 0, "Should find guides from literal text"

    # Test 3: List of selectors
    guides.vertical.clear_all()
    guides.vertical.from_content(markers=['text:contains("Total")', 'text:contains("Date")'])
    list_count = len(guides.vertical)
    assert list_count > 0, "Should find guides from list of selectors"

    # Test 4: ElementCollection
    elements = page.find_all('text:contains("Total")')
    if elements:
        guides.vertical.clear_all()
        guides.vertical.from_content(markers=elements)
        collection_count = len(guides.vertical)
        assert collection_count > 0, "Should find guides from ElementCollection"

    # Test 5: None/empty (without outer boundaries)
    guides.vertical.clear_all()
    guides.vertical.from_content(markers=None, outer=False)
    assert len(guides.vertical) == 0, "Should handle None markers gracefully"


def test_class_method_markers_flexibility():
    """Test that the class method also accepts flexible markers."""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    # Test single selector
    guides1 = Guides.from_content(page, axis="vertical", markers='text:contains("Total")')
    assert len(guides1.vertical) > 0, "Class method should handle single selector"

    # Test ElementCollection
    elements = page.find_all('text:contains("Total")')
    if elements:
        guides2 = Guides.from_content(page, axis="vertical", markers=elements)
        assert len(guides2.vertical) > 0, "Class method should handle ElementCollection"


def test_snap_to_content_markers_flexibility():
    """Test that snap_to_content also accepts flexible markers."""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    # Create some initial guides
    guides = Guides([100, 200, 300], context=page)
    original_positions = guides.vertical.data.copy()

    # Test snapping with single selector
    guides.vertical.snap_to_content(markers='text:contains("Total")', tolerance=50)

    # Should modify positions (though we can't predict exact values)
    # At minimum, the function should run without error
    assert len(guides.vertical) == len(original_positions), "Should preserve guide count"


def test_add_content_markers_flexibility():
    """Test that add_content method accepts flexible markers."""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    # Create empty guides
    guides = Guides(context=page)

    # Test adding with single selector
    guides.add_content(axis="vertical", markers='text:contains("Total")')
    assert len(guides.vertical) > 0, "add_content should handle single selector"

    # Test adding with ElementCollection
    elements = page.find_all('text:contains("Date")')
    if elements:
        initial_count = len(guides.vertical)
        guides.add_content(axis="vertical", markers=elements)
        assert len(guides.vertical) >= initial_count, "add_content should handle ElementCollection"


def test_add_method_flexibility():
    """Test that add method accepts both single values and lists."""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]
    guides = Guides(page)

    # Test 1: Add single position
    guides.vertical.add(100)
    assert len(guides.vertical) == 1
    assert 100 in guides.vertical

    # Test 2: Add list of positions
    guides.vertical.add([200, 300, 400])
    assert len(guides.vertical) == 4  # 100 + 3 new ones
    assert all(pos in guides.vertical for pos in [100, 200, 300, 400])

    # Test 3: Verify sorting
    assert guides.vertical.data == [100, 200, 300, 400]

    # Test 4: Test with horizontal guides
    guides.horizontal.add([50, 150])
    guides.horizontal.add(250)
    assert len(guides.horizontal) == 3
    assert guides.horizontal.data == [50, 150, 250]

    # Test 5: Test with tuple (should work like list)
    guides.vertical.clear_all()
    guides.vertical.add((10, 20, 30))
    assert guides.vertical.data == [10, 20, 30]


def test_pixel_based_line_detection():
    """Test that pixel-based line detection works in Guides API."""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    # Test 1: Create guides from pixel-based line detection
    guides = Guides.from_lines(
        page, detection_method="pixels", threshold="auto", max_lines_h=5, max_lines_v=5
    )

    # Should have detected some lines (at least in one direction)
    assert len(guides.vertical) > 0 or len(guides.horizontal) > 0, "Should detect lines from pixels"
    print(f"Detected {len(guides.vertical)} vertical and {len(guides.horizontal)} horizontal lines")

    # Test 2: Using fluent API with pixel detection for horizontal lines
    guides2 = Guides(page)
    guides2.horizontal.from_lines(detection_method="pixels", max_lines=3)
    assert len(guides2.horizontal) <= 3, "Should detect at most 3 horizontal lines"

    # Test 3: Pixel detection with threshold
    guides_threshold = Guides.from_lines(
        page,
        detection_method="pixels",
        threshold=0.1,  # Low threshold to find more lines
        axis="horizontal",  # Focus on horizontal since that's what the PDF has
    )

    assert len(guides_threshold.horizontal) > 0, "Should find horizontal lines with low threshold"

    # Test 4: Pixel detection with custom parameters
    guides3 = Guides.from_lines(
        page,
        detection_method="pixels",
        resolution=150,
        min_gap_h=10,
        min_gap_v=10,
        binarization_method="adaptive",
        axis="both",
    )
    # At least one direction should have lines
    assert len(guides3.vertical) > 0 or len(guides3.horizontal) > 0

    # Test 5: Ensure pixel detection creates actual LineElements
    # Check that the lines were added to the page
    pixel_lines = [l for l in page.lines if getattr(l, "source", None) == "guides_detection"]
    assert len(pixel_lines) > 0, "Pixel detection should create LineElement objects"


def test_property_accessors_with_negative_indexing():
    """Test property-based accessors with negative indexing."""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    # Create guides with 3x3 grid
    guides = Guides(page)
    guides.vertical.divide(3)  # Creates 4 vertical guides
    guides.horizontal.divide(3)  # Creates 4 horizontal guides

    # Test positive indexing
    assert len(guides.columns) == 3  # 4 guides = 3 columns
    assert len(guides.rows) == 3  # 4 guides = 3 rows

    # Test column access with positive index
    col0 = guides.columns[0]
    col2 = guides.columns[2]
    assert col0.x0 < col2.x0  # First column is to the left of last

    # Test column access with negative index
    last_col = guides.columns[-1]
    assert last_col.x0 == col2.x0  # -1 should give us the last column
    assert last_col.x1 == col2.x1

    second_last_col = guides.columns[-2]
    col1 = guides.columns[1]
    assert second_last_col.x0 == col1.x0  # -2 should give us the middle column

    # Test row access with negative index
    last_row = guides.rows[-1]
    row2 = guides.rows[2]
    assert last_row.top == row2.top  # -1 should give us the last row
    assert last_row.bottom == row2.bottom

    first_row = guides.rows[-3]
    row0 = guides.rows[0]
    assert first_row.top == row0.top  # -3 should give us the first row (when we have 3 rows)

    # Test cell access with negative indices
    # Using tuple notation
    bottom_right = guides.cells[-1, -1]
    cell_2_2 = guides.cells[2, 2]
    assert bottom_right.x0 == cell_2_2.x0
    assert bottom_right.top == cell_2_2.top

    # Using nested notation
    top_left = guides.cells[-3][-3]
    cell_0_0 = guides.cells[0][0]
    assert top_left.x0 == cell_0_0.x0
    assert top_left.top == cell_0_0.top

    # Mixed positive and negative indices
    mixed_cell = guides.cells[0, -1]  # First row, last column
    reference = guides.cells[0][2]
    assert mixed_cell.x0 == reference.x0
    assert mixed_cell.top == reference.top

    # Test that out of bounds negative index raises IndexError
    with pytest.raises(IndexError):
        _ = guides.columns[-4]  # We only have 3 columns

    with pytest.raises(IndexError):
        _ = guides.rows[-5]  # We only have 3 rows

    with pytest.raises(IndexError):
        _ = guides.cells[-4, 0]  # Row index out of bounds

    with pytest.raises(IndexError):
        _ = guides.cells[0, -4]  # Column index out of bounds


def test_property_accessors_with_slicing():
    """Test property-based accessors with slice notation."""
    pdf = PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    # Create guides with 3x3 grid
    guides = Guides(page)
    guides.vertical.divide(3)  # Creates 4 vertical guides = 3 columns
    guides.horizontal.divide(3)  # Creates 4 horizontal guides = 3 rows

    # Test getting all cells in a row
    row_cells = guides.cells[0][:]
    assert hasattr(row_cells, "__len__")  # Should be an ElementCollection
    assert len(row_cells) == 3  # 3 cells in a row

    # Test getting all cells in a row with tuple notation
    row_cells_tuple = guides.cells[0, :]
    assert len(row_cells_tuple) == 3
    # Should contain same cells
    assert all(c1.x0 == c2.x0 and c1.top == c2.top for c1, c2 in zip(row_cells, row_cells_tuple))

    # Test getting all cells in a column
    col_cells = guides.cells[:, 0]
    assert len(col_cells) == 3  # 3 cells in a column

    # Test getting all cells
    all_cells = guides.cells[:, :]
    assert len(all_cells) == 9  # 3x3 = 9 cells

    # Test getting all rows
    all_rows = guides.rows[:]
    assert len(all_rows) == 3

    # Test getting all columns
    all_cols = guides.columns[:]
    assert len(all_cols) == 3

    # Test slice with step
    every_other_col = guides.columns[::2]
    assert len(every_other_col) == 2  # columns 0 and 2

    # Test negative indices in slices
    last_row_cells = guides.cells[-1, :]
    assert len(last_row_cells) == 3

    # Test partial slices
    first_two_rows = guides.rows[:2]
    assert len(first_two_rows) == 2

    last_two_cols = guides.columns[-2:]
    assert len(last_two_cols) == 2

    # Test that cells are in correct order
    # First row cells should be ordered left to right
    first_row = guides.cells[0, :]
    for i in range(len(first_row) - 1):
        assert first_row[i].x0 < first_row[i + 1].x0

    # First column cells should be ordered top to bottom
    first_col = guides.cells[:, 0]
    for i in range(len(first_col) - 1):
        assert first_col[i].top < first_col[i + 1].top


if __name__ == "__main__":
    pytest.main([__file__])
