#!/usr/bin/env python3
"""Test that ElementCollection.show() respects the columns parameter for multi-page collections."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.elements.text import TextElement


def create_mock_element(page_index, text="Test"):
    """Create a mock TextElement with required attributes."""
    elem = Mock(spec=TextElement)
    elem.page = Mock()
    elem.page.page_index = page_index
    elem.page._highlighter = Mock()
    elem.text = text
    elem.bbox = (100, 100, 200, 200)

    # Mock the get_highlight_specs method if it exists
    if hasattr(TextElement, "get_highlight_specs"):
        elem.get_highlight_specs = Mock(return_value=None)

    return elem


class TestElementCollectionShowCols:
    """Test ElementCollection.show() with columns parameter."""

    def test_show_respects_columns_parameter(self):
        """Test that show() passes the columns parameter to unified_render."""
        # Create a collection with elements from multiple pages
        elements = [create_mock_element(i) for i in range(12)]
        collection = ElementCollection(elements)

        # Mock the highlighter and its unified_render method
        mock_highlighter = Mock()
        mock_highlighter.unified_render = Mock(return_value=Mock())  # Mock PIL Image

        # Ensure _get_highlighter returns our mock
        with patch.object(collection, "_get_highlighter", return_value=mock_highlighter):
            # Test with different column values
            test_cases = [
                (None, 6),  # Default should be 6
                (3, 3),  # Explicit 3 columns
                (4, 4),  # Explicit 4 columns
                (8, 8),  # Explicit 8 columns
            ]

            for input_cols, expected_cols in test_cases:
                mock_highlighter.unified_render.reset_mock()

                if input_cols is None:
                    collection.show()
                else:
                    collection.show(columns=input_cols)

                # Verify unified_render was called with correct columns
                assert mock_highlighter.unified_render.called
                call_kwargs = mock_highlighter.unified_render.call_args[1]
                assert (
                    call_kwargs["columns"] == expected_cols
                ), f"Expected columns={expected_cols}, got {call_kwargs.get('columns')}"

    def test_show_with_cols_parameter_works_as_alias(self):
        """Test that using 'cols' parameter works as an alias for 'columns'."""
        elements = [create_mock_element(i) for i in range(6)]
        collection = ElementCollection(elements)

        mock_highlighter = Mock()
        mock_highlighter.unified_render = Mock(return_value=Mock())

        with patch.object(collection, "_get_highlighter", return_value=mock_highlighter):
            # This should now work as an alias
            collection.show(cols=3)  # Using alias parameter name

            # Should use the cols value of 3
            call_kwargs = mock_highlighter.unified_render.call_args[1]
            assert (
                call_kwargs["columns"] == 3
            ), "Using 'cols' as alias for 'columns' should set columns to 3"

    def test_show_layout_defaults_for_multipage(self):
        """Test that multi-page collections default to grid layout."""
        elements = [create_mock_element(i) for i in range(4)]
        collection = ElementCollection(elements)

        mock_highlighter = Mock()
        mock_highlighter.unified_render = Mock(return_value=Mock())

        with patch.object(collection, "_get_highlighter", return_value=mock_highlighter):
            collection.show()

            call_kwargs = mock_highlighter.unified_render.call_args[1]
            assert call_kwargs["layout"] == "grid"
            assert call_kwargs["columns"] == 6  # Default columns


if __name__ == "__main__":
    print("=== Testing ElementCollection.show() columns parameter ===")

    test = TestElementCollectionShowCols()

    # Run tests
    test_methods = [
        ("show() respects columns parameter", test.test_show_respects_columns_parameter),
        (
            "show() with 'cols' parameter works as alias",
            test.test_show_with_cols_parameter_works_as_alias,
        ),
        ("Multi-page defaults to grid layout", test.test_show_layout_defaults_for_multipage),
    ]

    passed = 0
    failed = 0

    for desc, test_func in test_methods:
        try:
            test_func()
            print(f"✓ {desc}")
            passed += 1
        except Exception as e:
            print(f"✗ {desc}: {e}")
            failed += 1

    print(f"\n=== Results: {passed} passed, {failed} failed ===")

    if failed == 0:
        print("\n✅ Both 'columns' and 'cols' parameters now work!")
        print("    collection.show(columns=3)  # Original parameter")
        print("    collection.show(cols=3)     # Alias for convenience")
