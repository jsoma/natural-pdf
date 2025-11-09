#!/usr/bin/env python3
"""Test that ElementCollection.show() respects the columns parameter for multi-page collections."""

from pathlib import Path
from unittest.mock import Mock, patch

from natural_pdf import PDF
from natural_pdf.elements.element_collection import ElementCollection


def collect_real_text_elements(target_count: int, min_unique_pages: int = 1) -> list:
    """Collect TextElements from a real PDF to avoid brittle mocks."""
    pdf = PDF(Path("pdfs/multi-page-table.pdf"))
    collected = []
    per_page_limit = max(1, (target_count + min_unique_pages - 1) // max(1, min_unique_pages))

    for page in pdf:
        text_elements = list(page.find_all("text"))
        if not text_elements:
            continue
        take = min(per_page_limit, target_count - len(collected), len(text_elements))
        collected.extend(text_elements[:take])
        if (
            len(collected) >= target_count
            and len({elem.page for elem in collected}) >= min_unique_pages
        ):
            break

    unique_pages = {elem.page for elem in collected}
    if len(collected) < target_count or len(unique_pages) < min_unique_pages:
        raise RuntimeError(
            f"Unable to collect {target_count} text elements from at least {min_unique_pages} pages"
        )
    return collected[:target_count]


class TestElementCollectionShowCols:
    """Test ElementCollection.show() with columns parameter."""

    def test_show_respects_columns_parameter(self):
        """Test that show() passes the columns parameter to unified_render."""
        # Create a collection with elements from multiple pages
        elements = collect_real_text_elements(12)
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
        elements = collect_real_text_elements(6)
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
        elements = collect_real_text_elements(4, min_unique_pages=2)
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
