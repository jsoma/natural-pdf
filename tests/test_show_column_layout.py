#!/usr/bin/env python3
"""Test that show() method defaults to 6 columns and handles columns parameter correctly."""

from unittest.mock import Mock, patch

import pytest

from natural_pdf.core.render_spec import Visualizable


class MockMultiPageObject(Visualizable):
    """Mock object that simulates a multi-page PDF or collection."""

    def __init__(self, num_pages=4):
        self.num_pages = num_pages

    def _get_render_specs(self, **kwargs):
        """Return mock specs for each page."""
        return [{"page_index": i, "highlights": []} for i in range(self.num_pages)]

    def _get_highlighter(self):
        """Return mock highlighter."""
        mock_highlighter = Mock()
        mock_highlighter.unified_render.return_value = Mock()  # Mock PIL Image
        return mock_highlighter


def test_single_page_defaults_to_single_layout():
    """Test that single page objects use 'single' layout."""
    obj = MockMultiPageObject(num_pages=1)

    with patch.object(obj, "_get_highlighter") as mock_get_highlighter:
        mock_highlighter = Mock()
        mock_get_highlighter.return_value = mock_highlighter

        obj.show()

        # Check that unified_render was called with layout='single'
        assert mock_highlighter.unified_render.called
        call_kwargs = mock_highlighter.unified_render.call_args[1]
        assert call_kwargs["layout"] == "single"


def test_multipage_defaults_to_grid_layout():
    """Test that multi-page objects use 'grid' layout by default."""
    obj = MockMultiPageObject(num_pages=4)

    with patch.object(obj, "_get_highlighter") as mock_get_highlighter:
        mock_highlighter = Mock()
        mock_get_highlighter.return_value = mock_highlighter

        obj.show()

        # Check that unified_render was called with layout='grid'
        assert mock_highlighter.unified_render.called
        call_kwargs = mock_highlighter.unified_render.call_args[1]
        assert call_kwargs["layout"] == "grid"


def test_default_columns_is_six():
    """Test that the default columns parameter is 6."""
    obj = MockMultiPageObject(num_pages=8)

    with patch.object(obj, "_get_highlighter") as mock_get_highlighter:
        mock_highlighter = Mock()
        mock_get_highlighter.return_value = mock_highlighter

        obj.show()

        # Check that unified_render was called with columns=6
        call_kwargs = mock_highlighter.unified_render.call_args[1]
        assert call_kwargs["columns"] == 6


def test_columns_parameter_is_respected():
    """Test that explicit columns parameter is passed through."""
    obj = MockMultiPageObject(num_pages=6)

    with patch.object(obj, "_get_highlighter") as mock_get_highlighter:
        mock_highlighter = Mock()
        mock_get_highlighter.return_value = mock_highlighter

        obj.show(columns=3)

        # Check that unified_render was called with columns=3
        call_kwargs = mock_highlighter.unified_render.call_args[1]
        assert call_kwargs["columns"] == 3
        assert call_kwargs["layout"] == "grid"


def test_explicit_layout_parameter_overrides_default():
    """Test that explicit layout parameter overrides the automatic detection."""
    obj = MockMultiPageObject(num_pages=4)

    with patch.object(obj, "_get_highlighter") as mock_get_highlighter:
        mock_highlighter = Mock()
        mock_get_highlighter.return_value = mock_highlighter

        obj.show(layout="stack")

        # Check that unified_render was called with layout='stack'
        call_kwargs = mock_highlighter.unified_render.call_args[1]
        assert call_kwargs["layout"] == "stack"


def test_explicit_layout_none_uses_auto_detection():
    """Test that layout=None uses automatic detection logic."""
    obj = MockMultiPageObject(num_pages=3)

    with patch.object(obj, "_get_highlighter") as mock_get_highlighter:
        mock_highlighter = Mock()
        mock_get_highlighter.return_value = mock_highlighter

        obj.show(layout=None)

        # Check that unified_render was called with layout='grid' (auto-detected)
        call_kwargs = mock_highlighter.unified_render.call_args[1]
        assert call_kwargs["layout"] == "grid"


def test_columns_with_explicit_grid_layout():
    """Test columns parameter works with explicit grid layout."""
    obj = MockMultiPageObject(num_pages=5)

    with patch.object(obj, "_get_highlighter") as mock_get_highlighter:
        mock_highlighter = Mock()
        mock_get_highlighter.return_value = mock_highlighter

        obj.show(layout="grid", columns=2)

        # Check parameters
        call_kwargs = mock_highlighter.unified_render.call_args[1]
        assert call_kwargs["layout"] == "grid"
        assert call_kwargs["columns"] == 2


def test_no_specs_returns_none():
    """Test that show() returns None when no render specs are generated."""
    obj = MockMultiPageObject(num_pages=0)

    # Override to return empty specs
    with patch.object(obj, "_get_render_specs", return_value=[]):
        result = obj.show()
        assert result is None


if __name__ == "__main__":
    print("=== Running show() column layout tests ===")

    test_cases = [
        ("Single page → single layout", test_single_page_defaults_to_single_layout),
        ("Multi-page → grid layout", test_multipage_defaults_to_grid_layout),
        ("Default columns = 6", test_default_columns_is_six),
        ("Explicit columns respected", test_columns_parameter_is_respected),
        ("Explicit layout overrides default", test_explicit_layout_parameter_overrides_default),
        ("Layout=None uses auto-detection", test_explicit_layout_none_uses_auto_detection),
        ("Columns with grid layout", test_columns_with_explicit_grid_layout),
        ("No specs returns None", test_no_specs_returns_none),
    ]

    passed = 0
    failed = 0

    for desc, test_func in test_cases:
        try:
            test_func()
            print(f"✓ {desc}")
            passed += 1
        except Exception as e:
            print(f"✗ {desc}: {e}")
            failed += 1

    print(f"\n=== Results: {passed} passed, {failed} failed ===")

    if failed == 0:
        print("🎉 All tests passed! The show() method now:")
        print("   • Defaults to grid layout with 6 columns for multi-page content")
        print("   • Uses single layout for single pages")
        print("   • Respects explicit columns parameter")
        print("   • Maintains backward compatibility")
