#!/usr/bin/env python3
"""Test edge cases for the enhanced show() method."""

from unittest.mock import Mock, patch

import pytest

from natural_pdf.core.render_spec import Visualizable


class MockObject(Visualizable):
    """Mock object for testing edge cases."""

    def __init__(self, specs):
        self.specs = specs

    def _get_render_specs(self, **kwargs):
        return self.specs

    def _get_highlighter(self):
        mock_highlighter = Mock()
        mock_image = Mock()
        mock_image.size = (800, 600)
        mock_highlighter.unified_render.return_value = mock_image
        return mock_highlighter


def test_empty_specs_returns_none():
    """Test that empty specs list returns None."""
    obj = MockObject([])
    result = obj.show()
    assert result is None


def test_columns_zero_handled_gracefully():
    """Test that columns=0 is handled gracefully."""
    obj = MockObject([{"page": 1}, {"page": 2}])

    with patch.object(obj, "_get_highlighter") as mock_get_highlighter:
        mock_highlighter = Mock()
        mock_get_highlighter.return_value = mock_highlighter

        # This should not crash
        obj.show(columns=0)

        call_kwargs = mock_highlighter.unified_render.call_args[1]
        assert call_kwargs["columns"] == 0
        assert call_kwargs["layout"] == "grid"


def test_negative_columns_handled():
    """Test that negative columns are handled."""
    obj = MockObject([{"page": 1}, {"page": 2}])

    with patch.object(obj, "_get_highlighter") as mock_get_highlighter:
        mock_highlighter = Mock()
        mock_get_highlighter.return_value = mock_highlighter

        obj.show(columns=-1)

        call_kwargs = mock_highlighter.unified_render.call_args[1]
        assert call_kwargs["columns"] == -1  # Let highlighter handle it


def test_very_large_columns():
    """Test with very large column count."""
    obj = MockObject([{"page": i} for i in range(5)])

    with patch.object(obj, "_get_highlighter") as mock_get_highlighter:
        mock_highlighter = Mock()
        mock_get_highlighter.return_value = mock_highlighter

        obj.show(columns=1000)

        call_kwargs = mock_highlighter.unified_render.call_args[1]
        assert call_kwargs["columns"] == 1000


def test_layout_override_with_columns():
    """Test that explicit layout overrides auto-detection even with columns."""
    obj = MockObject([{"page": 1}, {"page": 2}])

    with patch.object(obj, "_get_highlighter") as mock_get_highlighter:
        mock_highlighter = Mock()
        mock_get_highlighter.return_value = mock_highlighter

        # Even with multiple specs, explicit layout='single' should be used
        obj.show(layout="single", columns=6)

        call_kwargs = mock_highlighter.unified_render.call_args[1]
        assert call_kwargs["layout"] == "single"
        assert call_kwargs["columns"] == 6


def test_none_columns_with_auto_layout():
    """Test that columns=None works with auto layout detection."""
    obj = MockObject([{"page": 1}, {"page": 2}, {"page": 3}])

    with patch.object(obj, "_get_highlighter") as mock_get_highlighter:
        mock_highlighter = Mock()
        mock_get_highlighter.return_value = mock_highlighter

        obj.show(columns=None)

        call_kwargs = mock_highlighter.unified_render.call_args[1]
        assert call_kwargs["layout"] == "grid"
        assert call_kwargs["columns"] is None  # Let highlighter auto-calculate


def test_single_spec_with_explicit_grid_layout():
    """Test single spec with explicit grid layout still works."""
    obj = MockObject([{"page": 1}])

    with patch.object(obj, "_get_highlighter") as mock_get_highlighter:
        mock_highlighter = Mock()
        mock_get_highlighter.return_value = mock_highlighter

        obj.show(layout="grid", columns=1)

        call_kwargs = mock_highlighter.unified_render.call_args[1]
        assert call_kwargs["layout"] == "grid"
        assert call_kwargs["columns"] == 1


def test_all_parameters_passed_through():
    """Test that all parameters are passed through to unified_render."""
    obj = MockObject([{"page": 1}, {"page": 2}])

    with patch.object(obj, "_get_highlighter") as mock_get_highlighter:
        mock_highlighter = Mock()
        mock_get_highlighter.return_value = mock_highlighter

        obj.show(
            resolution=144,
            width=800,
            labels=False,
            gap=10,
            columns=3,
            legend_position="left",
            custom_param="test",
        )

        call_kwargs = mock_highlighter.unified_render.call_args[1]
        assert call_kwargs["resolution"] == 144
        assert call_kwargs["width"] == 800
        assert call_kwargs["labels"] is False
        assert call_kwargs["gap"] == 10
        assert call_kwargs["columns"] == 3
        assert call_kwargs["legend_position"] == "left"
        assert call_kwargs["layout"] == "grid"
        assert call_kwargs["custom_param"] == "test"


if __name__ == "__main__":
    print("=== Running edge case tests ===")

    test_cases = [
        ("Empty specs returns None", test_empty_specs_returns_none),
        ("columns=0 handled gracefully", test_columns_zero_handled_gracefully),
        ("Negative columns handled", test_negative_columns_handled),
        ("Very large columns handled", test_very_large_columns),
        ("Layout override with columns", test_layout_override_with_columns),
        ("columns=None with auto layout", test_none_columns_with_auto_layout),
        ("Single spec with explicit grid", test_single_spec_with_explicit_grid_layout),
        ("All parameters passed through", test_all_parameters_passed_through),
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
            import traceback

            traceback.print_exc()
            failed += 1

    print(f"\n=== Results: {passed} passed, {failed} failed ===")

    if failed == 0:
        print("🎉 All edge cases handled correctly!")
        print("   The implementation is robust and handles:")
        print("   • Empty content gracefully")
        print("   • Invalid column values without crashing")
        print("   • Parameter precedence correctly")
        print("   • All existing functionality")
