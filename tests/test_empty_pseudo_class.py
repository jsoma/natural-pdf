"""Test the :empty and :not-empty pseudo-class selectors with various whitespace scenarios."""

from unittest.mock import Mock

import pytest

from natural_pdf.selectors.parser import PSEUDO_CLASS_FUNCTIONS


class TestEmptyPseudoClass:
    """Test cases for :empty and :not-empty pseudo-class selectors."""

    def test_empty_with_none_text(self):
        """Test that None text is considered empty."""
        element = Mock()
        element.text = None

        assert PSEUDO_CLASS_FUNCTIONS["empty"](element) is True
        assert PSEUDO_CLASS_FUNCTIONS["not-empty"](element) is False

    def test_empty_with_empty_string(self):
        """Test that empty string is considered empty."""
        element = Mock()
        element.text = ""

        assert PSEUDO_CLASS_FUNCTIONS["empty"](element) is True
        assert PSEUDO_CLASS_FUNCTIONS["not-empty"](element) is False

    def test_empty_with_single_space(self):
        """Test that single space is considered empty."""
        element = Mock()
        element.text = " "

        assert PSEUDO_CLASS_FUNCTIONS["empty"](element) is True
        assert PSEUDO_CLASS_FUNCTIONS["not-empty"](element) is False

    def test_empty_with_multiple_spaces(self):
        """Test that multiple spaces are considered empty."""
        element = Mock()
        element.text = "     "

        assert PSEUDO_CLASS_FUNCTIONS["empty"](element) is True
        assert PSEUDO_CLASS_FUNCTIONS["not-empty"](element) is False

    def test_empty_with_tabs(self):
        """Test that tabs are considered empty."""
        element = Mock()
        element.text = "\t\t\t"

        assert PSEUDO_CLASS_FUNCTIONS["empty"](element) is True
        assert PSEUDO_CLASS_FUNCTIONS["not-empty"](element) is False

    def test_empty_with_newlines(self):
        """Test that newlines are considered empty."""
        element = Mock()
        element.text = "\n\n\n"

        assert PSEUDO_CLASS_FUNCTIONS["empty"](element) is True
        assert PSEUDO_CLASS_FUNCTIONS["not-empty"](element) is False

    def test_empty_with_mixed_whitespace(self):
        """Test that mixed whitespace (spaces, tabs, newlines) is considered empty."""
        element = Mock()
        element.text = "   \n\t  \n   "

        assert PSEUDO_CLASS_FUNCTIONS["empty"](element) is True
        assert PSEUDO_CLASS_FUNCTIONS["not-empty"](element) is False

    def test_not_empty_with_content(self):
        """Test that actual content is considered not empty."""
        element = Mock()
        element.text = "Hello"

        assert PSEUDO_CLASS_FUNCTIONS["empty"](element) is False
        assert PSEUDO_CLASS_FUNCTIONS["not-empty"](element) is True

    def test_not_empty_with_content_and_leading_whitespace(self):
        """Test that content with leading whitespace is considered not empty."""
        element = Mock()
        element.text = "   Hello"

        assert PSEUDO_CLASS_FUNCTIONS["empty"](element) is False
        assert PSEUDO_CLASS_FUNCTIONS["not-empty"](element) is True

    def test_not_empty_with_content_and_trailing_whitespace(self):
        """Test that content with trailing whitespace is considered not empty."""
        element = Mock()
        element.text = "Hello   "

        assert PSEUDO_CLASS_FUNCTIONS["empty"](element) is False
        assert PSEUDO_CLASS_FUNCTIONS["not-empty"](element) is True

    def test_not_empty_with_content_and_surrounding_whitespace(self):
        """Test that content with surrounding whitespace is considered not empty."""
        element = Mock()
        element.text = "   Hello   "

        assert PSEUDO_CLASS_FUNCTIONS["empty"](element) is False
        assert PSEUDO_CLASS_FUNCTIONS["not-empty"](element) is True

    def test_not_empty_with_content_and_internal_whitespace(self):
        """Test that content with internal whitespace is considered not empty."""
        element = Mock()
        element.text = "Hello World"

        assert PSEUDO_CLASS_FUNCTIONS["empty"](element) is False
        assert PSEUDO_CLASS_FUNCTIONS["not-empty"](element) is True

    def test_not_empty_with_single_character(self):
        """Test that a single non-whitespace character is considered not empty."""
        element = Mock()
        element.text = "a"

        assert PSEUDO_CLASS_FUNCTIONS["empty"](element) is False
        assert PSEUDO_CLASS_FUNCTIONS["not-empty"](element) is True

    def test_not_empty_with_punctuation(self):
        """Test that punctuation is considered not empty."""
        element = Mock()
        element.text = "."

        assert PSEUDO_CLASS_FUNCTIONS["empty"](element) is False
        assert PSEUDO_CLASS_FUNCTIONS["not-empty"](element) is True

    def test_not_empty_with_number(self):
        """Test that numbers are considered not empty."""
        element = Mock()
        element.text = "0"

        assert PSEUDO_CLASS_FUNCTIONS["empty"](element) is False
        assert PSEUDO_CLASS_FUNCTIONS["not-empty"](element) is True


class TestEmptyPseudoClassEdgeCases:
    """Edge case tests for :empty and :not-empty pseudo-class selectors."""

    def test_empty_with_no_text_attribute(self):
        """Test that elements without text attribute are handled gracefully."""
        element = Mock(spec=[])  # No text attribute

        # Should not raise AttributeError
        try:
            result = PSEUDO_CLASS_FUNCTIONS["empty"](element)
            # Elements without text attribute should be considered empty
            assert result is True
        except AttributeError:
            pytest.fail("empty pseudo-class should handle missing text attribute")

    def test_not_empty_with_no_text_attribute(self):
        """Test that elements without text attribute are handled gracefully."""
        element = Mock(spec=[])  # No text attribute

        # Should not raise AttributeError
        try:
            result = PSEUDO_CLASS_FUNCTIONS["not-empty"](element)
            # Elements without text attribute should be considered empty (not not-empty)
            assert result is False
        except AttributeError:
            pytest.fail("not-empty pseudo-class should handle missing text attribute")


class TestEmptyPseudoClassIntegration:
    """Integration tests for :empty and :not-empty pseudo-class selectors."""

    def test_parse_and_filter_empty(self):
        """Test that :empty selector works correctly in parsing context."""
        from natural_pdf.selectors.parser import _build_filter_list, parse_selector

        # Parse the :empty selector
        parsed = parse_selector("text:empty")
        filters = _build_filter_list(parsed)

        # Find the pseudo-class filter
        pseudo_filter = None
        for f in filters:
            if "pseudo-class" in f["name"]:
                pseudo_filter = f["func"]
                break

        assert pseudo_filter is not None

        # Test with various elements
        empty_element = Mock()
        empty_element.text = "   "
        assert pseudo_filter(empty_element) is True

        not_empty_element = Mock()
        not_empty_element.text = "Hello"
        assert pseudo_filter(not_empty_element) is False

    def test_parse_and_filter_not_empty(self):
        """Test that :not-empty selector works correctly in parsing context."""
        from natural_pdf.selectors.parser import _build_filter_list, parse_selector

        # Parse the :not-empty selector
        parsed = parse_selector("text:not-empty")
        filters = _build_filter_list(parsed)

        # Find the pseudo-class filter
        pseudo_filter = None
        for f in filters:
            if "pseudo-class" in f["name"]:
                pseudo_filter = f["func"]
                break

        assert pseudo_filter is not None

        # Test with various elements
        empty_element = Mock()
        empty_element.text = "   "
        assert pseudo_filter(empty_element) is False

        not_empty_element = Mock()
        not_empty_element.text = "Hello"
        assert pseudo_filter(not_empty_element) is True
