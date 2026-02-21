"""Test the :empty and :not-empty pseudo-class selectors with various whitespace scenarios."""

from unittest.mock import Mock

import pytest

from natural_pdf.selectors.parser import _build_filter_list, parse_selector
from natural_pdf.selectors.registry import get_pseudo_handler


def _run_pseudo(name, element):
    """Helper to run a registered pseudo-class handler against an element."""
    handler = get_pseudo_handler(name)
    assert handler is not None, f"Pseudo-class :{name} not registered"
    from natural_pdf.selectors.registry import ClauseEvalContext

    ctx = ClauseEvalContext(selector_context=None, aggregates={}, options={})
    result = handler({"name": name, "args": None}, ctx)
    assert result is not None, f"Handler for :{name} returned None"
    return result["func"](element)


class TestEmptyPseudoClass:
    """Test cases for :empty and :not-empty pseudo-class selectors."""

    def test_empty_with_none_text(self):
        element = Mock()
        element.text = None
        assert _run_pseudo("empty", element) is True
        assert _run_pseudo("not-empty", element) is False

    def test_empty_with_empty_string(self):
        element = Mock()
        element.text = ""
        assert _run_pseudo("empty", element) is True
        assert _run_pseudo("not-empty", element) is False

    def test_empty_with_single_space(self):
        element = Mock()
        element.text = " "
        assert _run_pseudo("empty", element) is True
        assert _run_pseudo("not-empty", element) is False

    def test_empty_with_multiple_spaces(self):
        element = Mock()
        element.text = "     "
        assert _run_pseudo("empty", element) is True
        assert _run_pseudo("not-empty", element) is False

    def test_empty_with_tabs(self):
        element = Mock()
        element.text = "\t\t\t"
        assert _run_pseudo("empty", element) is True
        assert _run_pseudo("not-empty", element) is False

    def test_empty_with_newlines(self):
        element = Mock()
        element.text = "\n\n\n"
        assert _run_pseudo("empty", element) is True
        assert _run_pseudo("not-empty", element) is False

    def test_empty_with_mixed_whitespace(self):
        element = Mock()
        element.text = "   \n\t  \n   "
        assert _run_pseudo("empty", element) is True
        assert _run_pseudo("not-empty", element) is False

    def test_not_empty_with_content(self):
        element = Mock()
        element.text = "Hello"
        assert _run_pseudo("empty", element) is False
        assert _run_pseudo("not-empty", element) is True

    def test_not_empty_with_content_and_leading_whitespace(self):
        element = Mock()
        element.text = "   Hello"
        assert _run_pseudo("empty", element) is False
        assert _run_pseudo("not-empty", element) is True

    def test_not_empty_with_content_and_trailing_whitespace(self):
        element = Mock()
        element.text = "Hello   "
        assert _run_pseudo("empty", element) is False
        assert _run_pseudo("not-empty", element) is True

    def test_not_empty_with_content_and_surrounding_whitespace(self):
        element = Mock()
        element.text = "   Hello   "
        assert _run_pseudo("empty", element) is False
        assert _run_pseudo("not-empty", element) is True

    def test_not_empty_with_content_and_internal_whitespace(self):
        element = Mock()
        element.text = "Hello World"
        assert _run_pseudo("empty", element) is False
        assert _run_pseudo("not-empty", element) is True

    def test_not_empty_with_single_character(self):
        element = Mock()
        element.text = "a"
        assert _run_pseudo("empty", element) is False
        assert _run_pseudo("not-empty", element) is True

    def test_not_empty_with_punctuation(self):
        element = Mock()
        element.text = "."
        assert _run_pseudo("empty", element) is False
        assert _run_pseudo("not-empty", element) is True

    def test_not_empty_with_number(self):
        element = Mock()
        element.text = "0"
        assert _run_pseudo("empty", element) is False
        assert _run_pseudo("not-empty", element) is True


class TestEmptyPseudoClassEdgeCases:
    """Edge case tests for :empty and :not-empty pseudo-class selectors."""

    def test_empty_with_no_text_attribute(self):
        element = Mock(spec=[])
        try:
            result = _run_pseudo("empty", element)
            assert result is True
        except AttributeError:
            pytest.fail("empty pseudo-class should handle missing text attribute")

    def test_not_empty_with_no_text_attribute(self):
        element = Mock(spec=[])
        try:
            result = _run_pseudo("not-empty", element)
            assert result is False
        except AttributeError:
            pytest.fail("not-empty pseudo-class should handle missing text attribute")


class TestEmptyPseudoClassIntegration:
    """Integration tests for :empty and :not-empty pseudo-class selectors."""

    def test_parse_and_filter_empty(self):
        parsed = parse_selector("text:empty")
        filters, _, _ = _build_filter_list(parsed)

        pseudo_filter = None
        for f in filters:
            if "pseudo-class" in f["name"]:
                pseudo_filter = f["func"]
                break

        assert pseudo_filter is not None

        empty_element = Mock()
        empty_element.text = "   "
        assert pseudo_filter(empty_element) is True

        not_empty_element = Mock()
        not_empty_element.text = "Hello"
        assert pseudo_filter(not_empty_element) is False

    def test_parse_and_filter_not_empty(self):
        parsed = parse_selector("text:not-empty")
        filters, _, _ = _build_filter_list(parsed)

        pseudo_filter = None
        for f in filters:
            if "pseudo-class" in f["name"]:
                pseudo_filter = f["func"]
                break

        assert pseudo_filter is not None

        empty_element = Mock()
        empty_element.text = "   "
        assert pseudo_filter(empty_element) is False

        not_empty_element = Mock()
        not_empty_element.text = "Hello"
        assert pseudo_filter(not_empty_element) is True
