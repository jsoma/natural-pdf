"""Test `:first` and `:last` pseudo-selectors."""

import pytest

from natural_pdf.selectors.parser import parse_selector


def test_parse_first_pseudo_selector():
    """Test that :first pseudo-selector is parsed correctly."""
    selector = parse_selector("text:first")

    assert selector["type"] == "text"
    pseudo_classes = selector["pseudo_classes"]
    assert len(pseudo_classes) == 1
    assert pseudo_classes[0]["name"] == "first"
    assert pseudo_classes[0]["args"] is None


def test_parse_last_pseudo_selector():
    """Test that :last pseudo-selector is parsed correctly."""
    selector = parse_selector("rect:last")

    assert selector["type"] == "rect"
    pseudo_classes = selector["pseudo_classes"]
    assert len(pseudo_classes) == 1
    assert pseudo_classes[0]["name"] == "last"
    assert pseudo_classes[0]["args"] is None


def test_parse_combined_selectors():
    """Test combined selectors with :first and :last."""
    selector = parse_selector('text:contains("hello"):first')

    assert selector["type"] == "text"
    pseudo_classes = selector["pseudo_classes"]
    assert len(pseudo_classes) == 2

    # Should have both :contains and :first
    names = [p["name"] for p in pseudo_classes]
    assert "contains" in names
    assert "first" in names

    # Check :contains has args
    contains_pseudo = next(p for p in pseudo_classes if p["name"] == "contains")
    assert contains_pseudo["args"] == "hello"

    # Check :first has no args
    first_pseudo = next(p for p in pseudo_classes if p["name"] == "first")
    assert first_pseudo["args"] is None


def test_parse_attribute_with_first():
    """Test attribute selectors combined with :first."""
    selector = parse_selector("text[size>12]:first")

    assert selector["type"] == "text"
    assert len(selector["attributes"]) == 1
    assert selector["attributes"][0]["name"] == "size"
    assert selector["attributes"][0]["op"] == ">"
    assert selector["attributes"][0]["value"] == 12

    pseudo_classes = selector["pseudo_classes"]
    assert len(pseudo_classes) == 1
    assert pseudo_classes[0]["name"] == "first"


def test_parse_or_selector_with_first():
    """Test OR selectors with :first pseudo-selector."""
    selector = parse_selector("text:first,rect:last")

    assert selector["type"] == "or"
    sub_selectors = selector["selectors"]
    assert len(sub_selectors) == 2

    # First sub-selector: text:first
    first_sub = sub_selectors[0]
    assert first_sub["type"] == "text"
    assert len(first_sub["pseudo_classes"]) == 1
    assert first_sub["pseudo_classes"][0]["name"] == "first"

    # Second sub-selector: rect:last
    second_sub = sub_selectors[1]
    assert second_sub["type"] == "rect"
    assert len(second_sub["pseudo_classes"]) == 1
    assert second_sub["pseudo_classes"][0]["name"] == "last"


def test_wildcard_with_first():
    """Test wildcard selector with :first."""
    selector = parse_selector("*:first")

    assert selector["type"] == "any"
    pseudo_classes = selector["pseudo_classes"]
    assert len(pseudo_classes) == 1
    assert pseudo_classes[0]["name"] == "first"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
