# tests/test_selector_cache.py
"""
Tests for selector parsing cache and OR deduplication.

Tests cover:
- LRU cache functionality
- OR selector deduplication
- Cache isolation (mutations don't affect cached values)
"""

import pytest

from natural_pdf.selectors.parser import (
    clear_selector_cache,
    get_selector_cache_info,
    parse_selector,
)


class TestSelectorCache:
    """Tests for selector parsing cache."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_selector_cache()

    def test_cache_returns_correct_result(self):
        """Cache should return correct parsed results."""
        result = parse_selector('text:contains("hello")')
        assert result["type"] == "text"
        assert len(result["pseudo_classes"]) == 1
        assert result["pseudo_classes"][0]["name"] == "contains"
        assert result["pseudo_classes"][0]["args"] == "hello"

    def test_cache_hit_on_repeated_calls(self):
        """Repeated calls with same selector should hit cache."""
        clear_selector_cache()

        # First call - cache miss
        parse_selector('text:contains("test")')
        info1 = get_selector_cache_info()
        assert info1.misses >= 1

        # Second call - cache hit
        parse_selector('text:contains("test")')
        info2 = get_selector_cache_info()
        assert info2.hits >= 1

    def test_cache_returns_same_object(self):
        """Cache should return the same object (no wasteful deep copy)."""
        result1 = parse_selector("text:bold")
        result2 = parse_selector("text:bold")
        # Same cached object — no deep copy overhead
        assert result1 is result2

    def test_build_filter_list_does_not_mutate_selector(self):
        """build_execution_plan should not mutate the parsed selector dict."""
        from natural_pdf.selectors.parser import build_execution_plan

        parsed = parse_selector("text:bold:first:above(rect)")
        original_pseudo_count = len(parsed["pseudo_classes"])
        original_keys = set(parsed.keys())

        # Build execution plan — should NOT add post_pseudos/relational_pseudos to dict
        build_execution_plan(parsed)

        assert set(parsed.keys()) == original_keys
        assert len(parsed["pseudo_classes"]) == original_pseudo_count
        assert "post_pseudos" not in parsed
        assert "relational_pseudos" not in parsed

    def test_cache_clear(self):
        """clear_selector_cache should reset the cache."""
        # Populate cache
        parse_selector("text:first")
        parse_selector("rect:last")
        info1 = get_selector_cache_info()
        assert info1.currsize >= 2

        # Clear cache
        clear_selector_cache()
        info2 = get_selector_cache_info()
        assert info2.currsize == 0

    def test_cache_info_returns_valid_data(self):
        """get_selector_cache_info should return valid statistics."""
        clear_selector_cache()

        # Generate some cache activity
        parse_selector("text:bold")
        parse_selector("text:bold")  # hit
        parse_selector("rect[width>10]")  # miss

        info = get_selector_cache_info()
        assert hasattr(info, "hits")
        assert hasattr(info, "misses")
        assert hasattr(info, "maxsize")
        assert hasattr(info, "currsize")
        assert info.hits >= 1
        assert info.misses >= 2


class TestORDeduplication:
    """Tests for OR selector deduplication."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_selector_cache()

    def test_duplicate_or_parts_removed(self):
        """Duplicate OR parts should be removed."""
        # Same selector repeated
        result = parse_selector("text:bold|text:bold|text:bold")

        # Should be simplified to single selector (not OR)
        assert result["type"] == "text"
        assert any(p["name"] == "bold" for p in result["pseudo_classes"])

    def test_duplicate_or_parts_preserves_order(self):
        """Deduplication should preserve first occurrence order."""
        result = parse_selector("text:bold|rect:first|text:bold|line")

        # Should have 3 unique selectors in order
        assert result["type"] == "or"
        assert len(result["selectors"]) == 3
        assert result["selectors"][0]["type"] == "text"
        assert result["selectors"][1]["type"] == "rect"
        assert result["selectors"][2]["type"] == "line"

    def test_no_deduplication_when_different(self):
        """Different selectors should all be kept."""
        result = parse_selector('text:bold|text:italic|text:contains("x")')

        assert result["type"] == "or"
        assert len(result["selectors"]) == 3

    def test_deduplication_with_complex_selectors(self):
        """Complex duplicate selectors should be deduplicated."""
        selector = 'text[size>12]:contains("Total")|rect[width>100]|text[size>12]:contains("Total")'
        result = parse_selector(selector)

        assert result["type"] == "or"
        assert len(result["selectors"]) == 2

    def test_single_selector_after_deduplication(self):
        """If all OR parts are duplicates, return single selector."""
        result = parse_selector("text:bold,text:bold")

        # Should return single selector, not OR compound
        assert result["type"] == "text"


class TestCacheWithORSelectors:
    """Tests for cache behavior with OR selectors."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_selector_cache()

    def test_or_parts_cached_individually(self):
        """Individual OR parts should benefit from caching."""
        clear_selector_cache()

        # Parse OR selector
        parse_selector("text:bold|rect:first")

        # Now parse individual parts - should be cache hits
        info_before = get_selector_cache_info()
        parse_selector("text:bold")
        parse_selector("rect:first")
        info_after = get_selector_cache_info()

        # Should have cache hits for the individual parts
        assert info_after.hits > info_before.hits

    def test_nested_or_caching(self):
        """Nested/recursive OR parsing should use cache efficiently."""
        clear_selector_cache()

        # Parse a complex OR selector multiple times
        selector = "text:bold|rect:first|line[width>5]"
        parse_selector(selector)
        parse_selector(selector)
        parse_selector(selector)

        info = get_selector_cache_info()
        # Should have hits from repeated calls
        assert info.hits >= 2


class TestEdgeCases:
    """Tests for edge cases in caching and deduplication."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_selector_cache()

    def test_empty_selector(self):
        """Empty selector should be handled correctly."""
        result = parse_selector("")
        assert result["type"] == "any"

    def test_whitespace_only_selector(self):
        """Whitespace-only selector should be handled correctly."""
        result = parse_selector("   ")
        assert result["type"] == "any"

    def test_wildcard_selector(self):
        """Wildcard selector should be cached correctly."""
        result1 = parse_selector("*")
        result2 = parse_selector("*")
        assert result1 == result2

    def test_case_sensitivity_in_cache(self):
        """Cache should be case-sensitive for selector strings."""
        clear_selector_cache()

        result1 = parse_selector("TEXT:bold")
        result2 = parse_selector("text:bold")

        # These are different cache entries
        info = get_selector_cache_info()
        assert info.currsize >= 2

        # But both should parse to lowercase type
        assert result1["type"] == "text"
        assert result2["type"] == "text"
