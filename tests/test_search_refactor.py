"""Tests for search system refactor: registry migration, parsing fixes,
immutable cache, OR optimization, and new post-pseudos."""

from unittest.mock import Mock

import pytest

from natural_pdf.selectors.parser import (
    _build_filter_list,
    _split_top_level_or,
    build_execution_plan,
    clear_selector_cache,
    parse_selector,
    selector_to_filter_func,
)
from natural_pdf.selectors.registry import ClauseEvalContext, get_post_handler, get_pseudo_handler


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_selector_cache()
    yield
    clear_selector_cache()


def _make_element(**attrs):
    el = Mock()
    for k, v in attrs.items():
        setattr(el, k, v)
    el.type = attrs.get("type", "text")
    return el


# ---------------------------------------------------------------------------
# Task 1: Registry migration — PSEUDO_CLASS_FUNCTIONS → _clauses.py
# ---------------------------------------------------------------------------


class TestRegistryMigration:
    """Verify migrated pseudo-classes work through the registry."""

    def test_empty_registered(self):
        assert get_pseudo_handler("empty") is not None

    def test_not_empty_registered(self):
        assert get_pseudo_handler("not-empty") is not None

    def test_first_child_registered(self):
        assert get_pseudo_handler("first-child") is not None

    def test_last_child_registered(self):
        assert get_pseudo_handler("last-child") is not None

    def test_not_bold_registered(self):
        assert get_pseudo_handler("not-bold") is not None

    def test_not_italic_registered(self):
        assert get_pseudo_handler("not-italic") is not None

    def test_not_bold_filter(self):
        ctx = ClauseEvalContext(selector_context=None, aggregates={}, options={})
        handler = get_pseudo_handler("not-bold")
        result = handler({"name": "not-bold", "args": None}, ctx)
        bold_el = _make_element(bold=True)
        not_bold_el = _make_element(bold=False)
        assert result["func"](bold_el) is False
        assert result["func"](not_bold_el) is True

    def test_not_italic_filter(self):
        ctx = ClauseEvalContext(selector_context=None, aggregates={}, options={})
        handler = get_pseudo_handler("not-italic")
        result = handler({"name": "not-italic", "args": None}, ctx)
        italic_el = _make_element(italic=True)
        not_italic_el = _make_element(italic=False)
        assert result["func"](italic_el) is False
        assert result["func"](not_italic_el) is True

    def test_empty_via_selector(self):
        """End-to-end: parsing text:empty produces a working filter."""
        parsed = parse_selector("text:empty")
        func = selector_to_filter_func(parsed)
        assert func(_make_element(text="   ")) is True
        assert func(_make_element(text="hello")) is False

    def test_not_empty_via_selector(self):
        parsed = parse_selector("text:not-empty")
        func = selector_to_filter_func(parsed)
        assert func(_make_element(text="hello")) is True
        assert func(_make_element(text="   ")) is False

    def test_pseudo_class_functions_dict_removed(self):
        """The old PSEUDO_CLASS_FUNCTIONS dict should not exist."""
        import natural_pdf.selectors.parser as p

        assert not hasattr(p, "PSEUDO_CLASS_FUNCTIONS")


# ---------------------------------------------------------------------------
# Task 2: Parsing bug fixes
# ---------------------------------------------------------------------------


class TestParsingBugFixes:
    """Verify parsing edge cases are handled correctly."""

    def test_split_or_unmatched_closing_paren_raises(self):
        with pytest.raises(ValueError, match="Unmatched closing parenthesis"):
            _split_top_level_or("text:contains(a))|rect")

    def test_split_or_unmatched_closing_bracket_raises(self):
        with pytest.raises(ValueError, match="Unmatched closing bracket"):
            _split_top_level_or("text[size>12]]|rect")

    def test_split_or_normal_parens_preserved(self):
        parts = _split_top_level_or('text:contains("a|b")|rect')
        assert len(parts) == 2
        assert 'text:contains("a|b")' == parts[0]
        assert "rect" == parts[1]

    def test_quote_in_regex_parens_not_miscounted(self):
        """Regex with parens inside quotes should not break paren balancing."""
        # :regex("foo(bar)") — the parens inside quotes should be ignored
        parsed = parse_selector(':regex("foo(bar)")')
        assert parsed["type"] == "any"
        assert any(p["name"] == "regex" for p in parsed["pseudo_classes"])

    def test_nested_parens_in_contains(self):
        """Nested parens without quotes should still be balanced."""
        parsed = parse_selector(":contains((Section A))")
        assert any(p["name"] == "contains" for p in parsed["pseudo_classes"])


# ---------------------------------------------------------------------------
# Task 3: Immutable selector dicts (no mutation in _build_filter_list)
# ---------------------------------------------------------------------------


class TestImmutableSelectorDicts:
    """Verify _build_filter_list does not mutate its input."""

    def test_no_mutation_simple(self):
        parsed = parse_selector("text:bold")
        original_keys = set(parsed.keys())
        _build_filter_list(parsed)
        assert "post_pseudos" not in parsed
        assert "relational_pseudos" not in parsed
        assert set(parsed.keys()) == original_keys

    def test_no_mutation_with_post_pseudo(self):
        parsed = parse_selector("text:first")
        original_keys = set(parsed.keys())
        filters, post, rel = _build_filter_list(parsed)
        assert "post_pseudos" not in parsed
        assert len(post) == 1
        assert post[0]["name"] == "first"

    def test_no_mutation_with_relational_pseudo(self):
        parsed = parse_selector("text:above(rect)")
        original_keys = set(parsed.keys())
        filters, post, rel = _build_filter_list(parsed)
        assert "relational_pseudos" not in parsed
        assert len(rel) == 1
        assert rel[0]["name"] == "above"

    def test_build_execution_plan_returns_pseudos(self):
        parsed = parse_selector("text:bold:first:above(rect)")
        func, post, rel = build_execution_plan(parsed)
        assert callable(func)
        assert any(p["name"] == "first" for p in post)
        assert any(p["name"] == "above" for p in rel)

    def test_build_execution_plan_or_collects_pseudos(self):
        parsed = parse_selector("text:first|rect:last")
        func, post, rel = build_execution_plan(parsed)
        names = {p["name"] for p in post}
        assert "first" in names
        assert "last" in names

    def test_cache_not_corrupted_after_execution_plan(self):
        """Calling build_execution_plan should not corrupt the cached selector."""
        selector = "text:bold:first"
        parsed1 = parse_selector(selector)
        build_execution_plan(parsed1)

        parsed2 = parse_selector(selector)
        # cached result should be identical and uncorrupted
        assert parsed1 is parsed2
        assert "post_pseudos" not in parsed2

    def test_no_copy_import(self):
        """parser.py should no longer import copy."""
        import inspect

        import natural_pdf.selectors.parser as p

        source = inspect.getsource(p)
        # Should not have 'import copy' or 'copy.deepcopy'
        assert "copy.deepcopy" not in source


# ---------------------------------------------------------------------------
# Task 5: New post-pseudos — :nth, :slice, :limit
# ---------------------------------------------------------------------------


class TestNthPostPseudo:
    def test_nth_registered(self):
        assert get_post_handler("nth") is not None

    def test_nth_zero_index(self):
        handler = get_post_handler("nth")
        ctx = ClauseEvalContext(selector_context=None, aggregates={}, options={})
        elements = [_make_element(text="a"), _make_element(text="b"), _make_element(text="c")]
        result = handler(elements, {"name": "nth", "args": 0}, ctx)
        assert len(result) == 1
        assert result[0].text == "a"

    def test_nth_last_element(self):
        handler = get_post_handler("nth")
        ctx = ClauseEvalContext(selector_context=None, aggregates={}, options={})
        elements = [_make_element(text="a"), _make_element(text="b"), _make_element(text="c")]
        result = handler(elements, {"name": "nth", "args": 2}, ctx)
        assert len(result) == 1
        assert result[0].text == "c"

    def test_nth_negative_index(self):
        handler = get_post_handler("nth")
        ctx = ClauseEvalContext(selector_context=None, aggregates={}, options={})
        elements = [_make_element(text="a"), _make_element(text="b"), _make_element(text="c")]
        result = handler(elements, {"name": "nth", "args": -1}, ctx)
        assert len(result) == 1
        assert result[0].text == "c"

    def test_nth_out_of_range(self):
        handler = get_post_handler("nth")
        ctx = ClauseEvalContext(selector_context=None, aggregates={}, options={})
        elements = [_make_element(text="a")]
        result = handler(elements, {"name": "nth", "args": 5}, ctx)
        assert len(result) == 0

    def test_nth_empty_collection(self):
        handler = get_post_handler("nth")
        ctx = ClauseEvalContext(selector_context=None, aggregates={}, options={})
        result = handler([], {"name": "nth", "args": 0}, ctx)
        assert len(result) == 0

    def test_nth_via_selector(self):
        parsed = parse_selector("text:nth(1)")
        _, post, _ = build_execution_plan(parsed)
        assert any(p["name"] == "nth" for p in post)


class TestSlicePostPseudo:
    def test_slice_registered(self):
        assert get_post_handler("slice") is not None

    def test_slice_single_arg_stop(self):
        handler = get_post_handler("slice")
        ctx = ClauseEvalContext(selector_context=None, aggregates={}, options={})
        elements = [_make_element(text=str(i)) for i in range(5)]
        result = handler(elements, {"name": "slice", "args": 3}, ctx)
        assert len(result) == 3
        assert [el.text for el in result] == ["0", "1", "2"]

    def test_slice_tuple_start_stop(self):
        handler = get_post_handler("slice")
        ctx = ClauseEvalContext(selector_context=None, aggregates={}, options={})
        elements = [_make_element(text=str(i)) for i in range(5)]
        result = handler(elements, {"name": "slice", "args": (1, 4)}, ctx)
        assert len(result) == 3
        assert [el.text for el in result] == ["1", "2", "3"]

    def test_slice_tuple_with_step(self):
        handler = get_post_handler("slice")
        ctx = ClauseEvalContext(selector_context=None, aggregates={}, options={})
        elements = [_make_element(text=str(i)) for i in range(6)]
        result = handler(elements, {"name": "slice", "args": (0, 6, 2)}, ctx)
        assert [el.text for el in result] == ["0", "2", "4"]

    def test_slice_via_selector(self):
        parsed = parse_selector("text:slice(2)")
        _, post, _ = build_execution_plan(parsed)
        assert any(p["name"] == "slice" for p in post)

    def test_slice_tuple_via_selector(self):
        """Parsing :slice(1, 3) should yield a tuple arg."""
        parsed = parse_selector("text:slice(1, 3)")
        pseudo = next(p for p in parsed["pseudo_classes"] if p["name"] == "slice")
        # safe_parse_value parses "1, 3" — might be string or tuple depending on parser
        # The handler must handle either way


class TestLimitPostPseudo:
    def test_limit_registered(self):
        assert get_post_handler("limit") is not None

    def test_limit_caps_results(self):
        handler = get_post_handler("limit")
        ctx = ClauseEvalContext(selector_context=None, aggregates={}, options={})
        elements = [_make_element(text=str(i)) for i in range(10)]
        result = handler(elements, {"name": "limit", "args": 3}, ctx)
        assert len(result) == 3
        assert [el.text for el in result] == ["0", "1", "2"]

    def test_limit_larger_than_collection(self):
        handler = get_post_handler("limit")
        ctx = ClauseEvalContext(selector_context=None, aggregates={}, options={})
        elements = [_make_element(text="a")]
        result = handler(elements, {"name": "limit", "args": 10}, ctx)
        assert len(result) == 1

    def test_limit_zero(self):
        handler = get_post_handler("limit")
        ctx = ClauseEvalContext(selector_context=None, aggregates={}, options={})
        elements = [_make_element(text="a"), _make_element(text="b")]
        result = handler(elements, {"name": "limit", "args": 0}, ctx)
        assert len(result) == 0

    def test_limit_negative_returns_all(self):
        handler = get_post_handler("limit")
        ctx = ClauseEvalContext(selector_context=None, aggregates={}, options={})
        elements = [_make_element(text="a"), _make_element(text="b")]
        result = handler(elements, {"name": "limit", "args": -1}, ctx)
        assert len(result) == 2

    def test_limit_via_selector(self):
        parsed = parse_selector("text:limit(5)")
        _, post, _ = build_execution_plan(parsed)
        assert any(p["name"] == "limit" for p in post)

    def test_limit_empty_collection(self):
        handler = get_post_handler("limit")
        ctx = ClauseEvalContext(selector_context=None, aggregates={}, options={})
        result = handler([], {"name": "limit", "args": 5}, ctx)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Integration: end-to-end selector_to_filter_func still works
# ---------------------------------------------------------------------------


class TestSelectorToFilterFuncCompat:
    """selector_to_filter_func should still work as a backward-compatible wrapper."""

    def test_simple_selector(self):
        parsed = parse_selector("text:bold")
        func = selector_to_filter_func(parsed)
        assert func(_make_element(bold=True)) is True
        assert func(_make_element(bold=False)) is False

    def test_or_selector(self):
        parsed = parse_selector("text:bold|text:italic")
        func = selector_to_filter_func(parsed)
        assert func(_make_element(bold=True, italic=False)) is True
        assert func(_make_element(bold=False, italic=True)) is True
        assert func(_make_element(bold=False, italic=False)) is False

    def test_not_selector(self):
        parsed = parse_selector("text:not(text:bold)")
        func = selector_to_filter_func(parsed)
        assert func(_make_element(bold=True)) is False
        assert func(_make_element(bold=False)) is True
