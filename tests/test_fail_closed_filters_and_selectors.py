"""Regression tests for correctness-sensitive filters and selectors."""

from __future__ import annotations

from typing import Any

import pytest

from natural_pdf.elements.region import Region
from natural_pdf.exceptions import (
    ContentFilterError,
    ExclusionError,
    SelectorParseError,
)


def _public_text_hosts(page: Any) -> list[Any]:
    """Return representative Page, Region, and TextElement extraction hosts."""
    return [
        page,
        Region(page, (0, 0, page.width, page.height)),
        page.words[0],
    ]


def test_public_content_filters_raise_for_invalid_regex(practice_pdf):
    page = practice_pdf.pages[0]

    for host in _public_text_hosts(page):
        with pytest.raises(ContentFilterError, match="Invalid content_filter") as caught:
            host.extract_text(content_filter="[")

        assert isinstance(caught.value.__cause__, Exception)


def test_empty_text_hosts_still_validate_content_filter(needs_ocr_pdf_fresh):
    page = needs_ocr_pdf_fresh.pages[0]
    assert not page.words
    empty_region = Region(page, (0, 0, 1, 1))

    for host in (page, empty_region, page.find_all("text")):
        with pytest.raises(ContentFilterError, match="Invalid content_filter"):
            host.extract_text(content_filter="[")


def test_public_content_filters_raise_for_failing_callable(practice_pdf):
    page = practice_pdf.pages[0]

    def broken_filter(_character: str) -> bool:
        raise LookupError("filter configuration is missing")

    for host in _public_text_hosts(page):
        with pytest.raises(ContentFilterError, match="callable failed") as caught:
            host.extract_text(content_filter=broken_filter)

        assert isinstance(caught.value.__cause__, LookupError)


@pytest.mark.parametrize("granularity", ["chars", "words"])
def test_region_content_filter_errors_are_not_bypassed_by_granularity(practice_pdf, granularity):
    page = practice_pdf.pages[0]
    region = Region(page, (0, 0, page.width, page.height))

    with pytest.raises(ContentFilterError, match="Invalid content_filter"):
        region.extract_text(granularity=granularity, content_filter="[")


def test_region_alt_text_applies_content_filter(practice_pdf):
    page = practice_pdf.pages[0]
    region = Region(page, (0, 0, 10, 10))
    region.alt_text = "account 1234"

    assert region.extract_text(content_filter=r"\d") == "account "

    with pytest.raises(ContentFilterError, match="Invalid content_filter"):
        region.extract_text(content_filter="[")


def test_page_alt_text_only_applies_content_filter(needs_ocr_pdf_fresh):
    page = needs_ocr_pdf_fresh.pages[0]
    assert not page.words

    region = page.create_region(10, 10, 100, 30)
    region.alt_text = "account 1234"
    page.add_region(region, source="user")

    filtered = page.extract_text(content_filter=r"\d")
    assert filtered == "account"
    assert "1234" not in filtered


@pytest.mark.parametrize("method_name", ["get_children", "get_descendants"])
def test_region_hierarchy_methods_raise_for_malformed_selector(practice_pdf, method_name):
    page = practice_pdf.pages[0]
    region = Region(page, (0, 0, page.width, page.height))

    with pytest.raises(SelectorParseError, match="region selector") as caught:
        getattr(region, method_name)("text[")

    assert isinstance(caught.value.__cause__, ValueError)


def test_raising_exclusion_callable_aborts_extraction_and_restores_context(
    practice_pdf_fresh,
):
    page = practice_pdf_fresh.pages[0]

    def broken_exclusion(_page):
        raise RuntimeError("exclusion dependency failed")

    page.add_exclusion(broken_exclusion, label="confidential header")

    with pytest.raises(ExclusionError, match="confidential header") as caught:
        page.extract_text()

    assert isinstance(caught.value.__cause__, RuntimeError)
    assert page._computing_exclusions is False


@pytest.mark.parametrize(
    "result",
    [object(), [object()]],
    ids=["unsupported-value", "partially-unconvertible-iterable"],
)
def test_invalid_exclusion_callable_result_aborts_extraction(practice_pdf_fresh, result):
    page = practice_pdf_fresh.pages[0]
    page.add_exclusion(lambda _page: result, label="confidential area")

    with pytest.raises(ExclusionError, match="unsupported|Unable to convert"):
        page.extract_text()


def test_exclusion_callable_none_remains_an_explicit_noop(practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]
    expected = page.extract_text()
    page.add_exclusion(lambda _page: None, label="optional area")

    assert page.extract_text() == expected


def test_exclusion_expansion_failure_aborts_extraction(practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]
    element = page.words[0]

    def broken_expand():
        raise RuntimeError("cannot determine boundary")

    element.expand = broken_expand
    page._exclusions.append((element, "confidential word", "region"))

    with pytest.raises(ExclusionError, match="confidential word") as caught:
        page.extract_text()

    assert isinstance(caught.value.__cause__, RuntimeError)


def test_malformed_relational_reference_selector_raises(practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]

    with pytest.raises(SelectorParseError, match="relational reference") as caught:
        page.find_all("text:above(text[)")

    assert isinstance(caught.value.__cause__, ValueError)


@pytest.mark.parametrize("selector", ["text:nth(foo)", "text:slice(foo)", "text:limit(foo)"])
def test_invalid_collection_pseudo_arguments_do_not_broaden_results(practice_pdf_fresh, selector):
    page = practice_pdf_fresh.pages[0]

    with pytest.raises(ValueError, match="requires"):
        page.find_all(selector)
