"""Contract tests for structural ``extract_text`` aggregate hosts."""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from typing import Any

import pytest

from natural_pdf.core.context import PDFContext
from natural_pdf.core.page_collection import PageCollection
from natural_pdf.core.pdf import PDF
from natural_pdf.core.pdf_collection import PDFCollection
from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.exceptions import ContentFilterError
from natural_pdf.flows.collections import FlowElementCollection, FlowRegionCollection
from natural_pdf.flows.element import FlowElement
from natural_pdf.flows.flow import Flow
from natural_pdf.flows.region import FlowRegion


@dataclass
class _Leaf:
    text: str
    calls: list[dict[str, Any]] = field(default_factory=list)
    _context: PDFContext = field(default_factory=PDFContext.with_defaults)

    def extract_text(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        content_filter = kwargs.get("content_filter")
        if isinstance(content_filter, str):
            return re.sub(content_filter, "", self.text)
        return self.text


def _bare_pdf(leaves: list[_Leaf]) -> PDF:
    pdf = object.__new__(PDF)
    pdf._pages = leaves
    pdf._context = PDFContext.with_defaults()
    return pdf


def _bare_flow(leaves: list[_Leaf], *, arrangement: str = "vertical") -> Flow:
    flow = object.__new__(Flow)
    flow.segments = leaves
    flow.arrangement = arrangement
    return flow


def _bare_flow_region(leaves: list[_Leaf]) -> FlowRegion:
    region = object.__new__(FlowRegion)
    region.constituent_regions = leaves
    return region


def _bare_flow_region_collection(leaves: list[_Leaf]) -> FlowRegionCollection:
    collection = object.__new__(FlowRegionCollection)
    collection._flow_regions = leaves
    return collection


@pytest.mark.parametrize("separator", [None, "", "\n\n--\n\n"])
def test_pdf_and_page_collection_have_identical_stable_joining(separator: str | None) -> None:
    leaves = [_Leaf("first"), _Leaf(""), _Leaf("third")]
    pdf = _bare_pdf(leaves)
    pages = PageCollection(leaves)
    expected_separator = "\n" if separator is None else separator
    expected = expected_separator.join(["first", "", "third"])

    assert pdf.extract_text(separator=separator) == expected
    assert pages.extract_text(separator=separator) == expected


def test_aggregate_options_are_applied_independently_to_every_member() -> None:
    leaves = [_Leaf("first"), _Leaf("second")]
    pages = PageCollection(leaves)

    assert (
        pages.extract_text(
            layout=True,
            apply_exclusions=False,
            newlines="|",
            whitespace="normalize",
            strip=False,
            bidi=False,
        )
        == "first\nsecond"
    )
    assert leaves[0].calls == leaves[1].calls
    call = leaves[0].calls[0]
    assert call["layout"] is True
    assert {key: value for key, value in call.items() if key != "layout"} == {
        "apply_exclusions": False,
        "newlines": "|",
        "whitespace": "normalize",
        "strip": False,
        "bidi": False,
        "content_filter": None,
    }


def test_aggregate_filters_cannot_match_across_member_or_separator_boundaries() -> None:
    leaves = [_Leaf("left"), _Leaf("right")]
    pages = PageCollection(leaves)

    assert pages.extract_text(content_filter=r"left\nright") == "left\nright"
    assert all(leaf.calls[-1]["content_filter"] == r"left\nright" for leaf in leaves)


def test_aggregate_text_result_preserves_exact_member_offsets_and_empty_segments() -> None:
    leaves = [_Leaf("first"), _Leaf(""), _Leaf("third")]
    result = PageCollection(leaves).extract_text_result(separator="::")

    assert result.text == "first::::third"
    assert [
        (segment.output_start, segment.output_end, segment.source) for segment in result.segments
    ] == [
        (0, 5, leaves[0]),
        (7, 7, leaves[1]),
        (9, 14, leaves[2]),
    ]


@pytest.mark.parametrize("arrangement", ["vertical", "horizontal"])
def test_flow_family_uses_double_newline_in_declared_order(arrangement: str) -> None:
    leaves = [_Leaf("one"), _Leaf(""), _Leaf("three")]
    expected = "one\n\n\n\nthree"

    assert _bare_flow(leaves, arrangement=arrangement).extract_text() == expected
    assert _bare_flow_region(leaves).extract_text() == expected
    assert _bare_flow_region_collection(leaves).extract_text() == expected


def test_pdf_extract_text_does_not_accept_the_removed_selector_api() -> None:
    parameters = inspect.signature(PDF.extract_text).parameters
    assert "selector" not in parameters
    assert "page_separator" not in parameters
    assert "return_textmap" not in parameters


def test_pdf_collection_has_only_explicit_per_document_extraction() -> None:
    collection = object.__new__(PDFCollection)
    collection._pdfs = []

    assert not hasattr(collection, "extract_text")
    assert collection.extract_each_text() == []
    with pytest.raises(TypeError):
        collection.extract_each_text(use_exclusions=False)


def test_pdf_collection_extract_each_forwards_the_leaf_contract() -> None:
    leaf = _Leaf("document")
    collection = object.__new__(PDFCollection)
    collection._pdfs = [leaf]

    assert collection.extract_each_text(layout=True, apply_exclusions=False) == ["document"]
    assert leaf.calls == [
        {
            "layout": True,
            "apply_exclusions": False,
            "newlines": True,
            "whitespace": "preserve",
            "strip": True,
            "bidi": True,
            "content_filter": None,
        }
    ]


@pytest.mark.parametrize(
    "collection",
    [
        PageCollection([]),
        _bare_flow_region_collection([]),
        pytest.param(object.__new__(PDFCollection), id="pdf-collection"),
    ],
)
def test_extract_each_validates_requests_before_reading_empty_members(collection) -> None:
    if isinstance(collection, PDFCollection):
        collection._pdfs = []

    with pytest.raises(TypeError):
        collection.extract_each_text(layout="invalid")
    with pytest.raises(TypeError):
        collection.extract_each_text(apply_exclusions=1)
    with pytest.raises(ValueError):
        collection.extract_each_text(whitespace="invalid")
    with pytest.raises(ContentFilterError):
        collection.extract_each_text(content_filter="[")


def test_section_extract_each_signature_is_exact_and_keyword_only() -> None:
    for host in (PageCollection, FlowRegionCollection):
        parameters = inspect.signature(host.extract_each_text).parameters
        assert "kwargs" not in parameters
        assert all(
            parameter.kind is inspect.Parameter.KEYWORD_ONLY
            for name, parameter in parameters.items()
            if name != "self"
        )


def test_flow_element_has_an_ide_visible_no_argument_text_proxy() -> None:
    physical = _Leaf("visible")
    physical.bbox = (0, 0, 1, 1)  # type: ignore[attr-defined]
    physical.page = None  # type: ignore[attr-defined]
    flow = object.__new__(Flow)
    element = FlowElement(physical, flow)

    assert list(inspect.signature(FlowElement.extract_text).parameters) == ["self"]
    assert element.extract_text() == "visible"


def test_flow_element_collection_retains_selected_order_and_duplicates() -> None:
    physical = _Leaf("selected")
    physical.bbox = (0, 0, 1, 1)  # type: ignore[attr-defined]
    physical.page = None  # type: ignore[attr-defined]
    flow = object.__new__(Flow)
    selected = FlowElement(physical, flow)

    assert FlowElementCollection([selected, selected]).extract_text(separator="|") == (
        "selected|selected"
    )


def test_selected_nested_aggregate_preserves_its_leaf_and_separator_boundaries() -> None:
    nested = _bare_flow([_Leaf("left"), _Leaf("right")])
    selected = ElementCollection([nested])

    assert selected.extract_text(content_filter=r"left\n\nright") == "left\n\nright"
    assert selected.extract_text(newlines=False) == "left\n\nright"
