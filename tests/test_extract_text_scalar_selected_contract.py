from __future__ import annotations

import inspect

import pytest

from natural_pdf.elements.base import Element
from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.elements.image import ImageElement
from natural_pdf.elements.line import LineElement
from natural_pdf.elements.text import TextElement
from natural_pdf.exceptions import ContentFilterError
from natural_pdf.text.contracts import ExtractedText


class _Page:
    page_number = 1
    number = 1
    index = 0
    width = 100
    height = 100


def _text(value: str, x0: float = 0) -> TextElement:
    return TextElement(
        {
            "text": value,
            "x0": x0,
            "top": 0,
            "x1": x0 + 10,
            "bottom": 10,
            "size": 10,
        },
        _Page(),
    )


def _nontext(element_type):
    return element_type(
        {
            "x0": 0,
            "top": 0,
            "x1": 10,
            "bottom": 10,
            "width": 10,
            "height": 10,
        },
        _Page(),
    )


def test_scalar_signature_is_small_and_keyword_only() -> None:
    signature = inspect.signature(TextElement.extract_text)
    assert list(signature.parameters) == [
        "self",
        "newlines",
        "whitespace",
        "strip",
        "content_filter",
    ]
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for name, parameter in signature.parameters.items()
        if name != "self"
    )


def test_scalar_pipeline_has_canonical_newline_whitespace_and_filter_semantics() -> None:
    element = _text("  SECRET\r\n   value  ")

    assert element.extract_text(content_filter="SECRET") == "value"
    assert element.extract_text(newlines=False, whitespace="normalize") == "SECRET value"
    assert element.extract_text(newlines=" | ") == "SECRET |    value"


def test_scalar_filters_validate_and_fail_closed() -> None:
    element = _text("")
    with pytest.raises(ContentFilterError):
        element.extract_text(content_filter="[")

    def broken(_character: str) -> bool:
        raise LookupError("missing policy")

    with pytest.raises(ContentFilterError) as caught:
        _text("x").extract_text(content_filter=broken)
    assert isinstance(caught.value.__cause__, LookupError)


def test_nontext_contract_is_no_argument_and_rejects_irrelevant_options() -> None:
    for element in (_nontext(Element), _nontext(LineElement), _nontext(ImageElement)):
        assert element.extract_text() == ""
        with pytest.raises(TypeError):
            element.extract_text(strip=False)


def test_selected_collection_preserves_stored_order_and_duplicates() -> None:
    first = _text("first", 20)
    second = _text("second", 0)
    collection = ElementCollection([first, second, first])

    assert collection.extract_text(separator="|") == "first|second|first"


def test_selected_collection_omits_empty_contributions_without_cross_boundary_filters() -> None:
    collection = ElementCollection([_text("SEC"), _nontext(LineElement), _text("RET"), _text("")])

    assert collection.extract_text(content_filter="SECRET") == "SEC RET"
    assert collection.extract_text(separator="|") == "SEC|RET"


def test_selected_collection_validates_before_reading_empty_members() -> None:
    collection = ElementCollection([])
    with pytest.raises(ContentFilterError):
        collection.extract_text(content_filter="[")
    with pytest.raises(TypeError):
        collection.extract_text(layout=True)


def test_scalar_and_selected_result_offsets_are_explicit() -> None:
    first = _text("alpha")
    second = _text("beta")

    scalar = first.extract_text_result()
    assert isinstance(scalar, ExtractedText)
    assert scalar.text == "alpha"
    assert len(scalar.segments) == 1
    assert scalar.segments[0].source is first
    assert (scalar.segments[0].output_start, scalar.segments[0].output_end) == (0, 5)

    selected = ElementCollection([first, second]).extract_text_result(separator=" -- ")
    assert selected.text == "alpha -- beta"
    assert [(segment.output_start, segment.output_end) for segment in selected.segments] == [
        (0, 5),
        (9, 13),
    ]


def test_extract_each_validates_empty_requests_and_uses_default_for_no_text() -> None:
    empty = ElementCollection([])
    with pytest.raises(ContentFilterError):
        empty.extract_each_text(content_filter="[")
    with pytest.raises(ValueError, match="order preset"):
        empty.extract_each_text(order="sideways")
    with pytest.raises(TypeError, match="default"):
        empty.extract_each_text(default=object())

    nontext = ElementCollection([_nontext(LineElement)])
    assert nontext.extract_each_text(default=None) == [None]
    assert nontext.extract_each_text(default="missing") == ["missing"]


def test_extract_each_order_callback_failures_are_not_silently_ignored() -> None:
    collection = ElementCollection([_text("first"), _text("second")])

    def fail(_element):
        raise RuntimeError("ordering failed")

    with pytest.raises(RuntimeError, match="ordering failed"):
        collection.extract_each_text(order=fail)
