from __future__ import annotations

import inspect

import pytest

from natural_pdf import PDF
from natural_pdf.core.page import Page
from natural_pdf.elements.rect import RectangleElement
from natural_pdf.elements.region import Region
from natural_pdf.exceptions import ContentFilterError, TextExtractionError
from natural_pdf.text.contracts import ExtractedText, TextLayoutOptions
from natural_pdf.text.facades import SpatialTextMixin


@pytest.fixture
def pdf():
    document = PDF("pdfs/01-practice.pdf")
    yield document
    document.close()


@pytest.fixture
def empty_pdf():
    document = PDF("pdfs/01-practice.pdf", text_layer=False)
    yield document
    document.close()


def _full_page_region(page: Page) -> Region:
    return Region(page, (0, 0, page.width, page.height))


def _full_page_rectangle(page: Page) -> RectangleElement:
    return RectangleElement(
        {
            "x0": 0,
            "top": 0,
            "x1": page.width,
            "bottom": page.height,
            "object_type": "rect",
        },
        page,
    )


def test_spatial_hosts_inherit_one_exact_keyword_only_signature():
    expected_names = [
        "self",
        "layout",
        "apply_exclusions",
        "newlines",
        "whitespace",
        "strip",
        "bidi",
        "content_filter",
    ]

    assert Page.extract_text is SpatialTextMixin.extract_text
    assert Region.extract_text is SpatialTextMixin.extract_text
    assert RectangleElement.extract_text is SpatialTextMixin.extract_text
    signature = inspect.signature(SpatialTextMixin.extract_text)
    assert list(signature.parameters) == expected_names
    assert signature.parameters["self"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert all(
        signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        for name in expected_names[1:]
    )
    assert signature.parameters["layout"].default is False
    assert signature.parameters["apply_exclusions"].default is True
    assert signature.parameters["strip"].default is True


@pytest.mark.parametrize(
    "legacy_kwargs",
    [
        {"return_textmap": True},
        {"granularity": "words"},
        {"overlap": "partial"},
        {"preserve_whitespace": True},
        {"preserve_line_breaks": False},
        {"use_exclusions": False},
        {"debug_exclusions": True},
        {"x_tolerance": 4},
        {"separator": " "},
    ],
)
def test_legacy_and_wrong_family_options_fail_even_on_empty_hosts(empty_pdf, legacy_kwargs):
    page = empty_pdf.pages[0]
    hosts = (page, _full_page_region(page), _full_page_rectangle(page))
    for host in hosts:
        with pytest.raises(TypeError):
            host.extract_text(**legacy_kwargs)


@pytest.mark.parametrize(
    "layout",
    [
        False,
        True,
        TextLayoutOptions(enabled=False),
        TextLayoutOptions(enabled=True, x_tolerance=4.5, y_tolerance=4.5),
    ],
)
def test_page_and_full_page_region_share_spatial_acquisition(pdf, layout):
    page = pdf.pages[0]
    region = _full_page_region(page)
    assert page.extract_text(layout=layout) == region.extract_text(layout=layout)


def test_layout_true_uses_page_configured_tolerances(pdf, monkeypatch):
    page = pdf.pages[0]
    page._config["x_tolerance"] = 8.25
    page._config["y_tolerance"] = 6.5
    captured = []

    def fake_extract(spatial_input, *, layout):
        captured.append((layout, dict(spatial_input.layout_defaults)))
        return ExtractedText(text="", segments=())

    monkeypatch.setattr("natural_pdf.core.page.extract_spatial_text", fake_extract)
    page.extract_text_result(layout=True)
    assert captured == [(True, {"x_tolerance": 8.25, "y_tolerance": 6.5})]


def test_explicit_layout_options_are_authoritative(pdf, monkeypatch):
    page = pdf.pages[0]
    page._config["x_tolerance"] = 99
    requested = TextLayoutOptions(enabled=True, x_tolerance=1.25, keep_blank_chars=True)
    captured = []

    def fake_extract(spatial_input, *, layout):
        captured.append(layout)
        return ExtractedText(text="", segments=())

    monkeypatch.setattr("natural_pdf.core.page.extract_spatial_text", fake_extract)
    page.extract_text_result(layout=requested)
    assert captured == [requested]


def test_multicharacter_filter_runs_on_final_spatial_string(pdf):
    page = pdf.pages[0]
    original = page.extract_text()
    assert "Jungle" in original
    filtered = page.extract_text(content_filter="Jungle")
    assert "Jungle" not in filtered
    assert len(filtered) < len(original)


def test_content_filter_callable_failure_is_fail_closed(pdf):
    def fail(_character: str) -> bool:
        raise RuntimeError("filter exploded")

    with pytest.raises(ContentFilterError) as exc_info:
        pdf.pages[0].extract_text(content_filter=fail)
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_alt_only_text_uses_the_normal_transform_pipeline(empty_pdf):
    page = empty_pdf.pages[0]
    region = Region(page, (10, 10, 100, 50))
    region.alt_text = "  SECRET\nVALUE  "
    page.add_region(region, source="test")

    assert region.extract_text(content_filter="SECRET", newlines=" / ") == "/ VALUE"
    assert page.extract_text(content_filter="SECRET", newlines=False) == "VALUE"


def test_mixed_native_and_alt_text_share_final_filtering(pdf):
    page = pdf.pages[0]
    alt = Region(page, (10, 10, 40, 30))
    alt.alt_text = "ALTSECRET"
    page.add_region(alt, source="test")

    text = page.extract_text(content_filter="(?:Jungle|ALTSECRET)")
    assert "Jungle" not in text
    assert "ALTSECRET" not in text


def test_page_and_full_region_have_exclusion_parity(pdf):
    page = pdf.pages[0]
    word = page.find("text:contains('Jungle')")
    assert word is not None
    page.add_exclusion(word.expand(), label="target")
    region = _full_page_region(page)
    unfiltered = page.extract_text(apply_exclusions=False)
    filtered = page.extract_text(apply_exclusions=True)

    assert filtered == region.extract_text(apply_exclusions=True)
    assert unfiltered == region.extract_text(apply_exclusions=False)
    assert filtered.count("Jungle") == unfiltered.count("Jungle") - 1


def test_layout_failure_raises_contextual_error_without_fallback(pdf, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("backend exploded")

    monkeypatch.setattr("natural_pdf.text.pipeline.chars_to_textmap", fail)
    with pytest.raises(TextExtractionError) as exc_info:
        pdf.pages[0].extract_text(layout=True)
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "Page" in str(exc_info.value)


def test_spatial_result_preserves_source_textmap_words_and_geometry(pdf):
    page = pdf.pages[0]
    result = page.extract_text_result(layout=True)

    assert result.text
    assert len(result.segments) == 1
    segment = result.segments[0]
    assert segment.output_start == 0
    assert segment.output_end == len(result.text)
    assert segment.source is page
    assert segment.textmap is not None
    assert segment.words
    assert segment.words[0] is page.words[0]
    assert segment.page_number == page.number
    assert segment.bbox == (0.0, 0.0, page.width, page.height)
    first_native_char = page.words[0]._char_dicts[0]
    assert any(char is first_native_char for _, char in segment.textmap.tuples)


def test_empty_spatial_result_still_has_one_zero_length_source_segment(empty_pdf):
    page = empty_pdf.pages[0]
    result = page.extract_text_result()

    assert result.text == ""
    assert len(result.segments) == 1
    segment = result.segments[0]
    assert segment.output_start == segment.output_end == 0
    assert segment.source is page
    assert segment.textmap is None
    assert segment.words == ()
    assert segment.page_number == page.number


def test_rectangle_result_retains_rectangle_source_identity(pdf):
    page = pdf.pages[0]
    rectangle = _full_page_rectangle(page)
    result = rectangle.extract_text_result()

    assert result.text == page.extract_text()
    assert result.segments[0].source is rectangle
    assert result.segments[0].bbox == rectangle.bbox


def test_region_word_mode_migrates_to_explicit_selected_words(pdf):
    page = pdf.pages[0]
    region = page.find("text:contains('Jungle')").below(height=80)
    with pytest.raises(TypeError):
        region.extract_text(granularity="words")

    words = region.find_all("text", overlap="center")
    expected = " ".join(word.extract_text() for word in words)
    assert words.extract_text(separator=" ") == expected
