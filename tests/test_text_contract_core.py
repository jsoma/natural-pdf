from __future__ import annotations

import inspect
import re
from dataclasses import FrozenInstanceError

import pytest

import natural_pdf
from natural_pdf.exceptions import ContentFilterError, TextExtractionError
from natural_pdf.text.contracts import (
    AggregatePolicy,
    ExtractedText,
    SourceTextSegment,
    SpatialTextInput,
    TextLayoutOptions,
)
from natural_pdf.text.facades import (
    AggregateTextMixin,
    ScalarTextMixin,
    SelectedTextMixin,
    SpatialTextMixin,
)
from natural_pdf.text.pipeline import (
    extract_spatial_text,
    filter_text,
    join_extracted_text,
    transform_text,
)


class _Scalar(ScalarTextMixin):
    def __init__(self, text: str):
        self.text = text
        self.calls = 0

    def _extract_scalar_text(self) -> str:
        self.calls += 1
        return self.text


class _Spatial(SpatialTextMixin):
    def __init__(self, result: ExtractedText):
        self.result = result
        self.calls = 0

    def _extract_spatial_text_result(self, *, layout, apply_exclusions):
        self.calls += 1
        return self.result


class _Aggregate(AggregateTextMixin):
    def __init__(self, members, *, preserve_empty=True):
        self.members = members
        self.preserve_empty = preserve_empty
        self.touched = False

    def _iter_text_members(self):
        self.touched = True
        return iter(self.members)

    def _text_aggregate_policy(self):
        return AggregatePolicy(
            natural_separator="\r\n--\t--\r\n", preserve_empty=self.preserve_empty
        )


class _Selected(SelectedTextMixin):
    def __init__(self, members):
        self.members = members

    def _iter_selected_text_members(self):
        return iter(self.members)


def _result(text: str, source: object) -> ExtractedText:
    return ExtractedText(
        text=text,
        segments=(
            SourceTextSegment(
                output_start=0,
                output_end=len(text),
                source=source,
            ),
        ),
    )


def test_public_contract_exports() -> None:
    assert natural_pdf.ExtractedText is ExtractedText
    assert natural_pdf.TextLayoutOptions is TextLayoutOptions
    assert natural_pdf.TextExtractionError is TextExtractionError
    assert natural_pdf.ContentFilter is not None


def test_layout_options_are_frozen_keyword_only_and_eagerly_validated() -> None:
    with pytest.raises(TypeError):
        TextLayoutOptions(True)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        TextLayoutOptions().enabled = False  # type: ignore[misc]
    with pytest.raises(TypeError):
        TextLayoutOptions(enabled=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        TextLayoutOptions(x_tolerance=-1)
    with pytest.raises(ValueError):
        TextLayoutOptions(x_density=0)
    with pytest.raises(ValueError):
        TextLayoutOptions(char_dir="sideways")  # type: ignore[arg-type]


def test_layout_options_exclude_host_geometry_and_untyped_escape_hatches() -> None:
    fields = set(TextLayoutOptions.__dataclass_fields__)
    assert not fields.intersection(
        {"bbox", "layout_bbox", "layout_width", "layout_height", "x_shift", "y_shift"}
    )
    assert not fields.intersection({"presorted", "use_text_flow", "extra_attrs", "kwargs"})


def test_spatial_input_copies_layout_defaults_and_validates_them() -> None:
    defaults = {"x_tolerance": 8.0}
    value = SpatialTextInput(chars=(), source="page", layout_defaults=defaults)
    defaults["x_tolerance"] = 2.0
    assert value.layout_defaults["x_tolerance"] == 8.0
    with pytest.raises(TypeError, match="unsupported layout"):
        SpatialTextInput(chars=(), source="page", layout_defaults={"x_shift": 2})


def test_extracted_text_keeps_zero_length_segments_and_checks_bounds() -> None:
    segment = SourceTextSegment(output_start=0, output_end=0, source="empty")
    assert ExtractedText(text="", segments=(segment,)).segments == (segment,)
    with pytest.raises(ValueError, match="beyond"):
        ExtractedText(
            text="x",
            segments=(SourceTextSegment(output_start=0, output_end=2, source="bad"),),
        )


def test_regex_sequence_is_fully_compiled_before_any_filtering() -> None:
    class ObservedPattern:
        def __init__(self):
            self.calls = 0

        def sub(self, replacement, text):
            self.calls += 1
            return text

    observed = ObservedPattern()
    with pytest.raises(TypeError):
        filter_text("secret", [observed, object()])  # type: ignore[list-item]
    assert observed.calls == 0


def test_compiled_and_sequence_regex_filters_apply_to_complete_string() -> None:
    assert filter_text("SECRET 123", (re.compile("SECRET"), r"\d+")) == " "
    with pytest.raises(ContentFilterError) as error:
        filter_text("", ["okay", "["])
    assert isinstance(error.value.__cause__, re.error)


def test_callable_filter_is_per_codepoint_and_failures_are_chained() -> None:
    seen: list[str] = []

    def keep(character: str) -> bool:
        seen.append(character)
        return character != "b"

    assert filter_text("abc", keep) == "ac"
    assert seen == ["a", "b", "c"]

    cause = RuntimeError("broken")

    def fail(character: str) -> bool:
        raise cause

    with pytest.raises(ContentFilterError) as error:
        filter_text("x", fail)
    assert error.value.__cause__ is cause


def test_transform_order_and_horizontal_only_whitespace(monkeypatch) -> None:
    monkeypatch.setattr(
        "natural_pdf.text.operations.apply_bidi_processing",
        lambda text: text.replace("visual", "LOGICAL\r\n"),
    )
    assert (
        transform_text(
            "  visual\t SECRET  ",
            newlines="|",
            whitespace="normalize",
            strip=True,
            bidi=True,
            content_filter="SECRET",
        )
        == "LOGICAL|"
    )
    assert transform_text("a\t  b\n\n c", whitespace="normalize", strip=False) == "a b\n\n c"


def test_spatial_render_is_raw_sorted_and_merges_layout_options(monkeypatch) -> None:
    captured = {}

    class Map:
        as_string = "  raw\r\ntext  "

    def fake(chars, **kwargs):
        captured["texts"] = [char["text"] for char in chars]
        captured["kwargs"] = kwargs
        return Map()

    monkeypatch.setattr("natural_pdf.text.pipeline.chars_to_textmap", fake)
    spatial = SpatialTextInput(
        chars=(
            {"text": "b", "top": 2, "x0": 0},
            {"text": "a", "top": 1, "x0": 0},
        ),
        source="page",
        bbox=(10, 20, 110, 220),
        layout_defaults={"x_tolerance": 8.0, "y_tolerance": None},
    )
    result = extract_spatial_text(
        spatial,
        layout=TextLayoutOptions(enabled=True, y_tolerance=4.0),
    )
    assert result.text == "  raw\r\ntext  "
    assert captured["texts"] == ["a", "b"]
    assert captured["kwargs"]["x_tolerance"] == 8.0
    assert captured["kwargs"]["y_tolerance"] == 4.0
    assert captured["kwargs"]["x_density"] == 7.25
    assert captured["kwargs"]["layout_bbox"] == (10.0, 20.0, 110.0, 220.0)
    assert result.segments[0].textmap.__class__ is Map


def test_spatial_layout_failure_has_cause_and_never_falls_back(monkeypatch) -> None:
    cause = RuntimeError("backend failed")

    def fail(chars, **kwargs):
        raise cause

    monkeypatch.setattr("natural_pdf.text.pipeline.chars_to_textmap", fail)
    spatial = SpatialTextInput(chars=({"text": "x"},), source="page")
    with pytest.raises(TextExtractionError) as error:
        extract_spatial_text(spatial, layout=False)
    assert error.value.__cause__ is cause


def test_empty_spatial_layout_still_validates_host_geometry() -> None:
    spatial = SpatialTextInput(
        chars=(),
        source="empty-region",
        bbox=(0, 0, 0, 10),
    )

    with pytest.raises(TextExtractionError, match="positive width"):
        extract_spatial_text(spatial, layout=True)


def test_join_offsets_include_arbitrary_separator_and_empty_sources() -> None:
    first = _result("A", "first")
    empty = _result("", "empty")
    last = _result("BC", "last")
    joined = join_extracted_text([first, empty, last], separator=" -- ", preserve_empty=True)
    assert joined.text == "A --  -- BC"
    assert [(s.output_start, s.output_end, s.source) for s in joined.segments] == [
        (0, 1, "first"),
        (5, 5, "empty"),
        (9, 11, "last"),
    ]


def test_facades_have_exact_discoverable_signatures() -> None:
    assert list(inspect.signature(SpatialTextMixin.extract_text).parameters) == [
        "self",
        "layout",
        "apply_exclusions",
        "newlines",
        "whitespace",
        "strip",
        "bidi",
        "content_filter",
    ]
    assert list(inspect.signature(ScalarTextMixin.extract_text_result).parameters) == ["self"]
    assert list(inspect.signature(SelectedTextMixin.extract_text_result).parameters) == [
        "self",
        "separator",
    ]
    assert all(
        parameter.kind is not inspect.Parameter.VAR_KEYWORD
        for mixin in (SpatialTextMixin, ScalarTextMixin, AggregateTextMixin, SelectedTextMixin)
        for parameter in inspect.signature(mixin.extract_text).parameters.values()
    )


def test_invalid_facade_requests_do_not_touch_host_hooks() -> None:
    scalar = _Scalar("text")
    with pytest.raises(ContentFilterError):
        scalar.extract_text(content_filter="[")
    assert scalar.calls == 0

    spatial = _Spatial(_result("text", "page"))
    with pytest.raises(TypeError):
        spatial.extract_text(apply_exclusions="yes")  # type: ignore[arg-type]
    assert spatial.calls == 0

    aggregate = _Aggregate([])
    with pytest.raises(ValueError):
        aggregate.extract_text(whitespace="collapse")  # type: ignore[arg-type]
    assert aggregate.touched is False


def test_aggregate_transforms_members_but_never_separator() -> None:
    aggregate = _Aggregate(
        [_Spatial(_result(" A SECRET ", "one")), _Spatial(_result(" B ", "two"))]
    )
    assert aggregate.extract_text(content_filter="SECRET") == "A\r\n--\t--\r\nB"


def test_selected_order_duplicates_and_empty_omission() -> None:
    repeated = _Scalar(" x ")
    selected = _Selected([repeated, _Scalar(""), object(), repeated])
    assert selected.extract_text(separator="|") == "x|x"
    result = selected.extract_text_result(separator="--")
    assert result.text == " x -- x "
    assert [segment.source for segment in result.segments] == [repeated, repeated]
