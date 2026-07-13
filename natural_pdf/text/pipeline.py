"""Canonical acquisition, transformation, and provenance-aware text joins."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Pattern

from pdfplumber.utils.text import chars_to_textmap

from natural_pdf.exceptions import ContentFilterError, TextExtractionError
from natural_pdf.text.contracts import (
    ContentFilter,
    ExtractedText,
    SourceTextSegment,
    SpatialTextInput,
    TextLayoutOptions,
    WhitespaceMode,
)

_HORIZONTAL_WHITESPACE = re.compile(r"[^\S\n]+")


@dataclass(frozen=True, slots=True)
class _PreparedTextTransform:
    newlines: bool | str
    whitespace: WhitespaceMode
    strip: bool
    bidi: bool
    patterns: tuple[Pattern[str], ...]
    predicate: Callable[[str], bool] | None


def validate_layout_request(layout: bool | TextLayoutOptions) -> TextLayoutOptions:
    """Validate and normalize a public layout request without touching host data."""

    if isinstance(layout, bool):
        return TextLayoutOptions(enabled=layout)
    if isinstance(layout, TextLayoutOptions):
        return layout
    raise TypeError("layout must be a bool or TextLayoutOptions")


def _prepare_content_filter(
    content_filter: ContentFilter | None,
) -> tuple[tuple[Pattern[str], ...], Callable[[str], bool] | None]:
    if content_filter is None:
        return (), None
    if callable(content_filter):
        return (), content_filter

    raw_patterns: Sequence[str | Pattern[str]]
    if isinstance(content_filter, (str, re.Pattern)):
        raw_patterns = (content_filter,)
    elif isinstance(content_filter, Sequence) and not isinstance(
        content_filter, (str, bytes, bytearray)
    ):
        raw_patterns = content_filter
    else:
        raise TypeError(
            "content_filter must be a regex string, compiled pattern, sequence of "
            "regexes, callable, or None"
        )

    # Compile the complete sequence before any text is mutated.  A bad later
    # entry must never leak a partially filtered value.
    compiled: list[Pattern[str]] = []
    for index, candidate in enumerate(raw_patterns):
        if isinstance(candidate, re.Pattern):
            if not isinstance(candidate.pattern, str):
                raise TypeError(f"content_filter[{index}] must be a text regex")
            compiled.append(candidate)
            continue
        if not isinstance(candidate, str):
            raise TypeError(f"content_filter[{index}] must be a regex string or compiled pattern")
        try:
            compiled.append(re.compile(candidate))
        except re.error as exc:
            raise ContentFilterError(
                f"Invalid content_filter regular expression at index {index}: {candidate!r}"
            ) from exc
    return tuple(compiled), None


def prepare_text_transform(
    *,
    newlines: bool | str,
    whitespace: WhitespaceMode,
    strip: bool,
    bidi: bool,
    content_filter: ContentFilter | None,
) -> _PreparedTextTransform:
    """Eagerly validate every transformation option and compile all regexes."""

    if not isinstance(newlines, (bool, str)):
        raise TypeError("newlines must be a bool or replacement string")
    if whitespace not in ("preserve", "normalize"):
        raise ValueError("whitespace must be 'preserve' or 'normalize'")
    if not isinstance(strip, bool):
        raise TypeError("strip must be a bool")
    if not isinstance(bidi, bool):
        raise TypeError("bidi must be a bool")
    patterns, predicate = _prepare_content_filter(content_filter)
    return _PreparedTextTransform(
        newlines=newlines,
        whitespace=whitespace,
        strip=strip,
        bidi=bidi,
        patterns=patterns,
        predicate=predicate,
    )


def apply_prepared_text_transform(text: str, request: _PreparedTextTransform) -> str:
    """Apply the frozen transform order to one already-acquired string."""

    if not isinstance(text, str):
        raise TypeError("extracted text must be a str")

    result = text
    if request.bidi:
        # Delayed import avoids making the stdlib-only contracts module pull in
        # optional bidi functionality.
        from natural_pdf.text.operations import apply_bidi_processing

        result = apply_bidi_processing(result)

    result = result.replace("\r\n", "\n").replace("\r", "\n")
    if request.newlines is False:
        result = result.replace("\n", " ")
    elif isinstance(request.newlines, str) and not isinstance(request.newlines, bool):
        result = result.replace("\n", request.newlines)

    if request.whitespace == "normalize":
        # Newline policy has already run.  Normalize only horizontal whitespace
        # inside each surviving line; never collapse or invent line boundaries.
        result = "\n".join(_HORIZONTAL_WHITESPACE.sub(" ", line) for line in result.split("\n"))

    result = _apply_prepared_content_filter(result, request.patterns, request.predicate)

    if request.strip:
        result = result.strip()
    return result


def _apply_prepared_content_filter(
    text: str,
    patterns: tuple[Pattern[str], ...],
    predicate: Callable[[str], bool] | None,
) -> str:
    result = text
    for pattern in patterns:
        result = pattern.sub("", result)

    if predicate is not None:
        kept: list[str] = []
        for character in result:
            try:
                keep = predicate(character)
            except Exception as exc:
                raise ContentFilterError(
                    f"content_filter callable failed for character {character!r}"
                ) from exc
            if not isinstance(keep, bool):
                error = TypeError(
                    "content_filter callable must return bool for each character; "
                    f"received {type(keep).__name__}"
                )
                raise ContentFilterError(
                    f"content_filter callable returned an invalid value for {character!r}"
                ) from error
            if keep:
                kept.append(character)
        result = "".join(kept)
    return result


def filter_text(text: str, content_filter: ContentFilter | None) -> str:
    """Apply only the canonical filter contract to a complete string."""

    if not isinstance(text, str):
        raise TypeError("text must be a str")
    patterns, predicate = _prepare_content_filter(content_filter)
    return _apply_prepared_content_filter(text, patterns, predicate)


def validate_content_filter_request(content_filter: ContentFilter | None) -> None:
    """Compile and type-check a filter without evaluating host text."""

    _prepare_content_filter(content_filter)


def transform_text(
    text: str,
    *,
    newlines: bool | str = True,
    whitespace: WhitespaceMode = "preserve",
    strip: bool = True,
    bidi: bool = True,
    content_filter: ContentFilter | None = None,
) -> str:
    """Validate then transform text through the canonical public pipeline."""

    request = prepare_text_transform(
        newlines=newlines,
        whitespace=whitespace,
        strip=strip,
        bidi=bidi,
        content_filter=content_filter,
    )
    return apply_prepared_text_transform(text, request)


def _layout_kwargs(options: TextLayoutOptions, spatial: SpatialTextInput) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "layout": options.enabled,
        "x_tolerance": 5.0,
        "y_tolerance": 5.0,
        "x_density": 7.25,
        "y_density": 13.0,
    }
    kwargs.update(
        {key: value for key, value in spatial.layout_defaults.items() if value is not None}
    )
    for name in (
        "x_tolerance",
        "y_tolerance",
        "x_tolerance_ratio",
        "y_tolerance_ratio",
        "x_density",
        "y_density",
        "keep_blank_chars",
        "line_dir",
        "char_dir",
        "line_dir_rotated",
        "char_dir_rotated",
        "line_dir_render",
        "char_dir_render",
        "split_at_punctuation",
        "expand_ligatures",
    ):
        value = getattr(options, name)
        if value is not None:
            kwargs[name] = value

    if options.enabled and spatial.bbox is not None:
        x0, top, x1, bottom = spatial.bbox
        width = x1 - x0
        height = bottom - top
        if width <= 0 or height <= 0:
            raise TextExtractionError("spatial extraction bbox must have positive width and height")
        kwargs.update(
            layout_bbox=spatial.bbox,
            layout_width=width,
            layout_height=height,
            x_shift=x0,
            y_shift=top,
        )
    return kwargs


def extract_spatial_text(
    spatial: SpatialTextInput,
    *,
    layout: bool | TextLayoutOptions = False,
) -> ExtractedText:
    """Render positioned characters without filtering, transforms, or stripping."""

    options = validate_layout_request(layout)
    if not isinstance(spatial, SpatialTextInput):
        raise TypeError("spatial must be a SpatialTextInput")

    for index, char in enumerate(spatial.chars):
        if not isinstance(char.get("text"), str):
            raise TextExtractionError(f"spatial character {index} has no string text value")

    # Resolve and validate the complete acquisition request before inspecting
    # whether this host happens to contain text.  In particular, an invalid
    # layout bbox must not become valid merely because the current result is
    # empty.
    layout_kwargs = _layout_kwargs(options, spatial)

    if not spatial.chars:
        return ExtractedText(
            text="",
            segments=(
                SourceTextSegment(
                    output_start=0,
                    output_end=0,
                    source=spatial.source,
                    textmap=None,
                    words=spatial.words,
                    page_number=spatial.page_number,
                    bbox=spatial.bbox,
                ),
            ),
        )

    try:
        chars = list(spatial.chars)
        chars.sort(key=lambda char: (char.get("top", 0), char.get("x0", 0)))
        textmap = chars_to_textmap(chars, **layout_kwargs)
        text = textmap.as_string
    except TextExtractionError:
        raise
    except Exception as exc:
        raise TextExtractionError(
            f"Failed to extract spatial text from {type(spatial.source).__name__}"
        ) from exc

    return ExtractedText(
        text=text,
        segments=(
            SourceTextSegment(
                output_start=0,
                output_end=len(text),
                source=spatial.source,
                textmap=textmap,
                words=spatial.words,
                page_number=spatial.page_number,
                bbox=spatial.bbox,
            ),
        ),
    )


def scalar_text_result(text: str, *, source: object) -> ExtractedText:
    """Wrap scalar text in a source-aware immutable result."""

    if not isinstance(text, str):
        raise TypeError("scalar text hook must return str")
    return ExtractedText(
        text=text,
        segments=(SourceTextSegment(output_start=0, output_end=len(text), source=source),),
    )


def join_extracted_text(
    results: Sequence[ExtractedText],
    *,
    separator: str,
    preserve_empty: bool = True,
) -> ExtractedText:
    """Join results and shift every source span by the exact output offset."""

    if not isinstance(separator, str):
        raise TypeError("separator must be a str")
    if not isinstance(preserve_empty, bool):
        raise TypeError("preserve_empty must be a bool")
    checked: list[ExtractedText] = []
    for index, result in enumerate(results):
        if not isinstance(result, ExtractedText):
            raise TypeError(f"results[{index}] must be ExtractedText")
        if preserve_empty or result.text:
            checked.append(result)

    parts: list[str] = []
    segments: list[SourceTextSegment] = []
    offset = 0
    for index, result in enumerate(checked):
        if index:
            parts.append(separator)
            offset += len(separator)
        parts.append(result.text)
        for segment in result.segments:
            segments.append(
                replace(
                    segment,
                    output_start=segment.output_start + offset,
                    output_end=segment.output_end + offset,
                )
            )
        offset += len(result.text)
    return ExtractedText(text="".join(parts), segments=tuple(segments))


__all__ = [
    "apply_prepared_text_transform",
    "extract_spatial_text",
    "filter_text",
    "join_extracted_text",
    "prepare_text_transform",
    "scalar_text_result",
    "transform_text",
    "validate_content_filter_request",
    "validate_layout_request",
]
