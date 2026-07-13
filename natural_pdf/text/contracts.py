"""Public, host-neutral contracts for text extraction.

This module intentionally depends only on the Python standard library.  It is
safe for type checkers, plugins, and lightweight integrations to import without
initialising a PDF backend.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType
from typing import Any, Literal, Pattern, TypeAlias

WhitespaceMode: TypeAlias = Literal["preserve", "normalize"]
RegexFilter: TypeAlias = str | Pattern[str]
ContentFilter: TypeAlias = RegexFilter | Sequence[RegexFilter] | Callable[[str], bool]
TextDirection: TypeAlias = Literal["ltr", "rtl", "ttb", "btt"]
BBox: TypeAlias = tuple[float, float, float, float]


def _real(name: str, value: object, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if not positive and result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _optional_real(name: str, value: object, *, positive: bool = False) -> float | None:
    if value is None:
        return None
    return _real(name, value, positive=positive)


def _bool(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")


def _direction(name: str, value: object, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str):
        suffix = " or None" if optional else ""
        raise TypeError(f"{name} must be a text direction{suffix}")
    if value not in {"ltr", "rtl", "ttb", "btt"}:
        suffix = " or None" if optional else ""
        raise ValueError(f"{name} must be one of 'ltr', 'rtl', 'ttb', 'btt'{suffix}")


@dataclass(frozen=True, slots=True, kw_only=True)
class TextLayoutOptions:
    """Typed pdfplumber text-layout options.

    Host geometry (bbox, width/height, and coordinate shifts) belongs to
    :class:`SpatialTextInput` and cannot be overridden here.
    """

    enabled: bool = True
    x_tolerance: float | None = None
    y_tolerance: float | None = None
    x_tolerance_ratio: float | None = None
    y_tolerance_ratio: float | None = None
    x_density: float | None = None
    y_density: float | None = None
    keep_blank_chars: bool | None = None
    line_dir: TextDirection | None = None
    char_dir: TextDirection | None = None
    line_dir_rotated: TextDirection | None = None
    char_dir_rotated: TextDirection | None = None
    line_dir_render: TextDirection | None = None
    char_dir_render: TextDirection | None = None
    split_at_punctuation: bool | str | None = None
    expand_ligatures: bool | None = None

    def __post_init__(self) -> None:
        _bool("enabled", self.enabled)
        for name in ("x_tolerance", "y_tolerance"):
            object.__setattr__(self, name, _optional_real(name, getattr(self, name)))
        for name in ("x_tolerance_ratio", "y_tolerance_ratio"):
            object.__setattr__(self, name, _optional_real(name, getattr(self, name)))
        for name in ("x_density", "y_density"):
            object.__setattr__(self, name, _optional_real(name, getattr(self, name), positive=True))
        for name in ("keep_blank_chars", "expand_ligatures"):
            if getattr(self, name) is not None:
                _bool(name, getattr(self, name))
        for name in ("line_dir", "char_dir"):
            _direction(name, getattr(self, name), optional=True)
        for name in (
            "line_dir_rotated",
            "char_dir_rotated",
            "line_dir_render",
            "char_dir_render",
        ):
            _direction(name, getattr(self, name), optional=True)
        if self.split_at_punctuation is not None and not isinstance(
            self.split_at_punctuation, (bool, str)
        ):
            raise TypeError("split_at_punctuation must be a bool, str, or None")


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceTextSegment:
    """A source's exact half-open span in an :class:`ExtractedText` value."""

    output_start: int
    output_end: int
    source: object
    textmap: Any | None = None
    words: tuple[Any, ...] = ()
    page_number: int | None = None
    bbox: BBox | None = None

    def __post_init__(self) -> None:
        for name in ("output_start", "output_end"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.output_end < self.output_start:
            raise ValueError("output_end must not precede output_start")
        if not isinstance(self.words, tuple):
            object.__setattr__(self, "words", tuple(self.words))
        if self.page_number is not None and (
            isinstance(self.page_number, bool) or not isinstance(self.page_number, int)
        ):
            raise TypeError("page_number must be an int or None")
        if self.bbox is not None:
            if len(self.bbox) != 4:
                raise ValueError("bbox must contain exactly four coordinates")
            bbox = tuple(_real(f"bbox[{index}]", value) for index, value in enumerate(self.bbox))
            object.__setattr__(self, "bbox", bbox)


@dataclass(frozen=True, slots=True, kw_only=True)
class ExtractedText:
    """Immutable text plus exact source spans."""

    text: str
    segments: tuple[SourceTextSegment, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be a str")
        if not isinstance(self.segments, tuple):
            object.__setattr__(self, "segments", tuple(self.segments))
        for index, segment in enumerate(self.segments):
            if not isinstance(segment, SourceTextSegment):
                raise TypeError(f"segments[{index}] must be a SourceTextSegment")
            if segment.output_end > len(self.text):
                raise ValueError(f"segments[{index}] extends beyond the output text")


@dataclass(frozen=True, slots=True, kw_only=True)
class AggregatePolicy:
    """Host-owned rules for joining ordered text members."""

    natural_separator: str
    preserve_empty: bool

    def __post_init__(self) -> None:
        if not isinstance(self.natural_separator, str):
            raise TypeError("natural_separator must be a str")
        _bool("preserve_empty", self.preserve_empty)


@dataclass(frozen=True, slots=True, kw_only=True)
class SpatialTextInput:
    """Host-neutral positioned character input for spatial extraction."""

    chars: tuple[Mapping[str, Any], ...]
    source: object
    bbox: BBox | None = None
    page_number: int | None = None
    words: tuple[Any, ...] = field(default_factory=tuple)
    layout_defaults: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.chars, tuple):
            object.__setattr__(self, "chars", tuple(self.chars))
        for index, char in enumerate(self.chars):
            if not isinstance(char, Mapping):
                raise TypeError(f"chars[{index}] must be a mapping")
        if not isinstance(self.words, tuple):
            object.__setattr__(self, "words", tuple(self.words))
        if not isinstance(self.layout_defaults, Mapping):
            raise TypeError("layout_defaults must be a mapping")
        defaults = dict(self.layout_defaults)
        valid_fields = {
            name for name in TextLayoutOptions.__dataclass_fields__ if name != "enabled"
        }
        unknown = set(defaults) - valid_fields
        if unknown:
            names = ", ".join(sorted(unknown))
            raise TypeError(f"unsupported layout default(s): {names}")
        # Reuse the public value object's eager type/range validation.
        TextLayoutOptions(**defaults)
        object.__setattr__(self, "layout_defaults", MappingProxyType(defaults))
        if self.page_number is not None and (
            isinstance(self.page_number, bool) or not isinstance(self.page_number, int)
        ):
            raise TypeError("page_number must be an int or None")
        if self.bbox is not None:
            if len(self.bbox) != 4:
                raise ValueError("bbox must contain exactly four coordinates")
            bbox = tuple(_real(f"bbox[{index}]", value) for index, value in enumerate(self.bbox))
            object.__setattr__(self, "bbox", bbox)


__all__ = [
    "AggregatePolicy",
    "BBox",
    "ContentFilter",
    "ExtractedText",
    "RegexFilter",
    "SourceTextSegment",
    "SpatialTextInput",
    "TextDirection",
    "TextLayoutOptions",
    "WhitespaceMode",
]
