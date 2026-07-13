"""Public text-extraction contracts and implementation helpers.

Keeping this as a regular package (rather than an implicit namespace package)
lets documentation and type-introspection tools resolve the mixins used by
the public ``extract_text`` methods.
"""

from natural_pdf.text.contracts import (
    AggregatePolicy,
    BBox,
    ContentFilter,
    ExtractedText,
    RegexFilter,
    SourceTextSegment,
    SpatialTextInput,
    TextDirection,
    TextLayoutOptions,
    WhitespaceMode,
)

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
