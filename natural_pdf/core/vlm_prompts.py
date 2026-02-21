"""Prompt templates for VLM-based document conversion and OCR."""

from __future__ import annotations

CONVERSION_PROMPT = (
    "Convert this document page to Markdown. "
    "Preserve the structure including headings, lists, tables, and paragraphs. "
    "Do not add any commentary or explanation — output only the Markdown content."
)

OCR_GROUNDING_PROMPT = (
    "Extract all visible text from this document image. "
    "For each text segment, provide the bounding box coordinates as "
    "[x_min, y_min, x_max, y_max] in pixel values relative to the image dimensions, "
    "along with the text content and your confidence (0.0–1.0).\n\n"
    "Return the results as a JSON array of objects with keys: "
    '"bbox", "text", "confidence".\n'
    "Example:\n"
    '[{"bbox": [10, 20, 200, 45], "text": "Hello World", "confidence": 0.95}]\n\n'
    "Output ONLY the JSON array, no other text."
)


def build_conversion_prompt(*, format: str = "markdown") -> str:
    """Return a prompt for document-to-text conversion.

    Args:
        format: Target format (currently only ``"markdown"`` is supported).
    """
    if format == "markdown":
        return CONVERSION_PROMPT
    return CONVERSION_PROMPT


def build_ocr_prompt(*, grounding: bool = True) -> str:
    """Return a prompt for VLM-based OCR.

    Args:
        grounding: If True, request bounding box coordinates with each text span.
    """
    if grounding:
        return OCR_GROUNDING_PROMPT
    return (
        "Extract all visible text from this document image. "
        "Return the text content only, preserving the reading order."
    )
