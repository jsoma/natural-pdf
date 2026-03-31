"""Prompt templates for VLM-based document conversion and OCR."""

from __future__ import annotations

import re
from typing import List, Optional

# Mapping of language codes (ISO 639-1, PaddleOCR, EasyOCR) to human-readable names.
_LANGUAGE_NAMES = {
    # ISO 639-1
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "zh-cn": "Chinese (Simplified)",
    "zh-hans": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    "zh-hant": "Chinese (Traditional)",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "th": "Thai",
    "vi": "Vietnamese",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "fi": "Finnish",
    "cs": "Czech",
    "hu": "Hungarian",
    "ro": "Romanian",
    "uk": "Ukrainian",
    "el": "Greek",
    "he": "Hebrew",
    "id": "Indonesian",
    "ms": "Malay",
    "fa": "Persian",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
    "bn": "Bengali",
    # PaddleOCR-specific codes
    "ch": "Chinese",
    "chinese_cht": "Chinese (Traditional)",
    "japan": "Japanese",
    "korean": "Korean",
    "german": "German",
    "french": "French",
}


def languages_to_hint(languages: Optional[List[str]]) -> str:
    """Convert a list of language codes to a human-readable hint sentence.

    Returns ``""`` for ``None``, ``[]``, or ``["en"]`` (no noise for English).
    Single language: ``"The document is in Japanese."``
    Multiple: ``"The document is in Japanese and English."``
    Unknown codes use ``.capitalize()`` as fallback.
    """
    if not languages:
        return ""

    # Deduplicate while preserving order, skip non-string entries
    seen: set[str] = set()
    unique: list[str] = []
    for code in languages:
        if not isinstance(code, str):
            continue
        lower = code.strip().lower()
        if lower not in seen:
            seen.add(lower)
            unique.append(lower)

    # Nothing left after filtering, or English-only: no hint needed
    if not unique or unique == ["en"]:
        return ""

    names = [_LANGUAGE_NAMES.get(code, code.capitalize()) for code in unique]

    if len(names) == 1:
        return f"The document is in {names[0]}."
    return f"The document is in {' and '.join([', '.join(names[:-1]), names[-1]])}."


def detect_model_family(model_name: str | None) -> str:
    """Detect the VLM model family from the model name string.

    Returns ``"gutenocr"``, ``"glm_ocr"``, ``"dots_mocr"``, ``"chandra"``,
    ``"qwen_vl"``, ``"gemini"``, ``"openai"``, or ``"generic"``.
    GutenOCR, dots.mocr, and Chandra are checked before qwen_vl because
    they are built on Qwen architectures and would otherwise match the
    qwen pattern.
    """
    if model_name is None:
        return "generic"
    if re.search(r"(?i)gutenocr", model_name):
        return "gutenocr"
    if re.search(r"(?i)glm.?ocr", model_name):
        return "glm_ocr"
    if re.search(r"(?i)dots[\.\-_]?mocr", model_name):
        return "dots_mocr"
    if re.search(r"(?i)chandra", model_name):
        return "chandra"
    if re.search(r"(?i)qwen.*vl", model_name):
        return "qwen_vl"
    if re.search(r"(?i)gemini", model_name):
        return "gemini"
    if re.search(r"(?i)(gpt-|o[134]-|openai)", model_name):
        return "openai"
    return "generic"


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

QWEN_VL_OCR_PROMPT = (
    "Extract all visible text from this document image with grounding. "
    "For each text segment, provide the bounding box as bbox_2d using "
    "normalized coordinates in 0-1000 range [x_min, y_min, x_max, y_max], "
    "along with the text content as label.\n\n"
    "Return the results as a JSON array of objects with keys: "
    '"bbox_2d", "label".\n'
    "Example:\n"
    '[{"bbox_2d": [10, 20, 500, 45], "label": "Hello World"}]\n\n'
    "Output ONLY the JSON array, no other text."
)

GEMINI_OCR_PROMPT = (
    "Extract all visible text from this document image. "
    "For each text segment, detect the 2D bounding box as box_2d using "
    "normalized coordinates in 0-1000 range [y_min, x_min, y_max, x_max], "
    "along with the text content as label.\n\n"
    "Return the results as a JSON array of objects with keys: "
    '"box_2d", "label".\n'
    "Example:\n"
    '[{"box_2d": [20, 10, 45, 500], "label": "Hello World"}]\n\n'
    "Output ONLY the JSON array, no other text."
)

OPENAI_OCR_PROMPT = (
    "Extract all visible text from this document image. "
    "For each text segment, provide the bounding box coordinates as "
    "[x_min, y_min, x_max, y_max] in pixel values relative to the image dimensions, "
    "along with the text content and your confidence (0.0–1.0).\n\n"
    "Return the results as a JSON array of objects with keys: "
    '"bbox", "text", "confidence".\n'
    "Example:\n"
    "```json\n"
    '[{"bbox": [10, 20, 200, 45], "text": "Hello World", "confidence": 0.95}]\n'
    "```\n\n"
    "IMPORTANT: Output ONLY a valid JSON array. No explanation, no markdown "
    "headers, no extra text. Just the JSON array."
)

GUTENOCR_PROMPT = "Return a layout-sensitive TEXT2D representation of the image."

GLM_OCR_PROMPT = "Text Recognition:"

DOTS_MOCR_PROMPT = (
    "Please output the layout information from the PDF image, including each "
    "layout element's bbox, its category, and the corresponding text content "
    "within the bbox.\n\n"
    "1. Bbox format: [x1, y1, x2, y2]\n\n"
    "2. Layout Categories: The possible categories are ['Caption', 'Footnote', "
    "'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', "
    "'Section-header', 'Table', 'Text', 'Title'].\n\n"
    "3. Text Extraction & Formatting Rules:\n"
    "    - Picture: For the 'Picture' category, the text field should be omitted.\n"
    "    - Formula: Format its text as LaTeX.\n"
    "    - Table: Format its text as HTML.\n"
    "    - All Others (Text, Title, etc.): Format their text as Markdown.\n\n"
    "4. Constraints:\n"
    "    - The output text must be the original text from the image, with no translation.\n"
    "    - All layout elements must be sorted according to human reading order.\n\n"
    "5. Final Output: The entire output must be a single JSON object."
)


# Chandra v0.1.x prompt (matches mlx-community/chandra-4bit, datalab-to/chandra).
# The v0.2.0+ prompt differs (0-1000 normalization, extra labels/tags) and
# targets datalab-to/chandra-ocr-2 — the pip package (engine="chandra2")
# handles that internally.
_CHANDRA_ALLOWED_TAGS = [
    "math",
    "br",
    "i",
    "b",
    "u",
    "del",
    "sup",
    "sub",
    "table",
    "tr",
    "td",
    "p",
    "th",
    "div",
    "pre",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "ul",
    "ol",
    "li",
    "input",
    "a",
    "span",
    "img",
    "hr",
    "tbody",
    "small",
    "caption",
    "strong",
    "thead",
    "big",
    "code",
]
_CHANDRA_ALLOWED_ATTRIBUTES = [
    "class",
    "colspan",
    "rowspan",
    "display",
    "checked",
    "type",
    "border",
    "value",
    "style",
    "href",
    "alt",
    "align",
]

CHANDRA_OCR_LAYOUT_PROMPT = f"""OCR this image to HTML, arranged as layout blocks.  Each layout block should be a div with the data-bbox attribute representing the bounding box of the block in [x0, y0, x1, y1] format.  Bboxes are normalized 0-1024. The data-label attribute is the label for the block.

Use the following labels:
- Caption
- Footnote
- Equation-Block
- List-Group
- Page-Header
- Page-Footer
- Image
- Section-Header
- Table
- Text
- Complex-Block
- Code-Block
- Form
- Table-Of-Contents
- Figure

Only use these tags {_CHANDRA_ALLOWED_TAGS}, and these attributes {_CHANDRA_ALLOWED_ATTRIBUTES}.

Guidelines:
* Inline math: Surround math with <math>...</math> tags. Math expressions should be rendered in KaTeX-compatible LaTeX. Use display for block math.
* Tables: Use colspan and rowspan attributes to match table structure.
* Formatting: Maintain consistent formatting with the image, including spacing, indentation, subscripts/superscripts, and special characters.
* Images: Include a description of any images in the alt attribute of an <img> tag. Do not fill out the src property.
* Forms: Mark checkboxes and radio buttons properly.
* Text: join lines together properly into paragraphs using <p>...</p> tags.  Use <br> tags for line breaks within paragraphs, but only when absolutely necessary to maintain meaning.
* Use the simplest possible HTML structure that accurately represents the content of the block.
* Make sure the text is accurate and easy for a human to read and interpret.  Reading order should be correct and natural."""


def build_conversion_prompt(*, format: str = "markdown") -> str:
    """Return a prompt for document-to-text conversion.

    Args:
        format: Target format (currently only ``"markdown"`` is supported).
    """
    if format == "markdown":
        return CONVERSION_PROMPT
    return CONVERSION_PROMPT


def build_ocr_prompt(
    *,
    grounding: bool = True,
    family: str = "generic",
    languages: Optional[List[str]] = None,
) -> str:
    """Return a prompt for VLM-based OCR.

    Args:
        grounding: If True, request bounding box coordinates with each text span.
        family: Model family (``"generic"``, ``"qwen_vl"``, or ``"gutenocr"``).
        languages: Optional list of language codes to prepend as a hint.
    """
    hint = languages_to_hint(languages)

    if not grounding:
        base = (
            "Extract all visible text from this document image. "
            "Return the text content only, preserving the reading order."
        )
        return f"{hint} {base}" if hint else base
    if family == "qwen_vl":
        return f"{hint} {QWEN_VL_OCR_PROMPT}" if hint else QWEN_VL_OCR_PROMPT
    if family == "gemini":
        return f"{hint} {GEMINI_OCR_PROMPT}" if hint else GEMINI_OCR_PROMPT
    if family == "openai":
        return f"{hint} {OPENAI_OCR_PROMPT}" if hint else OPENAI_OCR_PROMPT
    if family == "gutenocr":
        return f"{hint} {GUTENOCR_PROMPT}" if hint else GUTENOCR_PROMPT
    if family == "glm_ocr":
        # GLM-OCR uses a fixed prompt; language hints are not applicable.
        return GLM_OCR_PROMPT
    if family == "dots_mocr":
        return DOTS_MOCR_PROMPT
    if family == "chandra":
        return CHANDRA_OCR_LAYOUT_PROMPT
    return f"{hint} {OCR_GROUNDING_PROMPT}" if hint else OCR_GROUNDING_PROMPT
