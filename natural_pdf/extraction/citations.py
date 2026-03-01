"""Citation/grounding and confidence support for structured extraction.

Maps extracted field values back to source locations in the PDF by:
1. Sending line-numbered text to the LLM
2. Getting line numbers back via source_lines fields in the schema
3. Resolving line numbers to source text via pdfplumber's TextMap provenance data

Also supports per-field confidence scoring via a parallel shadow-field pattern.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

from pydantic import BaseModel, Field, create_model

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Confidence configuration
# ------------------------------------------------------------------ #

DEFAULT_CONFIDENCE_SCALE: Dict[int, str] = {
    1: "Barely related — document touches the topic but provides no real evidence for this value",
    2: "Weakly supported — related information exists but is incomplete; significant inference required",
    3: "Moderately supported — relevant information present but noticeable inference or synthesis needed",
    4: "Strongly supported — document clearly implies the answer with only minor inference",
    5: "Explicitly supported — answer is directly and clearly stated in the document",
}


@dataclass
class ConfidenceConfig:
    """Describes the confidence scale the LLM should use."""

    scale: Dict[Any, Optional[str]]  # {value: description_or_None}
    is_numeric: bool
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    @property
    def is_integer_scale(self) -> bool:
        """True when all scale keys are integers (use int type, not float)."""
        return self.is_numeric and all(isinstance(k, int) for k in self.scale)


def normalize_confidence_config(
    confidence: Union[None, bool, str, list, dict],
) -> Optional[ConfidenceConfig]:
    """Normalize the user-facing ``confidence`` parameter into a :class:`ConfidenceConfig`.

    Accepts:
    - ``None`` / ``False`` → ``None``
    - ``True`` / ``'range'`` → default 0.0–1.0 numeric scale
    - ``list`` → categorical (no descriptions)
    - ``dict`` with all-numeric keys → numeric scale
    - ``dict`` with any string keys → categorical with descriptions
    """
    if confidence is None or confidence is False:
        return None

    if confidence is True or confidence == "range":
        return ConfidenceConfig(
            scale=DEFAULT_CONFIDENCE_SCALE,
            is_numeric=True,
            min_value=1,
            max_value=5,
        )

    if isinstance(confidence, list):
        if len(confidence) < 2:
            raise ValueError("Confidence list must have at least 2 items.")
        all_numeric = all(isinstance(v, (int, float)) for v in confidence)
        if all_numeric:
            sorted_vals = sorted(float(v) for v in confidence)
            return ConfidenceConfig(
                scale={v: None for v in confidence},
                is_numeric=True,
                min_value=sorted_vals[0],
                max_value=sorted_vals[-1],
            )
        return ConfidenceConfig(
            scale={v: None for v in confidence},
            is_numeric=False,
        )

    if isinstance(confidence, dict):
        if not confidence:
            raise ValueError("Confidence dict cannot be empty.")
        all_numeric = all(isinstance(k, (int, float)) for k in confidence)
        if all_numeric:
            sorted_keys = sorted(float(k) for k in confidence)
            return ConfidenceConfig(
                scale=confidence,
                is_numeric=True,
                min_value=sorted_keys[0],
                max_value=sorted_keys[-1],
            )
        return ConfidenceConfig(
            scale=confidence,
            is_numeric=False,
        )

    raise TypeError(
        f"Unsupported confidence type: {type(confidence).__name__}. "
        "Expected bool, 'range', list, or dict."
    )


@dataclass
class PageTextMapInfo:
    """Tracks per-page TextMap and line range for multi-page citation resolution."""

    page_number: int  # 1-indexed
    textmap: Any  # pdfplumber TextMap (or None)
    word_elements: list  # TextElement words for this page
    line_start: int  # first line in unified text (0-based)
    line_end: int  # last line (exclusive)


# ------------------------------------------------------------------ #
# Extended schema / prompt / result splitting
# ------------------------------------------------------------------ #


_UNSET = object()  # sentinel for "no default"


def _iter_user_fields(user_schema: Type[BaseModel]):
    """Yield (field_name, annotation, default, description, alias) for each user field.

    ``alias`` is the JSON property name when set (e.g. ``"violation count"``),
    or ``None`` when the field has no alias.
    """
    fields_iter = (
        user_schema.model_fields.items()
        if hasattr(user_schema, "model_fields")
        else user_schema.__fields__.items()
    )
    for field_name, field_obj in fields_iter:
        if hasattr(user_schema, "model_fields"):
            annotation = field_obj.annotation
            description = field_obj.description
            alias = field_obj.alias  # None when no alias is set
            # In Pydantic v2, field_obj.default is PydanticUndefined when
            # no default is set, vs None when default is explicitly None.
            # We must distinguish "no default" from "default=None".
            if field_obj.is_required():
                default = _UNSET
            else:
                default = field_obj.default
        else:
            annotation = field_obj.outer_type_
            default = field_obj.default if not field_obj.required else _UNSET
            description = field_obj.field_info.description if field_obj.field_info else None
            alias = field_obj.alias if field_obj.alias != field_name else None
        yield field_name, annotation, default, description, alias


def build_extended_schema(
    user_schema: Type[BaseModel],
    *,
    with_sources: bool = False,
    with_confidence: bool = False,
    confidence_config: Optional[ConfidenceConfig] = None,
) -> Type[BaseModel]:
    """Create an extended schema adding ``_source_lines`` and/or ``_confidence`` fields.

    Uses evidence-first field ordering per user field:
    ``{field}_source_lines`` → ``{field}`` (value) → ``{field}_confidence``.
    """
    field_defs: Dict[str, Any] = {}

    for field_name, annotation, default, description, alias in _iter_user_fields(user_schema):
        display_name = alias or field_name

        # Source field (before value for evidence-first ordering)
        if with_sources:
            source_desc = (
                f"Line numbers from the input text that support '{display_name}' "
                f"(e.g. [3, 7]). Use the Lnn prefixes to identify line numbers."
            )
            field_defs[f"{field_name}_source_lines"] = (
                Optional[List[int]],
                Field(None, description=source_desc),
            )

        # Copy original field (preserve alias and default)
        field_kwargs: Dict[str, Any] = {}
        if description:
            field_kwargs["description"] = description
        if alias:
            field_kwargs["alias"] = alias
        if default is _UNSET:
            field_defs[field_name] = (annotation, Field(**field_kwargs))
        else:
            field_defs[field_name] = (annotation, Field(default, **field_kwargs))

        # Confidence field (after value)
        if with_confidence and confidence_config is not None:
            if confidence_config.is_numeric:
                conf_type = Optional[int] if confidence_config.is_integer_scale else Optional[float]
                conf_desc = (
                    f"Confidence score for '{display_name}' "
                    f"({confidence_config.min_value}–{confidence_config.max_value})."
                )
            else:
                keys = tuple(confidence_config.scale.keys())
                conf_type = Optional[Literal[keys]]  # type: ignore[valid-type]
                conf_desc = f"Confidence level for '{field_name}' (one of {list(keys)})."
            field_defs[f"{field_name}_confidence"] = (
                conf_type,
                Field(None, description=conf_desc),
            )

    suffix_parts = []
    if with_sources:
        suffix_parts.append("SourceLines")
    if with_confidence:
        suffix_parts.append("Confidence")
    suffix = "With" + "And".join(suffix_parts) if suffix_parts else ""
    return create_model(f"{user_schema.__name__}{suffix}", **field_defs)


def build_extended_prompt(
    prompt: Optional[str],
    user_schema: Type[BaseModel],
    *,
    instructions: Optional[str] = None,
    with_sources: bool = False,
    with_confidence: bool = False,
    confidence_config: Optional[ConfidenceConfig] = None,
) -> str:
    """Assemble the full prompt: base → instructions → citation block → confidence block."""
    base = prompt or (
        f"Extract the information corresponding to the fields in the "
        f"{user_schema.__name__} schema. Respond only with the structured data."
    )

    parts = [base]

    # Instructions
    if instructions:
        parts.append(f"\n\n{instructions}")

    # Citation block
    if with_sources:
        parts.append(
            "\n\nIMPORTANT - Citation instructions:\n"
            "The input text has line numbers prefixed as 'Lnn: ' at the start of each line.\n"
            "For each field, also fill in the corresponding '_source_lines' field with a list of "
            "line numbers (integers) that support the extracted value.\n"
            "Example: if line 'L03: Invoice #12345' contains the invoice number, "
            "set invoice_number_source_lines to [3].\n"
            "If a value spans multiple lines, include each line number in the list."
        )

    # Confidence block
    if with_confidence and confidence_config is not None:
        parts.append("\n\nIMPORTANT - Confidence scoring instructions:\n")
        if confidence_config.is_numeric:
            parts.append(
                "For each field, also fill in the corresponding '_confidence' field "
                f"with a number between {confidence_config.min_value} and "
                f"{confidence_config.max_value} indicating how confident you are "
                "in the extracted value.\n"
            )
        else:
            keys = list(confidence_config.scale.keys())
            parts.append(
                "For each field, also fill in the corresponding '_confidence' field "
                f"with one of: {keys}.\n"
            )

        # Add scale descriptions as anchors
        has_descriptions = any(v is not None for v in confidence_config.scale.values())
        if has_descriptions:
            parts.append("Use the following scale as guidance:\n")
            for value, desc in confidence_config.scale.items():
                if desc is not None:
                    parts.append(f"- {value}: {desc}\n")

        parts.append("If a field's value is null/None, set its confidence to null as well.")

    return "".join(parts)


def build_confidence_schema(
    user_schema: Type[BaseModel],
    confidence_config: Optional[ConfidenceConfig] = None,
) -> Type[BaseModel]:
    """Build a schema with ONLY ``{field}_confidence`` fields for pass-2 scoring.

    Unlike :func:`build_extended_schema`, this schema contains no user data
    fields — it is used exclusively in the second LLM call to rate confidence
    on already-extracted values.

    For numeric scales, fields are typed as ``Optional[int]`` for integer
    scales or ``Optional[float]`` for float scales (no ``confloat``
    constraints) to maximise validation leniency.
    """
    field_defs: Dict[str, Any] = {}

    for field_name, _annotation, _default, _description, alias in _iter_user_fields(user_schema):
        display_name = alias or field_name
        if confidence_config is not None and confidence_config.is_numeric:
            conf_type = Optional[int] if confidence_config.is_integer_scale else Optional[float]
            conf_desc = (
                f"Confidence score for '{display_name}' "
                f"({confidence_config.min_value}–{confidence_config.max_value})."
            )
        elif confidence_config is not None:
            keys = tuple(confidence_config.scale.keys())
            conf_type = Optional[Literal[keys]]  # type: ignore[valid-type]
            conf_desc = f"Confidence level for '{display_name}' (one of {list(keys)})."
        else:
            conf_type = Optional[int]
            conf_desc = f"Confidence score for '{display_name}'."

        field_defs[f"{field_name}_confidence"] = (
            conf_type,
            Field(None, description=conf_desc),
        )

    return create_model(f"{user_schema.__name__}Confidence", **field_defs)


def build_confidence_prompt(
    user_schema: Type[BaseModel],
    extracted_data: Dict[str, Any],
    confidence_config: Optional[ConfidenceConfig] = None,
) -> str:
    """Build the prompt for the confidence-only pass.

    Shows the already-extracted values and asks the LLM to rate each field
    strictly based on how well the document supports each extracted value.
    The source text is sent separately as user message content (not embedded
    in this prompt) to avoid duplication.
    """
    # Determine whether we're using the default 1-5 scale
    is_default_scale = (
        confidence_config is not None
        and confidence_config.is_numeric
        and confidence_config.min_value == 1
        and confidence_config.max_value == 5
        and confidence_config.scale is DEFAULT_CONFIDENCE_SCALE
    )

    if is_default_scale:
        # Full grounding instructions for the default 1-5 scale
        lines = [
            "You are rating extracted values using ONLY the provided document.",
            "",
            "Rules:",
            "- Use only evidence found in the document. Do NOT use outside knowledge.",
            "- Confidence reflects strength of document support, NOT real-world correctness.",
            "- If the document does not contain enough information to support a field's value,",
            "  set that field's confidence to null. Do NOT assign a score to unsupported values.",
            "- If a field's value is null/None, set its confidence to null as well.",
            "",
            "For each field, assign a confidence score from 1 to 5 based strictly on "
            "how strongly the document supports the extracted value.",
            "",
            "Confidence Scale:",
            "",
            "5 – Explicitly Supported",
            "  The answer is directly and clearly stated in the document.",
            "  No interpretation or inference is required.",
            "",
            "4 – Strongly Supported",
            "  The document clearly implies the answer.",
            "  Only a minor or straightforward inference is required.",
            "",
            "3 – Moderately Supported",
            "  Relevant information is present but noticeable inference or synthesis is needed.",
            "  Some ambiguity or alternative interpretation is possible.",
            "",
            "2 – Weakly Supported",
            "  Related information exists but is incomplete.",
            "  Significant inference is required; multiple interpretations possible.",
            "",
            "1 – Barely Related",
            "  The document touches the topic but provides no real evidence for this value.",
            "  Evidence is thin, ambiguous, or highly indirect.",
        ]
    else:
        # Custom scale — compact prompt
        if confidence_config is not None and confidence_config.is_numeric:
            scale_desc = (
                f"an integer from {confidence_config.min_value} to {confidence_config.max_value}"
            )
        elif confidence_config is not None:
            keys = list(confidence_config.scale.keys())
            scale_desc = f"one of {keys}"
        else:
            scale_desc = "an integer from 1 to 5"

        lines = [
            "Rate your confidence in each extracted field value.",
            f"For each field, provide {scale_desc}.",
            "Confidence reflects strength of document evidence — NOT real-world correctness.",
        ]

        # Add scale descriptions if provided
        if confidence_config is not None:
            has_descriptions = any(v is not None for v in confidence_config.scale.values())
            if has_descriptions:
                lines.append("")
                for value, desc in confidence_config.scale.items():
                    if desc is not None:
                        lines.append(f"  {value}: {desc}")

        lines.append("")
        lines.append("If a field's value is null/None, set its confidence to null as well.")

    lines.append("")
    lines.append("Extracted values:")
    for field_name, value in extracted_data.items():
        lines.append(f"  {field_name}: {value!r}")

    return "\n".join(lines)


def build_citations_schema(user_schema: Type[BaseModel]) -> Type[BaseModel]:
    """Build a schema with ONLY ``{field}_source_lines`` fields for pass-2 citation extraction.

    Contains no user data fields — used exclusively in a second LLM call
    to locate source line numbers for already-extracted values.
    """
    field_defs: Dict[str, Any] = {}

    for field_name, _annotation, _default, _description, alias in _iter_user_fields(user_schema):
        display_name = alias or field_name
        source_desc = (
            f"Line numbers from the input text that support '{display_name}' "
            f"(e.g. [3, 7]). Use the Lnn prefixes to identify line numbers."
        )
        field_defs[f"{field_name}_source_lines"] = (
            Optional[List[int]],
            Field(None, description=source_desc),
        )

    return create_model(f"{user_schema.__name__}Citations", **field_defs)


def build_citations_prompt(
    user_schema: Type[BaseModel],
    extracted_data: Dict[str, Any],
) -> str:
    """Build the prompt for the citations-only pass.

    Shows the already-extracted values and asks the LLM to find source
    line numbers. The numbered text is sent separately as user message
    content (not embedded in this prompt) to avoid duplication.
    """
    lines = [
        "Find the source lines that support each extracted field value.",
        "",
        "The input text has line numbers prefixed as 'Lnn: ' at the start of each line.",
        "For each field, provide a list of line numbers (integers) where the supporting text appears.",
        "Example: if line 'L03: Invoice #12345' supports a field, include 3 in that field's list.",
        "",
        "If a field's value is null/None, set its source_lines to null.",
        "",
        "Extracted values:",
    ]
    for field_name, value in extracted_data.items():
        lines.append(f"  {field_name}: {value!r}")

    return "\n".join(lines)


def split_extended_result(
    data: Dict[str, Any],
    user_schema: Type[BaseModel],
    *,
    with_sources: bool = False,
    with_confidence: bool = False,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Split an extended LLM response into (user_data, sources_dict, confidences_dict).

    Returns:
        A 3-tuple where sources_dict and confidences_dict are ``None`` when
        the corresponding feature was not requested.
    """
    if hasattr(user_schema, "model_fields"):
        user_fields = set(user_schema.model_fields.keys())
    else:
        user_fields = set(user_schema.__fields__.keys())

    user_data = {k: v for k, v in data.items() if k in user_fields}

    sources_dict: Optional[Dict[str, Any]] = None
    if with_sources:
        sources_dict = {}
        for field_name in user_fields:
            sources_dict[field_name] = data.get(f"{field_name}_source_lines")

    confidences_dict: Optional[Dict[str, Any]] = None
    if with_confidence:
        confidences_dict = {}
        for field_name in user_fields:
            confidences_dict[field_name] = data.get(f"{field_name}_confidence")

    return user_data, sources_dict, confidences_dict


def build_meta_schema(
    user_schema: Type[BaseModel],
    *,
    with_sources: bool = False,
    with_confidence: bool = False,
    confidence_config: Optional[ConfidenceConfig] = None,
) -> Type[BaseModel]:
    """Build a combined citations+confidence schema with NO user data fields.

    Used by ``multipass=True`` mode's pass 2 to get both source line numbers
    and confidence scores in a single LLM call.
    """
    field_defs: Dict[str, Any] = {}

    for field_name, _annotation, _default, _description, alias in _iter_user_fields(user_schema):
        display_name = alias or field_name
        if with_sources:
            source_desc = (
                f"Line numbers from the input text that support '{display_name}' "
                f"(e.g. [3, 7]). Use the Lnn prefixes to identify line numbers."
            )
            field_defs[f"{field_name}_source_lines"] = (
                Optional[List[int]],
                Field(None, description=source_desc),
            )

        if with_confidence and confidence_config is not None:
            if confidence_config.is_numeric:
                conf_type = Optional[int] if confidence_config.is_integer_scale else Optional[float]
                conf_desc = (
                    f"Confidence score for '{display_name}' "
                    f"({confidence_config.min_value}–{confidence_config.max_value})."
                )
            else:
                keys = tuple(confidence_config.scale.keys())
                conf_type = Optional[Literal[keys]]  # type: ignore[valid-type]
                conf_desc = f"Confidence level for '{display_name}' (one of {list(keys)})."
            field_defs[f"{field_name}_confidence"] = (
                conf_type,
                Field(None, description=conf_desc),
            )

    return create_model(f"{user_schema.__name__}Meta", **field_defs)


def build_meta_prompt(
    user_schema: Type[BaseModel],
    extracted_data: Dict[str, Any],
    *,
    with_sources: bool = False,
    with_confidence: bool = False,
    confidence_config: Optional[ConfidenceConfig] = None,
) -> str:
    """Build combined citations+confidence prompt for multipass mode pass 2.

    Shows the already-extracted values and asks for source line numbers
    and/or confidence scores. Does NOT embed source text (it is sent
    separately as user message content).
    """
    parts = []

    if with_sources:
        parts.append(
            "Find the source lines and rate your confidence for each extracted field value.\n"
            if with_confidence
            else "Find the source lines that support each extracted field value.\n"
        )
        parts.append(
            "The input text has line numbers prefixed as 'Lnn: ' at the start of each line.\n"
            "For each field, provide a list of line numbers (integers) where the supporting text appears.\n"
            "Example: if line 'L03: Invoice #12345' supports a field, include 3 in that field's list.\n"
            "If a field's value is null/None, set its source_lines to null.\n"
        )
    elif with_confidence:
        parts.append("Rate your confidence for each extracted field value.\n")

    if with_confidence and confidence_config is not None:
        if confidence_config.is_numeric:
            parts.append(
                f"For each field, provide a confidence score between "
                f"{confidence_config.min_value} and {confidence_config.max_value}.\n"
            )
        else:
            keys = list(confidence_config.scale.keys())
            parts.append(f"For each field, provide a confidence level from: {keys}.\n")

        has_descriptions = any(v is not None for v in confidence_config.scale.values())
        if has_descriptions:
            parts.append("Use the following scale as guidance:\n")
            for value, desc in confidence_config.scale.items():
                if desc is not None:
                    parts.append(f"- {value}: {desc}\n")

        parts.append("If a field's value is null/None, set its confidence to null.\n")

    parts.append("\nExtracted values:\n")
    for field_name, value in extracted_data.items():
        parts.append(f"  {field_name}: {value!r}\n")

    return "".join(parts)


# ------------------------------------------------------------------ #
# Legacy wrappers — delegate to the extended builders
# ------------------------------------------------------------------ #


def add_line_numbers(text: str) -> Tuple[str, Dict[int, str]]:
    """Prefix each line with L{nn}: and return a line_map.

    Args:
        text: The raw layout text.

    Returns:
        (numbered_text, line_map) where line_map maps 0-based line index
        to the original line content.
    """
    lines = text.split("\n")
    total = len(lines)
    width = len(str(total))  # dynamic zero-padding

    numbered_lines = []
    line_map: Dict[int, str] = {}
    for i, line in enumerate(lines):
        line_map[i] = line
        prefix = f"L{str(i).zfill(width)}"
        numbered_lines.append(f"{prefix}: {line}")

    return "\n".join(numbered_lines), line_map


def build_shadow_schema(user_schema: Type[BaseModel]) -> Type[BaseModel]:
    """Create a shadow schema that adds {field}_source_lines: list[int] for each field.

    Delegates to :func:`build_extended_schema` with ``with_sources=True``.
    """
    return build_extended_schema(user_schema, with_sources=True)


def build_citation_prompt(user_prompt: Optional[str], user_schema: Type[BaseModel]) -> str:
    """Build the prompt that instructs the LLM to provide source line numbers.

    Delegates to :func:`build_extended_prompt` with ``with_sources=True``.
    """
    return build_extended_prompt(user_prompt, user_schema, with_sources=True)


def build_char_to_element_map(word_elements: list) -> Dict[int, Any]:
    """Build a mapping from id(char_dict) to parent TextElement.

    Args:
        word_elements: List of TextElement word objects.

    Returns:
        Dict mapping id(char_dict) -> TextElement for every char_dict
        in every word's _char_dicts.
    """
    mapping: Dict[int, Any] = {}
    for word in word_elements:
        for cd in getattr(word, "_char_dicts", []):
            mapping[id(cd)] = word
    return mapping


def _parse_line_prefix(quote: str) -> Tuple[Optional[int], str]:
    """Parse a line prefix like 'L03: ...' from a quote.

    Returns:
        (line_number, remaining_text) or (None, original_quote) if no prefix found.
    """
    m = re.match(r"^L(\d+):\s*(.*)$", quote.strip())
    if m:
        return int(m.group(1)), m.group(2)
    return None, quote.strip()


def _find_best_match_in_textmap(
    text_to_find: str,
    textmap: Any,
    line_map: Dict[int, str],
    line_number: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Search for text in a TextMap, optionally constrained by line number.

    Returns list of char_dicts from the best match, or empty list.
    """
    if textmap is None or not text_to_find.strip():
        return []

    # Try exact search first
    escaped = re.escape(text_to_find.strip())
    results = textmap.search(escaped, regex=True, return_chars=True)
    if results:
        # If we have a line number hint, prefer matches near that line
        if line_number is not None and len(results) > 1:
            # Use the line content to disambiguate
            line_content = line_map.get(line_number, "")
            for r in results:
                if text_to_find.strip() in line_content:
                    return r.get("chars", [])
        return results[0].get("chars", [])

    # Fallback: try fuzzy matching against each line if we have a line number
    if line_number is not None:
        line_content = line_map.get(line_number, "")
        if line_content.strip():
            escaped_line = re.escape(line_content.strip())
            results = textmap.search(escaped_line, regex=True, return_chars=True)
            if results:
                return results[0].get("chars", [])

    # Last resort: try searching for a substring
    words = text_to_find.strip().split()
    if len(words) >= 3:
        # Try first few words
        partial = re.escape(" ".join(words[:3]))
        results = textmap.search(partial, regex=True, return_chars=True)
        if results:
            return results[0].get("chars", [])

    return []


def resolve_citations(
    shadow_data: Any,
    user_schema: Type[BaseModel],
    line_map: Dict[int, str],
    textmap_info: Any,
    char_to_element_map: Dict[int, Any],
) -> Dict[str, Any]:
    """Resolve source line numbers to ElementCollections.

    Args:
        shadow_data: Dict or model with ``{field}_source_lines`` fields
            containing lists of line numbers (integers).
        user_schema: The original user schema (to identify field names).
        line_map: Mapping from line index to original line content.
        textmap_info: Either a single TextMap or list[PageTextMapInfo].
        char_to_element_map: Mapping from id(char_dict) to TextElement.

    Returns:
        Dict mapping field_name -> ElementCollection of source TextElements.
    """
    from natural_pdf.elements.element_collection import ElementCollection

    # Get data as dict
    if isinstance(shadow_data, dict):
        shadow_dict = shadow_data
    elif hasattr(shadow_data, "model_dump"):
        shadow_dict = shadow_data.model_dump()
    else:
        shadow_dict = shadow_data.dict()

    # Identify user fields
    if hasattr(user_schema, "model_fields"):
        user_fields = set(user_schema.model_fields.keys())
    else:
        user_fields = set(user_schema.__fields__.keys())

    # Determine if multi-page
    is_multi_page = isinstance(textmap_info, list)

    citations: Dict[str, Any] = {}

    for field_name in user_fields:
        source_key = f"{field_name}_source_lines"
        source_lines = shadow_dict.get(source_key)
        if not source_lines:
            citations[field_name] = ElementCollection([])
            continue

        matched_elements = set()  # Use set to avoid duplicates (by id)
        matched_elements_list = []  # Ordered list

        for line_num in source_lines:
            # Look up line text from line_map
            line_text = line_map.get(line_num, "")
            if not line_text.strip():
                logger.debug(
                    "Citation line %s for field '%s' not found in line_map (max=%s).",
                    line_num,
                    field_name,
                    max(line_map.keys()) if line_map else "empty",
                )
                continue

            if is_multi_page:
                # Find the right page's TextMap by line number
                textmap = None
                if line_num is not None:
                    for pinfo in textmap_info:
                        if pinfo.line_start <= line_num < pinfo.line_end:
                            textmap = pinfo.textmap
                            break
                if textmap is None and textmap_info:
                    # Fallback: try all pages
                    for pinfo in textmap_info:
                        textmap = pinfo.textmap
                        chars = _find_best_match_in_textmap(line_text, textmap, line_map, line_num)
                        if chars:
                            break
                    else:
                        continue
                else:
                    chars = _find_best_match_in_textmap(line_text, textmap, line_map, line_num)
            else:
                # Single TextMap (Page or Region)
                chars = _find_best_match_in_textmap(line_text, textmap_info, line_map, line_num)

            # Map char_dicts to TextElements
            for cd in chars:
                elem = char_to_element_map.get(id(cd))
                if elem is not None and id(elem) not in matched_elements:
                    matched_elements.add(id(elem))
                    matched_elements_list.append(elem)

        citations[field_name] = ElementCollection(matched_elements_list)

    return citations


def resolve_source_lines_to_text(
    sources_dict: Dict[str, Any],
    line_map: Dict[int, str],
) -> Dict[str, Any]:
    """Convert line-number lists to human-readable source text lists.

    Args:
        sources_dict: Mapping from field name to list of line numbers.
        line_map: Mapping from line index to original line content.

    Returns:
        Dict mapping field_name -> list of source text strings, or None.
    """
    result: Dict[str, Any] = {}
    for field_name, line_nums in sources_dict.items():
        if line_nums is None:
            result[field_name] = None
        else:
            result[field_name] = [
                line_map.get(n, "") for n in line_nums if line_map.get(n, "").strip()
            ]
    return result


def split_shadow_result(shadow_data: BaseModel, user_schema: Type[BaseModel]) -> BaseModel:
    """Extract only user's original fields from the shadow result.

    Constructs a new instance of the user schema with values from the shadow data.
    Delegates to :func:`split_extended_result`.
    """
    if hasattr(shadow_data, "model_dump"):
        shadow_dict = shadow_data.model_dump()
    else:
        shadow_dict = shadow_data.dict()

    user_data, _, _ = split_extended_result(
        shadow_dict, user_schema, with_sources=False, with_confidence=False
    )
    return user_schema(**user_data)
