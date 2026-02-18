"""Citation/grounding support for structured extraction.

Maps extracted field values back to source locations in the PDF by:
1. Sending line-numbered text to the LLM
2. Getting verbatim quotes with line prefixes back via a shadow schema
3. Aligning quotes to source text via pdfplumber's TextMap provenance data
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel, Field, create_model

logger = logging.getLogger(__name__)


@dataclass
class PageTextMapInfo:
    """Tracks per-page TextMap and line range for multi-page citation resolution."""

    page_number: int  # 1-indexed
    textmap: Any  # pdfplumber TextMap (or None)
    word_elements: list  # TextElement words for this page
    line_start: int  # first line in unified text (0-based)
    line_end: int  # last line (exclusive)


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
    """Create a shadow schema that adds {field}_source: list[str] for each field.

    The shadow schema copies all original fields and adds source fields
    so the LLM can return verbatim quotes alongside extracted values.
    """
    fields_iter = (
        user_schema.model_fields.items()
        if hasattr(user_schema, "model_fields")
        else user_schema.__fields__.items()
    )

    field_defs: Dict[str, Any] = {}
    for field_name, field_obj in fields_iter:
        # Copy original field
        if hasattr(user_schema, "model_fields"):
            # Pydantic v2
            annotation = field_obj.annotation
            default = field_obj.default
            description = field_obj.description
        else:
            # Pydantic v1
            annotation = field_obj.outer_type_
            default = field_obj.default
            description = field_obj.field_info.description if field_obj.field_info else None

        if default is not None:
            field_defs[field_name] = (
                annotation,
                Field(default, description=description),
            )
        else:
            field_defs[field_name] = (
                annotation,
                Field(description=description),
            )

        # Add source field
        source_desc = (
            f"Verbatim source quotes for '{field_name}' with line prefixes "
            f"(e.g. ['L03: Invoice #12345']). Copy text exactly as it appears."
        )
        field_defs[f"{field_name}_source"] = (
            Optional[List[str]],
            Field(None, description=source_desc),
        )

    return create_model(f"{user_schema.__name__}WithSources", **field_defs)


def build_citation_prompt(user_prompt: Optional[str], user_schema: Type[BaseModel]) -> str:
    """Build the prompt that instructs the LLM to provide source quotes.

    Appends citation instructions to the user's existing prompt (or default).
    """
    base = user_prompt or (
        f"Extract the information corresponding to the fields in the "
        f"{user_schema.__name__} schema. Respond only with the structured data."
    )

    citation_instructions = (
        "\n\nIMPORTANT - Citation instructions:\n"
        "The input text has line numbers prefixed as 'Lnn: ' at the start of each line.\n"
        "For each field, also fill in the corresponding '_source' field with a list of "
        "verbatim quotes from the text that support the extracted value.\n"
        "Each quote MUST:\n"
        "- Start with the line prefix (e.g. 'L03: ')\n"
        "- Copy the relevant text from that line EXACTLY as it appears\n"
        "- Be a separate string in the list for each line referenced\n"
        "Example: if line 'L03: Invoice #12345' contains the invoice number, "
        "set invoice_number_source to ['L03: Invoice #12345'].\n"
        "If a value spans multiple lines, include each line as a separate entry."
    )

    return base + citation_instructions


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
    shadow_data: BaseModel,
    user_schema: Type[BaseModel],
    line_map: Dict[int, str],
    textmap_info: Any,
    char_to_element_map: Dict[int, Any],
) -> Dict[str, Any]:
    """Resolve shadow schema source fields to ElementCollections.

    Args:
        shadow_data: The shadow schema instance with _source fields.
        user_schema: The original user schema (to identify field names).
        line_map: Mapping from line index to original line content.
        textmap_info: Either a single TextMap or list[PageTextMapInfo].
        char_to_element_map: Mapping from id(char_dict) to TextElement.

    Returns:
        Dict mapping field_name -> ElementCollection of source TextElements.
    """
    from natural_pdf.elements.element_collection import ElementCollection

    # Get shadow data as dict
    if hasattr(shadow_data, "model_dump"):
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
        source_key = f"{field_name}_source"
        source_quotes = shadow_dict.get(source_key)
        if not source_quotes:
            citations[field_name] = ElementCollection([])
            continue

        matched_elements = set()  # Use set to avoid duplicates (by id)
        matched_elements_list = []  # Ordered list

        for quote in source_quotes:
            line_num, quote_text = _parse_line_prefix(quote)

            if is_multi_page:
                # Find the right page's TextMap by line number
                textmap = None
                page_char_map = char_to_element_map
                if line_num is not None:
                    for pinfo in textmap_info:
                        if pinfo.line_start <= line_num < pinfo.line_end:
                            textmap = pinfo.textmap
                            break
                if textmap is None and textmap_info:
                    # Fallback: try all pages
                    for pinfo in textmap_info:
                        textmap = pinfo.textmap
                        chars = _find_best_match_in_textmap(quote_text, textmap, line_map, line_num)
                        if chars:
                            break
                    else:
                        continue
                else:
                    chars = _find_best_match_in_textmap(quote_text, textmap, line_map, line_num)
            else:
                # Single TextMap (Page or Region)
                chars = _find_best_match_in_textmap(quote_text, textmap_info, line_map, line_num)

            # Map char_dicts to TextElements
            for cd in chars:
                elem = char_to_element_map.get(id(cd))
                if elem is not None and id(elem) not in matched_elements:
                    matched_elements.add(id(elem))
                    matched_elements_list.append(elem)

        citations[field_name] = ElementCollection(matched_elements_list)

    return citations


def split_shadow_result(shadow_data: BaseModel, user_schema: Type[BaseModel]) -> BaseModel:
    """Extract only user's original fields from the shadow result.

    Constructs a new instance of the user schema with values from the shadow data.
    """
    if hasattr(shadow_data, "model_dump"):
        shadow_dict = shadow_data.model_dump()
    else:
        shadow_dict = shadow_data.dict()

    if hasattr(user_schema, "model_fields"):
        user_fields = set(user_schema.model_fields.keys())
    else:
        user_fields = set(user_schema.__fields__.keys())

    user_data = {k: v for k, v in shadow_dict.items() if k in user_fields}
    return user_schema(**user_data)
