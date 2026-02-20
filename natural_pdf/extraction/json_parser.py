"""Fuzzy JSON extraction and Pydantic validation utilities.

Used by the VLM adapter and the structured output ladder to parse
free-text model responses into validated Pydantic models.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _find_balanced_braces(text: str) -> Optional[str]:
    """Find the first balanced ``{…}`` span using brace-depth counting."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            if in_string:
                escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_json_from_text(text: str) -> str:
    """Find JSON in model output (```json blocks, {…} spans).

    Returns the extracted JSON string, or raises ``ValueError`` if no
    JSON-like content can be found.
    """
    # Try ```json ... ``` fenced blocks first (must be labeled "json")
    match = re.search(r"```json\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try unlabeled fences only when content looks like JSON
    match = re.search(r"```\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        content = match.group(1).strip()
        if content.startswith("{"):
            return content

    # Try bare {…} spans using balanced brace matching
    balanced = _find_balanced_braces(text)
    if balanced is not None:
        return balanced

    raise ValueError(f"No JSON-like content found in text: {text[:200]!r}")


def parse_json_response(text: str, schema: Type[T]) -> T:
    """Parse free-text into a Pydantic model.

    Tries direct JSON parse first, then falls back to fuzzy extraction.
    Raises ``ValueError`` on failure.
    """
    # Try direct parse
    try:
        data = json.loads(text)
        return schema.model_validate(data)
    except (json.JSONDecodeError, ValidationError):
        pass

    # Try fuzzy extraction
    try:
        json_str = extract_json_from_text(text)
        data = json.loads(json_str)
        return schema.model_validate(data)
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise ValueError(
            f"Failed to parse response into {schema.__name__}: {exc}\n" f"Raw text: {text[:500]!r}"
        ) from exc
