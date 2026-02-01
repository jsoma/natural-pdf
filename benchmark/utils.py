"""
Shared Utilities for Benchmark System

Centralized functions for:
- Text normalization (MS Word smart characters)
- Value comparison
- Column/header name normalization
- Data format detection
"""

from typing import Literal

DataFormat = Literal["tabular", "structured"]


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison - handle MS Word smart characters.

    Handles:
    - Curly quotes -> straight quotes
    - En-dash, em-dash, minus sign -> hyphen
    - Non-breaking space, thin space -> regular space
    - Zero-width space -> removed
    - Ellipsis character -> three dots

    Does NOT change case - that's intentional for exact matching.
    """
    text = str(text)  # Ensure input is a string

    # Normalize apostrophes/quotes (curly to straight) - use Unicode escapes
    text = text.replace("\u2019", "'").replace("\u2018", "'")  # curly single quotes
    text = text.replace("\u201c", '"').replace("\u201d", '"')  # curly double quotes

    # Normalize dashes/hyphens
    text = text.replace("\u2013", "-").replace("\u2014", "-")  # en-dash, em-dash
    text = text.replace("\u2212", "-")  # minus sign

    # Normalize spaces (non-breaking space, etc.)
    text = text.replace("\u00a0", " ")  # non-breaking space
    text = text.replace("\u2009", " ")  # thin space
    text = text.replace("\u200b", "")  # zero-width space

    # Normalize ellipsis
    text = text.replace("\u2026", "...")

    # Trim and consolidate whitespace
    return " ".join(text.strip().split())


def strip_whitespace(text: str) -> str:
    """
    Remove all whitespace from text for comparison.

    Used for trap checking where whitespace differences shouldn't matter:
    - "cat\\ndog" matches "cat dog"
    - "(444)123-4567" matches "(444) 123-4567"
    """
    import re

    return re.sub(r"\s+", "", normalize_text(text))


# Values that represent "no data" / "missing" / "empty"
EMPTY_VALUES = frozenset(
    [
        "",
        "-",
        "--",
        "n/a",
        "na",
        "none",
        "null",
        "missing",
        "unknown",
        "not available",
        "not applicable",
        ".",
    ]
)


def is_empty_value(value: str) -> bool:
    """
    Check if a value represents empty/missing data.

    Treats values like "N/A", "-", "none", "missing" as empty.
    """
    if not value:
        return True
    normalized = normalize_text(value).lower().strip()
    return normalized in EMPTY_VALUES


def values_match(expected: str, actual: str, case_sensitive: bool = False) -> bool:
    """
    Check if two values match after normalization.

    Args:
        expected: Ground truth value
        actual: LLM extracted value
        case_sensitive: If True, comparison is case-sensitive (default: False)

    Returns:
        True if values match after normalization
    """
    # No ground truth to compare against
    if not expected:
        return True

    exp_norm = normalize_text(expected)
    act_norm = normalize_text(actual)

    # Both empty/missing values are considered a match
    if is_empty_value(exp_norm) and is_empty_value(act_norm):
        return True

    if case_sensitive:
        return exp_norm == act_norm
    else:
        return exp_norm.lower() == act_norm.lower()


def normalize_header(header: str) -> str:
    """
    Normalize a column/field header for matching.

    Handles variations like:
    - "LICENSE NUMBER" vs "license_number" vs "License Number"
    - "Repeat?" vs "repeat" vs "Repeat"
    """
    return str(header).lower().replace(" ", "_").replace("?", "").replace("-", "_").strip()


def detect_data_format(page_data: list[dict]) -> DataFormat:
    """
    Detect whether page data is tabular or structured.

    Tabular: Multiple rows of similar flat dictionaries (like CSV rows)
    Structured: Single dict with nested objects/arrays (like JSON with form_fields + violations)

    Args:
        page_data: List of dictionaries from a page

    Returns:
        "tabular" or "structured"
    """
    if not page_data:
        return "structured"  # Default for empty data

    # Multiple rows = tabular
    if len(page_data) > 1:
        return "tabular"

    # Single row - check if it has nested structures
    first_row = page_data[0]
    has_nested = any(isinstance(v, (list, dict)) for v in first_row.values())

    if has_nested:
        return "structured"
    else:
        return "tabular"


def find_matching_column(target_header: str, available_headers: list[str]) -> str:
    """
    Find a matching column header from available headers.

    Args:
        target_header: The header we're looking for
        available_headers: List of headers to search in

    Returns:
        The matching header from available_headers, or empty string if not found
    """
    target_norm = normalize_header(target_header)

    # Try exact match first
    for h in available_headers:
        if h == target_header:
            return h

    # Try normalized match
    for h in available_headers:
        if normalize_header(h) == target_norm:
            return h

    return ""
