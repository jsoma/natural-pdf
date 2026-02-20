"""Tests for natural_pdf.extraction.json_parser."""

import pytest
from pydantic import BaseModel

from natural_pdf.extraction.json_parser import extract_json_from_text, parse_json_response


class SimpleSchema(BaseModel):
    name: str
    age: int


# ---- extract_json_from_text ----


def test_extract_json_from_fenced_block():
    text = 'Here is the result:\n```json\n{"name": "Alice", "age": 30}\n```\nDone.'
    assert extract_json_from_text(text) == '{"name": "Alice", "age": 30}'


def test_extract_json_from_bare_braces():
    text = 'The answer is {"name": "Bob", "age": 25} and that is it.'
    assert extract_json_from_text(text) == '{"name": "Bob", "age": 25}'


def test_extract_json_from_fenced_no_lang():
    text = '```\n{"name": "Charlie", "age": 40}\n```'
    assert extract_json_from_text(text) == '{"name": "Charlie", "age": 40}'


def test_extract_json_non_json_fence_falls_through():
    """A ```yaml fence should not be returned; fallback to brace matching."""
    text = '```yaml\nkey: value\n```\nHere is JSON: {"name": "Alice", "age": 30}'
    assert extract_json_from_text(text) == '{"name": "Alice", "age": 30}'


def test_extract_json_raises_on_no_json():
    with pytest.raises(ValueError, match="No JSON-like content"):
        extract_json_from_text("This has no json at all")


# ---- parse_json_response ----


def test_parse_json_direct():
    text = '{"name": "Alice", "age": 30}'
    result = parse_json_response(text, SimpleSchema)
    assert result.name == "Alice"
    assert result.age == 30


def test_parse_json_with_fenced_block():
    text = '```json\n{"name": "Bob", "age": 25}\n```'
    result = parse_json_response(text, SimpleSchema)
    assert result.name == "Bob"
    assert result.age == 25


def test_parse_json_with_surrounding_text():
    text = 'Here is the data: {"name": "Charlie", "age": 40}. Hope that helps!'
    result = parse_json_response(text, SimpleSchema)
    assert result.name == "Charlie"
    assert result.age == 40


def test_parse_json_raises_on_invalid():
    with pytest.raises(ValueError, match="Failed to parse"):
        parse_json_response("not json at all", SimpleSchema)


def test_parse_json_raises_on_schema_mismatch():
    # Valid JSON but missing required fields
    with pytest.raises(ValueError, match="Failed to parse"):
        parse_json_response('{"foo": "bar"}', SimpleSchema)


# ---- Brace-balancing tests ----


def test_extract_json_multiple_objects():
    """With multiple JSON objects, should extract the first complete one."""
    text = '{"name": "Alice", "age": 30} some text {"name": "Bob", "age": 25}'
    result = extract_json_from_text(text)
    assert result == '{"name": "Alice", "age": 30}'


def test_extract_json_nested_objects():
    """Nested JSON should extract the full outer object."""
    text = 'result: {"outer": {"inner": 1}, "val": 2} done'
    result = extract_json_from_text(text)
    assert result == '{"outer": {"inner": 1}, "val": 2}'


def test_extract_json_braces_in_strings():
    """Braces inside JSON string values should not confuse the parser."""
    text = 'data: {"msg": "hello {world}"} end'
    result = extract_json_from_text(text)
    assert result == '{"msg": "hello {world}"}'


def test_extract_json_escaped_quotes():
    """Escaped quotes inside strings should be handled correctly."""
    text = r'data: {"msg": "say \"hi\""} end'
    result = extract_json_from_text(text)
    assert result == r'{"msg": "say \"hi\""}'
