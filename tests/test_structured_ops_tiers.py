"""Tests for the tiered structured output ladder in structured_ops.py."""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from natural_pdf.extraction.result import StructuredDataResult
from natural_pdf.extraction.structured_ops import extract_structured_data


class SimpleSchema(BaseModel):
    answer: Optional[str] = Field(None, description="The answer")


def _make_client(
    *,
    tier1_result=None,
    tier1_error=None,
    tier2_result=None,
    tier2_error=None,
    tier3_result=None,
    tier3_error=None,
):
    """Build a mock client that can selectively succeed/fail at each tier."""
    client = MagicMock()

    # Tier 1: beta.chat.completions.parse
    if tier1_error:
        client.beta.chat.completions.parse.side_effect = tier1_error
    elif tier1_result:
        completion = MagicMock()
        completion.choices = [MagicMock()]
        completion.choices[0].message.parsed = tier1_result
        client.beta.chat.completions.parse.return_value = completion
    else:
        # No Tier 1 support — getattr chain returns None
        client.beta = None

    # Tier 2+3: chat.completions.create
    call_count = {"n": 0}
    tier2_err = tier2_error
    tier3_err = tier3_error

    def create_side_effect(**kwargs):
        call_count["n"] += 1
        is_json_mode = kwargs.get("response_format") == {"type": "json_object"}

        if is_json_mode:
            # Tier 2
            if tier2_err:
                raise tier2_err
            if tier2_result is not None:
                completion = MagicMock()
                completion.choices = [MagicMock()]
                completion.choices[0].message.content = tier2_result
                return completion
            raise TypeError("json_object not supported")
        else:
            # Tier 3
            if tier3_err:
                raise tier3_err
            if tier3_result is not None:
                completion = MagicMock()
                completion.choices = [MagicMock()]
                completion.choices[0].message.content = tier3_result
                return completion
            raise RuntimeError("tier3 failed")

    client.chat.completions.create.side_effect = create_side_effect
    return client


class TestTier1:
    def test_tier1_success(self):
        instance = SimpleSchema(answer="hello")
        client = _make_client(tier1_result=instance)
        result = extract_structured_data(content="test", schema=SimpleSchema, client=client)
        assert result.success is True
        assert result.data.answer == "hello"

    def test_tier1_missing_parse_falls_to_tier2(self):
        """When client has no beta.parse, should fall to Tier 2."""
        client = _make_client(
            tier2_result='{"answer": "from-tier2"}',
        )
        # Default _make_client sets client.beta = None (no Tier 1)
        result = extract_structured_data(content="test", schema=SimpleSchema, client=client)
        assert result.success is True
        assert result.data.answer == "from-tier2"

    def test_tier1_attribute_error_propagates(self):
        """AttributeError from parse call itself should propagate (real bug)."""
        client = _make_client(
            tier1_error=AttributeError("bad response shape"),
        )
        with pytest.raises(AttributeError, match="bad response shape"):
            extract_structured_data(content="test", schema=SimpleSchema, client=client)

    def test_tier1_connection_error_reraises(self):
        """ConnectionError should NOT be swallowed — re-raise immediately."""
        client = _make_client(tier1_error=ConnectionError("refused"))
        with pytest.raises(ConnectionError, match="refused"):
            extract_structured_data(content="test", schema=SimpleSchema, client=client)

    def test_tier1_timeout_error_reraises(self):
        """TimeoutError should NOT be swallowed — re-raise immediately."""
        client = _make_client(tier1_error=TimeoutError("timed out"))
        with pytest.raises(TimeoutError, match="timed out"):
            extract_structured_data(content="test", schema=SimpleSchema, client=client)


class TestTier2:
    def test_tier2_success(self):
        client = _make_client(tier2_result='{"answer": "tier2-ok"}')
        result = extract_structured_data(content="test", schema=SimpleSchema, client=client)
        assert result.success is True
        assert result.data.answer == "tier2-ok"

    def test_tier2_json_parse_failure_falls_to_tier3(self):
        """ValueError from parse_json_response should fall to Tier 3."""
        client = _make_client(
            tier2_result="not valid json {{{",
            tier3_result='{"answer": "from-tier3"}',
        )
        result = extract_structured_data(content="test", schema=SimpleSchema, client=client)
        assert result.success is True
        assert result.data.answer == "from-tier3"

    def test_tier2_connection_error_reraises(self):
        """ConnectionError in Tier 2 should re-raise."""
        client = _make_client(tier2_error=ConnectionError("refused"))
        with pytest.raises(ConnectionError, match="refused"):
            extract_structured_data(content="test", schema=SimpleSchema, client=client)


class TestTier3:
    def test_tier3_success(self):
        client = _make_client(
            tier2_error=TypeError("no json_object"),
            tier3_result='{"answer": "tier3-ok"}',
        )
        result = extract_structured_data(content="test", schema=SimpleSchema, client=client)
        assert result.success is True
        assert result.data.answer == "tier3-ok"

    def test_tier3_failure_preserves_raw_text(self):
        """When all tiers fail, Tier 3 should preserve raw_text in raw_output."""
        client = _make_client(
            tier2_error=TypeError("no json_object"),
            tier3_result="This is not JSON at all, just random text",
        )
        result = extract_structured_data(content="test", schema=SimpleSchema, client=client)
        assert result.success is False
        assert result.raw_output == "This is not JSON at all, just random text"

    def test_tier3_connection_error_reraises(self):
        """ConnectionError in Tier 3 should re-raise, not be swallowed."""
        client = _make_client(
            tier2_error=TypeError("no json_object"),
            tier3_error=ConnectionError("refused"),
        )
        with pytest.raises(ConnectionError, match="refused"):
            extract_structured_data(content="test", schema=SimpleSchema, client=client)

    def test_all_tiers_fail(self):
        """When everything fails, result should have success=False."""
        client = _make_client(
            tier2_error=TypeError("no json_object"),
            tier3_error=RuntimeError("total failure"),
        )
        result = extract_structured_data(content="test", schema=SimpleSchema, client=client)
        assert result.success is False
        assert "All structured output tiers failed" in result.error_message
