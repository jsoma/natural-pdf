"""Tests for per-field confidence scoring in structured extraction."""

import logging
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from natural_pdf.extraction.citations import (
    DEFAULT_CONFIDENCE_SCALE,
    ConfidenceConfig,
    build_confidence_prompt,
    build_confidence_schema,
    build_extended_prompt,
    build_extended_schema,
    normalize_confidence_config,
    split_extended_result,
)
from natural_pdf.extraction.result import FieldResult, StructuredDataResult


# ------------------------------------------------------------------ #
# Test schemas
# ------------------------------------------------------------------ #
class SimpleSchema(BaseModel):
    name: str
    age: str


class InvoiceSchema(BaseModel):
    invoice_number: str = Field(description="The invoice number")
    date: Optional[str] = None
    total: str = Field(description="Total amount")


# ------------------------------------------------------------------ #
# normalize_confidence_config
# ------------------------------------------------------------------ #
class TestNormalizeConfidenceConfig:
    def test_none_returns_none(self):
        assert normalize_confidence_config(None) is None

    def test_false_returns_none(self):
        assert normalize_confidence_config(False) is None

    def test_true_returns_default_numeric(self):
        cfg = normalize_confidence_config(True)
        assert cfg is not None
        assert cfg.is_numeric is True
        assert cfg.min_value == 1
        assert cfg.max_value == 5
        assert cfg.scale == DEFAULT_CONFIDENCE_SCALE

    def test_range_string_returns_default_numeric(self):
        cfg = normalize_confidence_config("range")
        assert cfg is not None
        assert cfg.is_numeric is True
        assert cfg.min_value == 1
        assert cfg.max_value == 5

    def test_list_categorical(self):
        cfg = normalize_confidence_config(["low", "medium", "high"])
        assert cfg is not None
        assert cfg.is_numeric is False
        assert list(cfg.scale.keys()) == ["low", "medium", "high"]
        assert all(v is None for v in cfg.scale.values())

    def test_list_numeric(self):
        cfg = normalize_confidence_config([1, 2, 3, 4, 5])
        assert cfg is not None
        assert cfg.is_numeric is True
        assert cfg.min_value == 1.0
        assert cfg.max_value == 5.0

    def test_dict_numeric_with_descriptions(self):
        cfg = normalize_confidence_config(
            {
                0: "No confidence",
                5: "Medium",
                10: "Very confident",
            }
        )
        assert cfg is not None
        assert cfg.is_numeric is True
        assert cfg.min_value == 0.0
        assert cfg.max_value == 10.0
        assert cfg.scale[5] == "Medium"

    def test_dict_categorical_with_descriptions(self):
        cfg = normalize_confidence_config(
            {
                "low": "Not confident",
                "high": "Very confident",
            }
        )
        assert cfg is not None
        assert cfg.is_numeric is False
        assert cfg.scale["low"] == "Not confident"

    def test_custom_numeric_range(self):
        cfg = normalize_confidence_config(
            {
                1.0: "Low",
                3.0: "Medium",
                5.0: "High",
            }
        )
        assert cfg.min_value == 1.0
        assert cfg.max_value == 5.0

    def test_empty_dict_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            normalize_confidence_config({})

    def test_short_list_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            normalize_confidence_config(["only_one"])

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported confidence type"):
            normalize_confidence_config(42)


# ------------------------------------------------------------------ #
# build_confidence_schema (pass-2 schema)
# ------------------------------------------------------------------ #
class TestBuildConfidenceSchema:
    def test_has_only_confidence_fields(self):
        cfg = normalize_confidence_config(True)
        ConfSchema = build_confidence_schema(SimpleSchema, cfg)
        fields = set(ConfSchema.model_fields.keys())
        assert fields == {"name_confidence", "age_confidence"}

    def test_no_user_data_fields(self):
        cfg = normalize_confidence_config(True)
        ConfSchema = build_confidence_schema(SimpleSchema, cfg)
        fields = set(ConfSchema.model_fields.keys())
        assert "name" not in fields
        assert "age" not in fields

    def test_fields_are_optional_int(self):
        cfg = normalize_confidence_config(True)
        ConfSchema = build_confidence_schema(SimpleSchema, cfg)
        instance = ConfSchema(name_confidence=4, age_confidence=None)
        assert instance.name_confidence == 4
        assert instance.age_confidence is None

    def test_categorical_confidence_schema(self):
        cfg = normalize_confidence_config(["low", "medium", "high"])
        ConfSchema = build_confidence_schema(SimpleSchema, cfg)
        instance = ConfSchema(name_confidence="high", age_confidence="low")
        assert instance.name_confidence == "high"


# ------------------------------------------------------------------ #
# build_confidence_prompt (pass-2 prompt)
# ------------------------------------------------------------------ #
class TestBuildConfidencePrompt:
    def test_contains_extracted_values(self):
        cfg = normalize_confidence_config(True)
        prompt = build_confidence_prompt(
            SimpleSchema,
            {"name": "Alice", "age": "30"},
            cfg,
        )
        assert "Alice" in prompt
        assert "30" in prompt

    def test_no_embedded_source_text(self):
        """Source text is no longer embedded in the confidence prompt."""
        cfg = normalize_confidence_config(True)
        prompt = build_confidence_prompt(
            SimpleSchema,
            {"name": "Alice", "age": "30"},
            cfg,
        )
        assert "Source text:" not in prompt

    def test_contains_scale_description(self):
        cfg = normalize_confidence_config(True)
        prompt = build_confidence_prompt(
            SimpleSchema,
            {"name": "Alice", "age": "30"},
            cfg,
        )
        assert "Explicitly Supported" in prompt
        assert "Barely Related" in prompt
        assert "document support" in prompt

    def test_contains_abstention_rule(self):
        cfg = normalize_confidence_config(True)
        prompt = build_confidence_prompt(
            SimpleSchema,
            {"name": "Alice", "age": "30"},
            cfg,
        )
        assert "null" in prompt
        assert "outside knowledge" in prompt.lower()


# ------------------------------------------------------------------ #
# build_extended_schema (single-pass combined schema)
# ------------------------------------------------------------------ #
class TestBuildExtendedSchema:
    def test_sources_only(self):
        Extended = build_extended_schema(SimpleSchema, with_sources=True)
        fields = set(Extended.model_fields.keys())
        assert "name" in fields
        assert "name_source_lines" in fields
        assert "age_source_lines" in fields
        assert "name_confidence" not in fields

    def test_confidence_only_numeric(self):
        cfg = normalize_confidence_config(True)
        Extended = build_extended_schema(SimpleSchema, with_confidence=True, confidence_config=cfg)
        fields = set(Extended.model_fields.keys())
        assert "name" in fields
        assert "name_confidence" in fields
        assert "age_confidence" in fields
        assert "name_source_lines" not in fields

    def test_both_sources_and_confidence(self):
        cfg = normalize_confidence_config(True)
        Extended = build_extended_schema(
            SimpleSchema,
            with_sources=True,
            with_confidence=True,
            confidence_config=cfg,
        )
        fields = set(Extended.model_fields.keys())
        assert "name" in fields
        assert "name_source_lines" in fields
        assert "name_confidence" in fields
        assert "age_source_lines" in fields
        assert "age_confidence" in fields

    def test_categorical_confidence(self):
        cfg = normalize_confidence_config(["low", "medium", "high"])
        Extended = build_extended_schema(SimpleSchema, with_confidence=True, confidence_config=cfg)
        fields = set(Extended.model_fields.keys())
        assert "name_confidence" in fields

    def test_schema_instantiation_numeric(self):
        cfg = normalize_confidence_config(True)
        Extended = build_extended_schema(SimpleSchema, with_confidence=True, confidence_config=cfg)
        instance = Extended(name="Alice", age="30", name_confidence=4, age_confidence=3)
        assert instance.name == "Alice"
        assert instance.name_confidence == 4

    def test_schema_instantiation_null_confidence(self):
        cfg = normalize_confidence_config(True)
        Extended = build_extended_schema(SimpleSchema, with_confidence=True, confidence_config=cfg)
        instance = Extended(name="Alice", age="30", name_confidence=None, age_confidence=None)
        assert instance.name_confidence is None

    def test_schema_instantiation_categorical(self):
        cfg = normalize_confidence_config(["low", "medium", "high"])
        Extended = build_extended_schema(SimpleSchema, with_confidence=True, confidence_config=cfg)
        instance = Extended(name="Alice", age="30", name_confidence="high", age_confidence="low")
        assert instance.name_confidence == "high"

    def test_no_extensions(self):
        Extended = build_extended_schema(SimpleSchema)
        fields = set(Extended.model_fields.keys())
        assert fields == {"name", "age"}

    def test_confidence_type_is_optional_int(self):
        """Confidence fields use Optional[int], not confloat."""
        cfg = normalize_confidence_config(True)
        Extended = build_extended_schema(SimpleSchema, with_confidence=True, confidence_config=cfg)
        # Should accept out-of-range values without validation error
        instance = Extended(name="Alice", age="30", name_confidence=99, age_confidence=0)
        assert instance.name_confidence == 99

    def test_evidence_first_ordering(self):
        """Source lines appear before value in field order."""
        cfg = normalize_confidence_config(True)
        Extended = build_extended_schema(
            SimpleSchema,
            with_sources=True,
            with_confidence=True,
            confidence_config=cfg,
        )
        field_names = list(Extended.model_fields.keys())
        # For each user field, source_lines should come before value,
        # and value before confidence
        name_sl_idx = field_names.index("name_source_lines")
        name_idx = field_names.index("name")
        name_conf_idx = field_names.index("name_confidence")
        assert name_sl_idx < name_idx < name_conf_idx


# ------------------------------------------------------------------ #
# build_extended_prompt
# ------------------------------------------------------------------ #
class TestBuildExtendedPrompt:
    def test_base_only(self):
        prompt = build_extended_prompt(None, SimpleSchema)
        assert "SimpleSchema" in prompt
        assert "confidence" not in prompt.lower()
        assert "citation" not in prompt.lower()

    def test_instructions_appended(self):
        prompt = build_extended_prompt(
            "Extract data.", SimpleSchema, instructions="Be precise about dates."
        )
        assert prompt.startswith("Extract data.")
        assert "Be precise about dates." in prompt

    def test_citation_block(self):
        prompt = build_extended_prompt(None, SimpleSchema, with_sources=True)
        assert "_source_lines" in prompt
        assert "line numbers" in prompt.lower()

    def test_confidence_block_numeric(self):
        cfg = normalize_confidence_config(True)
        prompt = build_extended_prompt(
            None, SimpleSchema, with_confidence=True, confidence_config=cfg
        )
        assert "_confidence" in prompt
        assert "1" in prompt
        assert "5" in prompt

    def test_confidence_block_categorical(self):
        cfg = normalize_confidence_config(
            {
                "low": "Not confident",
                "high": "Very confident",
            }
        )
        prompt = build_extended_prompt(
            None, SimpleSchema, with_confidence=True, confidence_config=cfg
        )
        assert "low" in prompt
        assert "high" in prompt
        assert "Not confident" in prompt

    def test_assembly_order(self):
        cfg = normalize_confidence_config(True)
        prompt = build_extended_prompt(
            "Base prompt.",
            SimpleSchema,
            instructions="Instructions here.",
            with_sources=True,
            with_confidence=True,
            confidence_config=cfg,
        )
        base_idx = prompt.index("Base prompt.")
        instr_idx = prompt.index("Instructions here.")
        cite_idx = prompt.index("Citation instructions")
        conf_idx = prompt.index("Confidence scoring")
        assert base_idx < instr_idx < cite_idx < conf_idx

    def test_confidence_null_guidance(self):
        cfg = normalize_confidence_config(True)
        prompt = build_extended_prompt(
            None, SimpleSchema, with_confidence=True, confidence_config=cfg
        )
        assert "null" in prompt.lower()

    def test_numeric_scale_descriptions_included(self):
        cfg = normalize_confidence_config(True)
        prompt = build_extended_prompt(
            None, SimpleSchema, with_confidence=True, confidence_config=cfg
        )
        # Default scale should have descriptions
        assert "Explicitly supported" in prompt or "directly and clearly stated" in prompt


# ------------------------------------------------------------------ #
# split_extended_result
# ------------------------------------------------------------------ #
class TestSplitExtendedResult:
    def test_confidence_only(self):
        data = {
            "name": "Alice",
            "age": "30",
            "name_confidence": 4,
            "age_confidence": 3,
        }
        user_data, sources, confidences = split_extended_result(
            data, SimpleSchema, with_confidence=True
        )
        assert user_data == {"name": "Alice", "age": "30"}
        assert sources is None
        assert confidences == {"name": 4, "age": 3}

    def test_sources_only(self):
        data = {
            "name": "Alice",
            "age": "30",
            "name_source_lines": [0],
            "age_source_lines": [1],
        }
        user_data, sources, confidences = split_extended_result(
            data, SimpleSchema, with_sources=True
        )
        assert user_data == {"name": "Alice", "age": "30"}
        assert sources == {"name": [0], "age": [1]}
        assert confidences is None

    def test_both_sources_and_confidence(self):
        data = {
            "name": "Alice",
            "age": "30",
            "name_source_lines": [0],
            "age_source_lines": None,
            "name_confidence": 5,
            "age_confidence": 3,
        }
        user_data, sources, confidences = split_extended_result(
            data, SimpleSchema, with_sources=True, with_confidence=True
        )
        assert user_data == {"name": "Alice", "age": "30"}
        assert sources["name"] == [0]
        assert sources["age"] is None
        assert confidences["name"] == 5
        assert confidences["age"] == 3

    def test_neither(self):
        data = {"name": "Alice", "age": "30"}
        user_data, sources, confidences = split_extended_result(data, SimpleSchema)
        assert user_data == {"name": "Alice", "age": "30"}
        assert sources is None
        assert confidences is None


# ------------------------------------------------------------------ #
# FieldResult with confidence
# ------------------------------------------------------------------ #
class TestFieldResultWithConfidence:
    def test_confidence_default_none(self):
        fr = FieldResult(value="hello", citations=None)
        assert fr.confidence is None

    def test_confidence_set(self):
        fr = FieldResult(value="hello", citations=None, confidence=4)
        assert fr.confidence == 4

    def test_repr_with_confidence(self):
        fr = FieldResult(value="hello", citations=[], confidence=4)
        r = repr(fr)
        assert "confidence=4" in r

    def test_repr_without_confidence(self):
        fr = FieldResult(value="hello", citations=[])
        r = repr(fr)
        assert "confidence" not in r


# ------------------------------------------------------------------ #
# StructuredDataResult.confidences
# ------------------------------------------------------------------ #
class TestStructuredDataResultConfidences:
    def _make_result(self, with_confidences=False):
        data = SimpleSchema(name="Alice", age="30")
        from natural_pdf.elements.element_collection import ElementCollection

        confidences = None
        if with_confidences:
            confidences = {"name": 4, "age": 3}
        return StructuredDataResult(
            data=data,
            success=True,
            confidences=confidences,
        )

    def test_confidences_property_without(self):
        result = self._make_result(with_confidences=False)
        confs = result.confidences
        assert confs == {"name": None, "age": None}

    def test_confidences_property_with(self):
        result = self._make_result(with_confidences=True)
        confs = result.confidences
        assert confs == {"name": 4, "age": 3}

    def test_field_result_confidence_access(self):
        result = self._make_result(with_confidences=True)
        assert result["name"].confidence == 4
        assert result["age"].confidence == 3


# ------------------------------------------------------------------ #
# to_dict() with confidence
# ------------------------------------------------------------------ #
class TestToDictWithConfidence:
    def test_to_dict_includes_confidence(self):
        data = SimpleSchema(name="Alice", age="30")
        result = StructuredDataResult(data=data, success=True, confidences={"name": 5, "age": 3})
        d = result.to_dict()
        assert d == {"name": "Alice", "name_confidence": 5, "age": "30", "age_confidence": 3}

    def test_to_dict_confidence_false_omits(self):
        data = SimpleSchema(name="Alice", age="30")
        result = StructuredDataResult(data=data, success=True, confidences={"name": 5, "age": 3})
        d = result.to_dict(confidence=False)
        assert d == {"name": "Alice", "age": "30"}

    def test_to_dict_no_confidence_no_extra_keys(self):
        data = SimpleSchema(name="Alice", age="30")
        result = StructuredDataResult(data=data, success=True)
        d = result.to_dict()
        assert d == {"name": "Alice", "age": "30"}

    def test_to_dict_includes_sources_when_requested(self):
        data = SimpleSchema(name="Alice", age="30")
        result = StructuredDataResult(
            data=data,
            success=True,
            sources={"name": ["Alice"], "age": ["Age: 30"]},
        )
        d = result.to_dict(citations=True)
        assert d["name_sources"] == ["Alice"]
        assert d["age_sources"] == ["Age: 30"]

    def test_to_dict_omits_sources_by_default(self):
        data = SimpleSchema(name="Alice", age="30")
        result = StructuredDataResult(
            data=data,
            success=True,
            sources={"name": ["Alice"], "age": ["Age: 30"]},
        )
        d = result.to_dict()
        assert "name_sources" not in d
        assert "age_sources" not in d

    def test_to_dict_all_features(self):
        data = SimpleSchema(name="Alice", age="30")
        result = StructuredDataResult(
            data=data,
            success=True,
            confidences={"name": 5, "age": 3},
            sources={"name": ["Alice"], "age": None},
        )
        d = result.to_dict(citations=True)
        assert d["name"] == "Alice"
        assert d["name_confidence"] == 5
        assert d["name_sources"] == ["Alice"]
        assert d["age"] == "30"
        assert d["age_confidence"] == 3
        assert "age_sources" not in d  # None sources are omitted

    def test_sources_property(self):
        data = SimpleSchema(name="Alice", age="30")
        result = StructuredDataResult(
            data=data,
            success=True,
            sources={"name": ["Alice"], "age": None},
        )
        assert result.sources == {"name": ["Alice"], "age": None}

    def test_sources_property_without(self):
        data = SimpleSchema(name="Alice", age="30")
        result = StructuredDataResult(data=data, success=True)
        assert result.sources == {"name": None, "age": None}


# ------------------------------------------------------------------ #
# End-to-end mock tests
# ------------------------------------------------------------------ #
class TestExtractWithConfidenceMock:
    @staticmethod
    def _mock_completion(model_dump_data):
        """Create a mock completion whose .choices[0].message.parsed.model_dump() returns *data*."""
        parsed = MagicMock()
        parsed.model_dump.return_value = model_dump_data
        completion = MagicMock()
        completion.choices = [MagicMock()]
        completion.choices[0].message.parsed = parsed
        return completion

    def _make_mock_client(self, schema_class, response_data):
        """Single-pass mock (no confidence)."""
        mock_client = MagicMock()
        parsed_instance = schema_class(**response_data)
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.parsed = parsed_instance
        mock_client.beta.chat.completions.parse.return_value = mock_completion
        return mock_client

    def test_confidence_single_pass_default(self):
        """confidence=True with default single-pass: 1 LLM call with extended schema."""
        from natural_pdf import PDF

        pdf = PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]

        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.return_value = self._mock_completion(
            {"name": "Jungle Health", "name_confidence": 5, "age": "1905", "age_confidence": 3}
        )

        result = page.extract(SimpleSchema, client=mock_client, confidence=True)

        assert isinstance(result, StructuredDataResult)
        assert result.success
        assert result.name == "Jungle Health"
        assert result["name"].confidence == 5
        assert result["age"].confidence == 3
        assert result.confidences == {"name": 5, "age": 3}
        # Single pass: only 1 LLM call
        assert mock_client.beta.chat.completions.parse.call_count == 1
        pdf.close()

    def test_confidence_multipass_two_calls(self):
        """confidence=True with multipass=True: 2 LLM calls."""
        from natural_pdf import PDF

        pdf = PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]

        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.side_effect = [
            self._mock_completion({"name": "Jungle Health", "age": "1905"}),
            self._mock_completion({"name_confidence": 5, "age_confidence": 3}),
        ]

        result = page.extract(SimpleSchema, client=mock_client, confidence=True, multipass=True)

        assert result.success
        assert result["name"].confidence == 5
        assert result["age"].confidence == 3
        assert mock_client.beta.chat.completions.parse.call_count == 2
        pdf.close()

    def test_no_confidence_single_pass(self):
        """Without confidence, only one LLM call is made."""
        from natural_pdf import PDF

        pdf = PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]

        mock_client = self._make_mock_client(
            SimpleSchema,
            {"name": "Jungle Health", "age": "1905"},
        )

        result = page.extract(SimpleSchema, client=mock_client)

        assert result.success
        assert result["name"].confidence is None
        # Only one call
        assert mock_client.beta.chat.completions.parse.call_count == 1
        pdf.close()

    def test_categorical_confidence(self):
        from natural_pdf import PDF

        pdf = PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]

        levels = ["low", "medium", "high"]
        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.return_value = self._mock_completion(
            {
                "name": "Jungle Health",
                "name_confidence": "high",
                "age": "1905",
                "age_confidence": "medium",
            }
        )

        result = page.extract(SimpleSchema, client=mock_client, confidence=levels)

        assert result.success
        assert result["name"].confidence == "high"
        assert result["age"].confidence == "medium"
        pdf.close()

    def test_confidence_and_citations_single_pass(self):
        """citations=True + confidence=True default: 1 call with combined schema."""
        from natural_pdf import PDF
        from natural_pdf.extraction.citations import add_line_numbers

        pdf = PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]

        # Find a line with actual content for reliable resolution
        text = page.extract_text(layout=True)
        _, line_map = add_line_numbers(text)
        content_line = next((i for i, t in line_map.items() if "Jungle" in t), 0)

        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.return_value = self._mock_completion(
            {
                "name_source_lines": [content_line],
                "name": "Jungle Health and Safety Inspection Service",
                "name_confidence": 5,
                "age_source_lines": [content_line + 2],
                "age": "1905",
                "age_confidence": 3,
            }
        )

        result = page.extract(SimpleSchema, client=mock_client, citations=True, confidence=True)

        assert result.success
        assert result["name"].confidence == 5
        assert result["name"].sources is not None
        d = result.to_dict(citations=True)
        assert "name_sources" in d
        assert mock_client.beta.chat.completions.parse.call_count == 1
        pdf.close()

    def test_confidence_and_citations_multipass(self):
        """citations=True + confidence=True + multipass=True: 2 calls."""
        from natural_pdf import PDF

        pdf = PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]

        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.side_effect = [
            # Pass 1: extraction
            self._mock_completion({"name": "Jungle Health", "age": "1905"}),
            # Pass 2: combined meta
            self._mock_completion(
                {
                    "name_source_lines": [0],
                    "name_confidence": 5,
                    "age_source_lines": [2],
                    "age_confidence": 3,
                }
            ),
        ]

        result = page.extract(
            SimpleSchema, client=mock_client, citations=True, confidence=True, multipass=True
        )

        assert result.success
        assert result["name"].confidence == 5
        assert mock_client.beta.chat.completions.parse.call_count == 2
        pdf.close()

    def test_confidence_with_vision(self):
        from natural_pdf import PDF

        pdf = PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]

        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.return_value = self._mock_completion(
            {"name": "Jungle Health", "name_confidence": 4, "age": "1905", "age_confidence": 3}
        )

        result = page.extract(SimpleSchema, client=mock_client, using="vision", confidence=True)

        assert result.success
        assert result["name"].confidence == 4
        pdf.close()

    def test_multipass_meta_failure_graceful_degradation(self):
        """If pass 2 (meta) fails in multipass mode, extraction result is still returned."""
        from natural_pdf import PDF

        pdf = PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]

        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.side_effect = [
            self._mock_completion({"name": "Jungle Health", "age": "1905"}),
            Exception("LLM error on meta pass"),
        ]

        result = page.extract(SimpleSchema, client=mock_client, confidence=True, multipass=True)

        assert result.success
        assert result.name == "Jungle Health"
        assert result["name"].confidence is None
        assert result["age"].confidence is None
        pdf.close()

    def test_instructions_parameter(self):
        from natural_pdf import PDF

        pdf = PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]

        mock_client = self._make_mock_client(
            SimpleSchema,
            {"name": "Jungle Health", "age": "1905"},
        )

        result = page.extract(
            SimpleSchema,
            client=mock_client,
            instructions="Focus on header text only.",
        )

        assert result.success
        # Verify the instructions were in the prompt
        call_args = mock_client.beta.chat.completions.parse.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        system_msg = messages[0]["content"]
        assert "Focus on header text only." in system_msg
        pdf.close()

    def test_backward_compat_no_confidence(self):
        from natural_pdf import PDF

        pdf = PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]

        mock_client = self._make_mock_client(
            SimpleSchema,
            {"name": "Jungle Health", "age": "1905"},
        )

        result = page.extract(SimpleSchema, client=mock_client)

        assert result.success
        assert result["name"].confidence is None
        assert result.confidences == {"name": None, "age": None}
        pdf.close()

    def test_null_field_null_confidence(self):
        from natural_pdf import PDF

        pdf = PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]

        # Single-pass with confidence
        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.return_value = self._mock_completion(
            {
                "invoice_number": "INV-001",
                "invoice_number_confidence": 5,
                "date": None,
                "date_confidence": 2,  # LLM returned a score but field is null
                "total": "$100",
                "total_confidence": 4,
            }
        )

        result = page.extract(InvoiceSchema, client=mock_client, confidence=True)

        assert result.success
        assert result.date is None
        # Null field should have null confidence
        assert result["date"].confidence is None
        assert result["invoice_number"].confidence == 5
        pdf.close()


# ------------------------------------------------------------------ #
# Clamping tests
# ------------------------------------------------------------------ #
class TestClampConfidences:
    def test_clamp_above_max(self, caplog):
        from natural_pdf.services.extraction_service import ExtractionService

        cfg = ConfidenceConfig(
            scale={0.0: None, 1.0: None},
            is_numeric=True,
            min_value=0.0,
            max_value=1.0,
        )
        confidences = {"name": 1.5, "age": 0.8}

        with caplog.at_level(logging.WARNING):
            result = ExtractionService._clamp_confidences(confidences, cfg)

        assert result["name"] == 1.0
        assert result["age"] == 0.8
        assert "clamping" in caplog.text.lower()

    def test_clamp_below_min(self, caplog):
        from natural_pdf.services.extraction_service import ExtractionService

        cfg = ConfidenceConfig(
            scale={0.0: None, 1.0: None},
            is_numeric=True,
            min_value=0.0,
            max_value=1.0,
        )
        confidences = {"name": -0.1, "age": 0.5}

        with caplog.at_level(logging.WARNING):
            result = ExtractionService._clamp_confidences(confidences, cfg)

        assert result["name"] == 0.0
        assert result["age"] == 0.5

    def test_no_clamp_for_categorical(self):
        from natural_pdf.services.extraction_service import ExtractionService

        cfg = ConfidenceConfig(
            scale={"low": None, "high": None},
            is_numeric=False,
        )
        confidences = {"name": "high", "age": "low"}
        result = ExtractionService._clamp_confidences(confidences, cfg)
        assert result == confidences

    def test_null_confidence_preserved(self):
        from natural_pdf.services.extraction_service import ExtractionService

        cfg = ConfidenceConfig(
            scale={0.0: None, 1.0: None},
            is_numeric=True,
            min_value=0.0,
            max_value=1.0,
        )
        confidences = {"name": None, "age": 0.5}
        result = ExtractionService._clamp_confidences(confidences, cfg)
        assert result["name"] is None
        assert result["age"] == 0.5

    def test_clamp_end_to_end_mock(self, caplog):
        """End-to-end: LLM returns out-of-range confidence, verify clamped."""
        from natural_pdf import PDF

        pdf = PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]

        mock_client = MagicMock()
        # Single-pass with confidence
        mock_client.beta.chat.completions.parse.return_value = (
            TestExtractWithConfidenceMock._mock_completion(
                {"name": "Test", "name_confidence": 7, "age": "25", "age_confidence": 4}
            )
        )

        with caplog.at_level(logging.WARNING):
            result = page.extract(SimpleSchema, client=mock_client, confidence=True)

        assert result.success
        assert result["name"].confidence == 5  # clamped from 7 to max=5
        assert result["age"].confidence == 4  # unchanged
        assert "clamping" in caplog.text.lower()
        pdf.close()

    def test_custom_range_clamp(self, caplog):
        """Custom range 1-5: verify clamping works with non-default bounds."""
        from natural_pdf import PDF
        from natural_pdf.services.extraction_service import ExtractionService

        cfg = normalize_confidence_config(
            {
                1.0: "Low",
                3.0: "Medium",
                5.0: "High",
            }
        )
        confidences = {"name": 6.0, "age": 0.5}

        with caplog.at_level(logging.WARNING):
            result = ExtractionService._clamp_confidences(confidences, cfg)

        assert result["name"] == 5.0  # clamped to max
        assert result["age"] == 1.0  # clamped to min


# ------------------------------------------------------------------ #
# Multi-line legend support
# ------------------------------------------------------------------ #
class TestMultiLineLegend:
    def test_single_line_labels(self):
        from natural_pdf.utils.visualization import create_legend

        labels_colors = {"alpha": (255, 0, 0, 100), "beta": (0, 255, 0, 100)}
        img = create_legend(labels_colors)
        assert img.width == 250
        assert img.height > 0

    def test_multiline_labels_taller(self):
        from natural_pdf.utils.visualization import create_legend

        single = {"alpha": (255, 0, 0, 100)}
        multi = {"alpha:\n  some value text": (255, 0, 0, 100)}
        img_single = create_legend(single)
        img_multi = create_legend(multi)
        # Multi-line label should produce a taller image
        assert img_multi.height > img_single.height

    def test_empty_labels(self):
        from natural_pdf.utils.visualization import create_legend

        img = create_legend({})
        assert img.width == 250
        assert img.height > 0  # padding only


# ------------------------------------------------------------------ #
# Enriched legend labels in StructuredDataResult.show()
# ------------------------------------------------------------------ #
class TestEnrichedLegendLabels:
    def test_show_passes_enriched_labels(self):
        """Verify that show() builds labels containing field name + value."""
        from natural_pdf.elements.element_collection import ElementCollection

        # Create a mock element with a page that has a .show() method
        mock_page = MagicMock()
        mock_page.show.return_value = MagicMock()  # fake image

        mock_elem = MagicMock()
        mock_elem.page = mock_page

        ec = ElementCollection([mock_elem])

        data = SimpleSchema(name="Alice", age="30")
        result = StructuredDataResult(
            data=data,
            success=True,
            citations={"name": ec, "age": ElementCollection([])},
        )

        # Capture what label is passed to highlight()
        highlight_labels = []
        original_highlight = ec.highlight

        def capture_highlight(**kwargs):
            highlight_labels.append(kwargs.get("label"))

        ec.highlight = capture_highlight

        result.show()

        assert len(highlight_labels) == 1
        assert "name:" in highlight_labels[0]
        assert "Alice" in highlight_labels[0]

    def test_show_wraps_and_truncates_long_values(self):
        """Long values should wrap at ~30 chars and truncate to 3 value lines."""
        from natural_pdf.elements.element_collection import ElementCollection

        mock_page = MagicMock()
        mock_page.show.return_value = MagicMock()
        mock_elem = MagicMock()
        mock_elem.page = mock_page

        ec = ElementCollection([mock_elem])

        # A value long enough to exceed 3 wrapped lines at width=30
        long_value = "word " * 40  # 200 chars, ~6-7 lines at width 30
        data = SimpleSchema(name=long_value.strip(), age="30")
        result = StructuredDataResult(
            data=data,
            success=True,
            citations={"name": ec, "age": ElementCollection([])},
        )

        highlight_labels = []
        ec.highlight = lambda **kw: highlight_labels.append(kw.get("label"))
        result.show()

        label = highlight_labels[0]
        lines = label.split("\n")
        # First line is the field name, then up to 4 value lines
        assert lines[0] == "name:"
        value_lines = lines[1:]
        assert len(value_lines) <= 4
        assert "..." in value_lines[-1]

    def test_show_short_value_no_truncation(self):
        """Short values should not be truncated."""
        from natural_pdf.elements.element_collection import ElementCollection

        mock_page = MagicMock()
        mock_page.show.return_value = MagicMock()
        mock_elem = MagicMock()
        mock_elem.page = mock_page

        ec = ElementCollection([mock_elem])

        data = SimpleSchema(name="Alice", age="30")
        result = StructuredDataResult(
            data=data,
            success=True,
            citations={"name": ec, "age": ElementCollection([])},
        )

        highlight_labels = []
        ec.highlight = lambda **kw: highlight_labels.append(kw.get("label"))
        result.show()

        label = highlight_labels[0]
        assert "..." not in label
        assert "Alice" in label


# ------------------------------------------------------------------ #
# Annotated PDF export
# ------------------------------------------------------------------ #
class TestAnnotatedPdfExport:
    def test_create_annotated_pdf(self, tmp_path):
        """End-to-end: create an annotated PDF from mock extraction results."""
        pytest.importorskip("pikepdf")
        from natural_pdf import PDF

        pdf = PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]

        # Find some real elements to use as citations
        elements = page.find_all("text")[:3]
        if len(elements) == 0:
            pytest.skip("No text elements found on test page")

        from natural_pdf.elements.element_collection import ElementCollection

        citations_ec = ElementCollection(list(elements))

        fr = FieldResult(value="Test Value", citations=citations_ec, confidence=0.95)
        fields = {"test_field": fr}

        output = tmp_path / "annotated.pdf"

        from natural_pdf.exporters.annotated_pdf import create_annotated_pdf

        create_annotated_pdf(fields, str(output))

        assert output.exists()
        assert output.stat().st_size > 0

        # Verify annotations exist via pikepdf
        import pikepdf

        with pikepdf.Pdf.open(str(output)) as check_pdf:
            page0 = check_pdf.pages[0]
            assert "/Annots" in page0
            annots = page0.Annots
            assert len(annots) > 0
            # Check highlight annotations
            highlights = [a for a in annots if str(a.Subtype) == "/Highlight"]
            assert len(highlights) > 0
            assert "test_field" in str(highlights[0].T)
            # Verify sidebar legend widened the page
            orig_pdf = pikepdf.Pdf.open("pdfs/01-practice.pdf")
            orig_width = float(orig_pdf.pages[0].MediaBox[2])
            new_width = float(page0.MediaBox[2])
            assert new_width > orig_width  # page was extended
            orig_pdf.close()
            # Verify Helvetica font registered for legend text
            assert "/HelvLegend" in page0.Resources.Font

        pdf.close()

    def test_save_pdf_method(self, tmp_path):
        """Verify StructuredDataResult.save_pdf() delegates correctly."""
        pytest.importorskip("pikepdf")
        from natural_pdf import PDF

        pdf = PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]

        elements = page.find_all("text")[:2]
        if len(elements) == 0:
            pytest.skip("No text elements found on test page")

        from natural_pdf.elements.element_collection import ElementCollection

        data = SimpleSchema(name="Test", age="25")
        result = StructuredDataResult(
            data=data,
            success=True,
            citations={"name": ElementCollection(list(elements)), "age": ElementCollection([])},
        )

        output = tmp_path / "result_annotated.pdf"
        result.save_pdf(str(output))

        assert output.exists()
        assert output.stat().st_size > 0
        pdf.close()

    def test_save_pdf_no_citations_raises(self, tmp_path):
        """save_pdf with no citation elements should raise ValueError."""
        pytest.importorskip("pikepdf")
        data = SimpleSchema(name="Test", age="25")
        result = StructuredDataResult(data=data, success=True)

        output = tmp_path / "empty.pdf"
        with pytest.raises(ValueError, match="No citation elements"):
            result.save_pdf(str(output))

    def test_annotated_pdf_popup_text(self, tmp_path):
        """Verify popup text includes field name, value, and confidence."""
        pytest.importorskip("pikepdf")
        from natural_pdf import PDF

        pdf = PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]

        elements = page.find_all("text")[:1]
        if len(elements) == 0:
            pytest.skip("No text elements found")

        from natural_pdf.elements.element_collection import ElementCollection

        fr = FieldResult(
            value="INV-001", citations=ElementCollection(list(elements)), confidence=0.95
        )
        fields = {"invoice_number": fr}

        output = tmp_path / "popup_test.pdf"

        from natural_pdf.exporters.annotated_pdf import create_annotated_pdf

        create_annotated_pdf(fields, str(output))

        import pikepdf

        with pikepdf.Pdf.open(str(output)) as check_pdf:
            annot = check_pdf.pages[0].Annots[0]
            contents = str(annot.Contents)
            assert "invoice_number" in contents
            assert "INV-001" in contents
            assert "confidence" in contents
            assert "0.95" in contents

        pdf.close()
