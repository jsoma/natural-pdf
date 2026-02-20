"""Tests for per-field confidence scoring in structured extraction."""

import logging
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from natural_pdf.extraction.citations import (
    DEFAULT_CONFIDENCE_SCALE,
    ConfidenceConfig,
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
        assert cfg.min_value == 0.0
        assert cfg.max_value == 1.0
        assert cfg.scale == DEFAULT_CONFIDENCE_SCALE

    def test_range_string_returns_default_numeric(self):
        cfg = normalize_confidence_config("range")
        assert cfg is not None
        assert cfg.is_numeric is True
        assert cfg.min_value == 0.0
        assert cfg.max_value == 1.0

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
# build_extended_schema
# ------------------------------------------------------------------ #
class TestBuildExtendedSchema:
    def test_sources_only(self):
        Extended = build_extended_schema(SimpleSchema, with_sources=True)
        fields = set(Extended.model_fields.keys())
        assert "name" in fields
        assert "name_source" in fields
        assert "age_source" in fields
        assert "name_confidence" not in fields

    def test_confidence_only_numeric(self):
        cfg = normalize_confidence_config(True)
        Extended = build_extended_schema(SimpleSchema, with_confidence=True, confidence_config=cfg)
        fields = set(Extended.model_fields.keys())
        assert "name" in fields
        assert "name_confidence" in fields
        assert "age_confidence" in fields
        assert "name_source" not in fields

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
        assert "name_source" in fields
        assert "name_confidence" in fields
        assert "age_source" in fields
        assert "age_confidence" in fields

    def test_categorical_confidence(self):
        cfg = normalize_confidence_config(["low", "medium", "high"])
        Extended = build_extended_schema(SimpleSchema, with_confidence=True, confidence_config=cfg)
        fields = set(Extended.model_fields.keys())
        assert "name_confidence" in fields

    def test_schema_instantiation_numeric(self):
        cfg = normalize_confidence_config(True)
        Extended = build_extended_schema(SimpleSchema, with_confidence=True, confidence_config=cfg)
        instance = Extended(name="Alice", age="30", name_confidence=0.9, age_confidence=0.8)
        assert instance.name == "Alice"
        assert instance.name_confidence == 0.9

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
        assert "_source" in prompt
        assert "verbatim" in prompt.lower()

    def test_confidence_block_numeric(self):
        cfg = normalize_confidence_config(True)
        prompt = build_extended_prompt(
            None, SimpleSchema, with_confidence=True, confidence_config=cfg
        )
        assert "_confidence" in prompt
        assert "0.0" in prompt
        assert "1.0" in prompt

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
        assert "Explicitly stated" in prompt


# ------------------------------------------------------------------ #
# split_extended_result
# ------------------------------------------------------------------ #
class TestSplitExtendedResult:
    def test_confidence_only(self):
        data = {
            "name": "Alice",
            "age": "30",
            "name_confidence": 0.9,
            "age_confidence": 0.7,
        }
        user_data, sources, confidences = split_extended_result(
            data, SimpleSchema, with_confidence=True
        )
        assert user_data == {"name": "Alice", "age": "30"}
        assert sources is None
        assert confidences == {"name": 0.9, "age": 0.7}

    def test_sources_only(self):
        data = {
            "name": "Alice",
            "age": "30",
            "name_source": ["L0: Alice"],
            "age_source": ["L1: 30"],
        }
        user_data, sources, confidences = split_extended_result(
            data, SimpleSchema, with_sources=True
        )
        assert user_data == {"name": "Alice", "age": "30"}
        assert sources == {"name": ["L0: Alice"], "age": ["L1: 30"]}
        assert confidences is None

    def test_both_sources_and_confidence(self):
        data = {
            "name": "Alice",
            "age": "30",
            "name_source": ["L0: Alice"],
            "age_source": None,
            "name_confidence": 0.9,
            "age_confidence": 0.5,
        }
        user_data, sources, confidences = split_extended_result(
            data, SimpleSchema, with_sources=True, with_confidence=True
        )
        assert user_data == {"name": "Alice", "age": "30"}
        assert sources["name"] == ["L0: Alice"]
        assert sources["age"] is None
        assert confidences["name"] == 0.9
        assert confidences["age"] == 0.5

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
        fr = FieldResult(value="hello", citations=None, confidence=0.95)
        assert fr.confidence == 0.95

    def test_repr_with_confidence(self):
        fr = FieldResult(value="hello", citations=[], confidence=0.9)
        r = repr(fr)
        assert "confidence=0.9" in r

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
            confidences = {"name": 0.9, "age": 0.7}
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
        assert confs == {"name": 0.9, "age": 0.7}

    def test_field_result_confidence_access(self):
        result = self._make_result(with_confidences=True)
        assert result["name"].confidence == 0.9
        assert result["age"].confidence == 0.7


# ------------------------------------------------------------------ #
# End-to-end mock tests
# ------------------------------------------------------------------ #
class TestExtractWithConfidenceMock:
    def _make_mock_client(self, schema_class, response_data):
        mock_client = MagicMock()
        parsed_instance = schema_class(**response_data)
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.parsed = parsed_instance
        mock_client.beta.chat.completions.parse.return_value = mock_completion
        return mock_client

    def test_confidence_true(self):
        from natural_pdf import PDF

        pdf = PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]

        cfg = normalize_confidence_config(True)
        Extended = build_extended_schema(SimpleSchema, with_confidence=True, confidence_config=cfg)
        mock_client = self._make_mock_client(
            Extended,
            {
                "name": "Jungle Health",
                "age": "1905",
                "name_confidence": 0.95,
                "age_confidence": 0.6,
            },
        )

        result = page.extract(SimpleSchema, client=mock_client, confidence=True)

        assert isinstance(result, StructuredDataResult)
        assert result.success
        assert result.name == "Jungle Health"
        assert result["name"].confidence == 0.95
        assert result["age"].confidence == 0.6
        assert result.confidences == {"name": 0.95, "age": 0.6}
        pdf.close()

    def test_categorical_confidence(self):
        from natural_pdf import PDF

        pdf = PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]

        levels = ["low", "medium", "high"]
        cfg = normalize_confidence_config(levels)
        Extended = build_extended_schema(SimpleSchema, with_confidence=True, confidence_config=cfg)
        mock_client = self._make_mock_client(
            Extended,
            {
                "name": "Jungle Health",
                "age": "1905",
                "name_confidence": "high",
                "age_confidence": "medium",
            },
        )

        result = page.extract(SimpleSchema, client=mock_client, confidence=levels)

        assert result.success
        assert result["name"].confidence == "high"
        assert result["age"].confidence == "medium"
        pdf.close()

    def test_confidence_and_citations(self):
        from natural_pdf import PDF
        from natural_pdf.extraction.citations import build_shadow_schema

        pdf = PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]

        cfg = normalize_confidence_config(True)
        Extended = build_extended_schema(
            SimpleSchema,
            with_sources=True,
            with_confidence=True,
            confidence_config=cfg,
        )
        mock_client = self._make_mock_client(
            Extended,
            {
                "name": "Jungle Health and Safety Inspection Service",
                "age": "1905",
                "name_source": ["L0: Jungle Health and Safety Inspection Service"],
                "age_source": ["L2: Date: February 3, 1905"],
                "name_confidence": 0.95,
                "age_confidence": 0.6,
            },
        )

        result = page.extract(SimpleSchema, client=mock_client, citations=True, confidence=True)

        assert result.success
        assert result["name"].confidence == 0.95
        assert result["name"].citations is not None
        assert len(result["name"].citations) > 0
        pdf.close()

    def test_confidence_with_vision(self):
        from natural_pdf import PDF

        pdf = PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]

        cfg = normalize_confidence_config(True)
        Extended = build_extended_schema(SimpleSchema, with_confidence=True, confidence_config=cfg)
        mock_client = self._make_mock_client(
            Extended,
            {
                "name": "Jungle Health",
                "age": "1905",
                "name_confidence": 0.9,
                "age_confidence": 0.7,
            },
        )

        result = page.extract(SimpleSchema, client=mock_client, using="vision", confidence=True)

        assert result.success
        assert result["name"].confidence == 0.9
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

        cfg = normalize_confidence_config(True)
        Extended = build_extended_schema(InvoiceSchema, with_confidence=True, confidence_config=cfg)
        mock_client = self._make_mock_client(
            Extended,
            {
                "invoice_number": "INV-001",
                "date": None,
                "total": "$100",
                "invoice_number_confidence": 0.95,
                "date_confidence": 0.3,  # LLM returned a score but field is null
                "total_confidence": 0.9,
            },
        )

        result = page.extract(InvoiceSchema, client=mock_client, confidence=True)

        assert result.success
        assert result.date is None
        # Null field should have null confidence
        assert result["date"].confidence is None
        assert result["invoice_number"].confidence == 0.95
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

        cfg = normalize_confidence_config(True)
        Extended = build_extended_schema(SimpleSchema, with_confidence=True, confidence_config=cfg)

        mock_client = MagicMock()
        # Build a mock parsed object that bypasses Pydantic validation
        # (simulating an LLM that returns out-of-range values)
        parsed = MagicMock()
        parsed.model_dump.return_value = {
            "name": "Test",
            "age": "25",
            "name_confidence": 1.5,
            "age_confidence": 0.8,
        }
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.parsed = parsed
        mock_client.beta.chat.completions.parse.return_value = mock_completion

        with caplog.at_level(logging.WARNING):
            result = page.extract(SimpleSchema, client=mock_client, confidence=True)

        assert result.success
        assert result["name"].confidence == 1.0  # clamped from 1.5
        assert result["age"].confidence == 0.8  # unchanged
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
