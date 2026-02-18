"""Tests for citation/grounding support in structured extraction."""

import re
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from natural_pdf.extraction.citations import (
    PageTextMapInfo,
    add_line_numbers,
    build_char_to_element_map,
    build_citation_prompt,
    build_shadow_schema,
    resolve_citations,
    split_shadow_result,
)
from natural_pdf.extraction.result import FieldResult, StructuredDataResult


# ------------------------------------------------------------------ #
# Test schemas
# ------------------------------------------------------------------ #
class InvoiceSchema(BaseModel):
    invoice_number: str = Field(description="The invoice number")
    date: Optional[str] = None
    total: str = Field(description="Total amount")


class SimpleSchema(BaseModel):
    name: str
    age: str


class ListSchema(BaseModel):
    items: List[str] = Field(default_factory=list, description="Line items")
    summary: str = Field(description="Summary text")


# ------------------------------------------------------------------ #
# add_line_numbers
# ------------------------------------------------------------------ #
class TestAddLineNumbers:
    def test_basic(self):
        text = "Hello\nWorld\nFoo"
        numbered, line_map = add_line_numbers(text)
        assert "L0: Hello" in numbered
        assert "L1: World" in numbered
        assert "L2: Foo" in numbered
        assert line_map[0] == "Hello"
        assert line_map[1] == "World"
        assert line_map[2] == "Foo"

    def test_single_line(self):
        numbered, line_map = add_line_numbers("Only one line")
        assert numbered == "L0: Only one line"
        assert line_map == {0: "Only one line"}

    def test_dynamic_padding(self):
        lines = [f"Line {i}" for i in range(105)]
        text = "\n".join(lines)
        numbered, line_map = add_line_numbers(text)
        assert "L000: Line 0" in numbered
        assert "L009: Line 9" in numbered
        assert "L099: Line 99" in numbered
        assert "L104: Line 104" in numbered
        assert len(line_map) == 105

    def test_empty_lines_preserved(self):
        text = "Line A\n\nLine C"
        numbered, line_map = add_line_numbers(text)
        lines = numbered.split("\n")
        assert len(lines) == 3
        assert line_map[1] == ""

    def test_empty_string(self):
        numbered, line_map = add_line_numbers("")
        assert numbered == "L0: "
        assert line_map == {0: ""}


# ------------------------------------------------------------------ #
# build_shadow_schema
# ------------------------------------------------------------------ #
class TestBuildShadowSchema:
    def test_flat_schema(self):
        Shadow = build_shadow_schema(SimpleSchema)
        fields = set(Shadow.model_fields.keys())
        assert "name" in fields
        assert "name_source" in fields
        assert "age" in fields
        assert "age_source" in fields

    def test_optional_fields(self):
        Shadow = build_shadow_schema(InvoiceSchema)
        fields = set(Shadow.model_fields.keys())
        assert "invoice_number" in fields
        assert "invoice_number_source" in fields
        assert "date" in fields
        assert "date_source" in fields

    def test_list_fields(self):
        Shadow = build_shadow_schema(ListSchema)
        fields = set(Shadow.model_fields.keys())
        assert "items" in fields
        assert "items_source" in fields

    def test_source_fields_are_optional_list_str(self):
        Shadow = build_shadow_schema(SimpleSchema)
        instance = Shadow(name="test", age="25")
        assert instance.name_source is None

    def test_shadow_schema_instantiation(self):
        Shadow = build_shadow_schema(SimpleSchema)
        instance = Shadow(
            name="Alice",
            name_source=["L01: Name: Alice"],
            age="30",
            age_source=["L02: Age: 30"],
        )
        assert instance.name == "Alice"
        assert instance.name_source == ["L01: Name: Alice"]


# ------------------------------------------------------------------ #
# build_citation_prompt
# ------------------------------------------------------------------ #
class TestBuildCitationPrompt:
    def test_default_prompt(self):
        prompt = build_citation_prompt(None, SimpleSchema)
        assert "SimpleSchema" in prompt
        assert "_source" in prompt
        assert "verbatim" in prompt.lower()

    def test_custom_prompt_preserved(self):
        custom = "Extract invoice details carefully."
        prompt = build_citation_prompt(custom, InvoiceSchema)
        assert prompt.startswith(custom)
        assert "_source" in prompt

    def test_contains_example(self):
        prompt = build_citation_prompt(None, SimpleSchema)
        assert "L03:" in prompt or "Lnn:" in prompt or "line prefix" in prompt.lower()


# ------------------------------------------------------------------ #
# build_char_to_element_map
# ------------------------------------------------------------------ #
class TestBuildCharToElementMap:
    def test_basic_mapping(self):
        cd1 = {"text": "H", "x0": 0}
        cd2 = {"text": "i", "x0": 5}
        cd3 = {"text": "!", "x0": 10}

        word1 = MagicMock()
        word1._char_dicts = [cd1, cd2]
        word2 = MagicMock()
        word2._char_dicts = [cd3]

        mapping = build_char_to_element_map([word1, word2])
        assert mapping[id(cd1)] is word1
        assert mapping[id(cd2)] is word1
        assert mapping[id(cd3)] is word2

    def test_empty_words(self):
        assert build_char_to_element_map([]) == {}

    def test_words_without_char_dicts(self):
        word = MagicMock(spec=[])
        assert build_char_to_element_map([word]) == {}


# ------------------------------------------------------------------ #
# split_shadow_result
# ------------------------------------------------------------------ #
class TestSplitShadowResult:
    def test_only_user_fields(self):
        Shadow = build_shadow_schema(SimpleSchema)
        shadow_instance = Shadow(
            name="Alice",
            name_source=["L01: Alice"],
            age="30",
            age_source=["L02: Age 30"],
        )
        user_data = split_shadow_result(shadow_instance, SimpleSchema)
        assert isinstance(user_data, SimpleSchema)
        assert user_data.name == "Alice"
        assert user_data.age == "30"
        assert not hasattr(user_data, "name_source")

    def test_preserves_optional_none(self):
        Shadow = build_shadow_schema(InvoiceSchema)
        shadow_instance = Shadow(
            invoice_number="INV-001",
            invoice_number_source=["L01: INV-001"],
            date=None,
            date_source=None,
            total="$100",
            total_source=["L05: Total: $100"],
        )
        user_data = split_shadow_result(shadow_instance, InvoiceSchema)
        assert user_data.invoice_number == "INV-001"
        assert user_data.date is None
        assert user_data.total == "$100"


# ------------------------------------------------------------------ #
# resolve_citations (with real TextMap)
# ------------------------------------------------------------------ #
class TestResolveCitations:
    def _make_textmap_and_elements(self):
        from pdfplumber.utils.text import chars_to_textmap

        line1 = "Invoice #12345"
        line2 = "Date: 2024-01-01"
        char_dicts = []
        words = []

        x = 10
        word_chars = []
        for ch in line1:
            cd = {
                "text": ch,
                "x0": x,
                "x1": x + 7,
                "top": 10,
                "bottom": 20,
                "doctop": 10,
                "upright": True,
                "fontname": "Arial",
                "size": 12,
                "object_type": "char",
            }
            char_dicts.append(cd)
            word_chars.append(cd)
            x += 7

        word1 = MagicMock()
        word1._char_dicts = list(word_chars)
        word1.bbox = (10, 10, x, 20)
        words.append(word1)

        x = 10
        word_chars2 = []
        for ch in line2:
            cd = {
                "text": ch,
                "x0": x,
                "x1": x + 7,
                "top": 30,
                "bottom": 40,
                "doctop": 30,
                "upright": True,
                "fontname": "Arial",
                "size": 12,
                "object_type": "char",
            }
            char_dicts.append(cd)
            word_chars2.append(cd)
            x += 7

        word2 = MagicMock()
        word2._char_dicts = list(word_chars2)
        word2.bbox = (10, 30, x, 40)
        words.append(word2)

        textmap = chars_to_textmap(
            char_dicts,
            layout=False,
            x_tolerance=5,
            y_tolerance=5,
        )
        return textmap, words, char_dicts

    def test_exact_match(self):
        textmap, words, _ = self._make_textmap_and_elements()
        char_to_elem = build_char_to_element_map(words)
        text = "Invoice #12345\nDate: 2024-01-01"
        _, line_map = add_line_numbers(text)

        Shadow = build_shadow_schema(SimpleSchema)
        shadow_data = Shadow(
            name="12345",
            name_source=["L0: Invoice #12345"],
            age="2024-01-01",
            age_source=["L1: Date: 2024-01-01"],
        )

        citations = resolve_citations(
            shadow_data=shadow_data,
            user_schema=SimpleSchema,
            line_map=line_map,
            textmap_info=textmap,
            char_to_element_map=char_to_elem,
        )
        assert len(citations["name"]) > 0
        assert len(citations["age"]) > 0

    def test_no_source_quotes(self):
        textmap, words, _ = self._make_textmap_and_elements()
        char_to_elem = build_char_to_element_map(words)
        text = "Invoice #12345\nDate: 2024-01-01"
        _, line_map = add_line_numbers(text)

        Shadow = build_shadow_schema(SimpleSchema)
        shadow_data = Shadow(
            name="12345",
            name_source=None,
            age="2024-01-01",
            age_source=None,
        )

        citations = resolve_citations(
            shadow_data=shadow_data,
            user_schema=SimpleSchema,
            line_map=line_map,
            textmap_info=textmap,
            char_to_element_map=char_to_elem,
        )
        assert len(citations["name"]) == 0
        assert len(citations["age"]) == 0


# ------------------------------------------------------------------ #
# FieldResult
# ------------------------------------------------------------------ #
class TestFieldResult:
    def test_str(self):
        fr = FieldResult(value="hello", citations=None)
        assert str(fr) == "hello"

    def test_repr(self):
        fr = FieldResult(value="hello", citations=[])
        assert "hello" in repr(fr)


# ------------------------------------------------------------------ #
# StructuredDataResult
# ------------------------------------------------------------------ #
class TestStructuredDataResult:
    def _make_result(self, with_citations=False):
        data = SimpleSchema(name="Alice", age="30")
        citations = None
        if with_citations:
            from natural_pdf.elements.element_collection import ElementCollection

            citations = {
                "name": ElementCollection([]),
                "age": ElementCollection([]),
            }
        return StructuredDataResult(
            data=data,
            success=True,
            citations=citations,
        )

    def test_attribute_access(self):
        result = self._make_result()
        assert result.name == "Alice"
        assert result.age == "30"

    def test_attribute_error_for_unknown(self):
        result = self._make_result()
        with pytest.raises(AttributeError, match="no field"):
            result.nonexistent

    def test_item_access_returns_field_result(self):
        result = self._make_result()
        fr = result["name"]
        assert isinstance(fr, FieldResult)
        assert fr.value == "Alice"

    def test_item_key_error(self):
        result = self._make_result()
        with pytest.raises(KeyError):
            result["nonexistent"]

    def test_contains(self):
        result = self._make_result()
        assert "name" in result
        assert "nonexistent" not in result

    def test_len(self):
        result = self._make_result()
        assert len(result) == 2

    def test_iter(self):
        result = self._make_result()
        keys = list(result)
        assert "name" in keys
        assert "age" in keys

    def test_items(self):
        result = self._make_result()
        pairs = list(result.items())
        assert len(pairs) == 2
        names = [n for n, _ in pairs]
        assert "name" in names
        assert "age" in names
        for _, fr in pairs:
            assert isinstance(fr, FieldResult)

    def test_to_dict(self):
        result = self._make_result()
        d = result.to_dict()
        assert d == {"name": "Alice", "age": "30"}

    def test_data_gives_pydantic_model(self):
        result = self._make_result()
        assert isinstance(result.data, SimpleSchema)
        assert result.data.name == "Alice"

    def test_citations_property_empty_when_no_citations(self):
        result = self._make_result(with_citations=False)
        cits = result.citations
        assert "name" in cits
        assert len(cits["name"]) == 0

    def test_citations_property_with_citations(self):
        result = self._make_result(with_citations=True)
        cits = result.citations
        assert "name" in cits
        assert "age" in cits

    def test_failed_result(self):
        result = StructuredDataResult(
            data=None,
            success=False,
            error_message="test error",
        )
        assert not result.success
        assert result.error_message == "test error"
        assert len(result) == 0

    def test_repr(self):
        result = self._make_result()
        r = repr(result)
        assert "Alice" in r
        assert "StructuredDataResult" in r


# ------------------------------------------------------------------ #
# Integration: return_textmap
# ------------------------------------------------------------------ #
class TestReturnTextmap:
    def test_page_return_textmap(self):
        from natural_pdf import PDF

        pdf = PDF("pdfs/01-practice.pdf")
        text, tm = pdf.pages[0].extract_text(layout=True, return_textmap=True)
        assert isinstance(text, str) and len(text) > 0
        assert tm is not None and hasattr(tm, "search")
        pdf.close()

    def test_page_extract_text_unchanged(self):
        from natural_pdf import PDF

        pdf = PDF("pdfs/01-practice.pdf")
        text = pdf.pages[0].extract_text(layout=True)
        assert isinstance(text, str) and len(text) > 0
        pdf.close()

    def test_region_return_textmap(self):
        from natural_pdf import PDF

        pdf = PDF("pdfs/01-practice.pdf")
        region = pdf.pages[0].find("text:contains('Jungle')").below()
        text, tm = region.extract_text(layout=True, return_textmap=True)
        assert isinstance(text, str) and len(text) > 0
        pdf.close()

    def test_pdf_return_textmap(self):
        from natural_pdf import PDF

        pdf = PDF("pdfs/01-practice.pdf")
        text, infos = pdf.extract_text(layout=True, return_textmap=True)
        assert isinstance(text, str)
        assert isinstance(infos, list) and len(infos) == len(pdf.pages)
        for info in infos:
            assert isinstance(info, PageTextMapInfo)
        pdf.close()


# ------------------------------------------------------------------ #
# Integration: _get_extraction_content
# ------------------------------------------------------------------ #
class TestGetExtractionContent:
    def test_page_with_textmap(self):
        from natural_pdf import PDF

        pdf = PDF("pdfs/01-practice.pdf")
        result = pdf.pages[0]._get_extraction_content(using="text", _return_textmap=True)
        text, textmap, word_elements = result
        assert len(text) > 0 and textmap is not None and len(word_elements) > 0
        pdf.close()

    def test_page_without_textmap(self):
        from natural_pdf import PDF

        pdf = PDF("pdfs/01-practice.pdf")
        result = pdf.pages[0]._get_extraction_content(using="text")
        assert isinstance(result, str) and len(result) > 0
        pdf.close()

    def test_region_with_textmap(self):
        from natural_pdf import PDF

        pdf = PDF("pdfs/01-practice.pdf")
        region = pdf.pages[0].find("text:contains('Jungle')").below()
        text, textmap, words = region._get_extraction_content(using="text", _return_textmap=True)
        assert len(text) > 0 and len(words) > 0
        pdf.close()


# ------------------------------------------------------------------ #
# End-to-end mock test
# ------------------------------------------------------------------ #
class TestExtractWithCitationsMock:
    def _make_mock_client(self, schema_class, response_data):
        mock_client = MagicMock()
        parsed_instance = schema_class(**response_data)
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.parsed = parsed_instance
        mock_client.beta.chat.completions.parse.return_value = mock_completion
        return mock_client

    def test_page_extract_returns_result(self):
        from natural_pdf import PDF

        pdf = PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]

        Shadow = build_shadow_schema(SimpleSchema)
        mock_client = self._make_mock_client(
            Shadow,
            {
                "name": "Jungle Health and Safety Inspection Service",
                "name_source": ["L0: Jungle Health and Safety Inspection Service"],
                "age": "1905",
                "age_source": ["L2: Date: February 3, 1905"],
            },
        )

        result = page.extract(SimpleSchema, client=mock_client, citations=True)

        assert isinstance(result, StructuredDataResult)
        assert result.success
        assert result.name == "Jungle Health and Safety Inspection Service"
        assert result["name"].citations is not None
        assert len(result["name"].citations) > 0
        pdf.close()

    def test_extract_without_citations(self):
        from natural_pdf import PDF

        pdf = PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]

        mock_client = self._make_mock_client(
            SimpleSchema,
            {
                "name": "Test",
                "age": "25",
            },
        )

        result = page.extract(SimpleSchema, client=mock_client)
        assert isinstance(result, StructuredDataResult)
        assert result.name == "Test"
        assert result.age == "25"
        # Citations are empty, not None
        assert len(result["name"].citations) == 0
        pdf.close()

    def test_extracted_returns_stored_result(self):
        from natural_pdf import PDF

        pdf = PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]

        mock_client = self._make_mock_client(
            SimpleSchema,
            {
                "name": "Test",
                "age": "25",
            },
        )

        result1 = page.extract(SimpleSchema, client=mock_client)
        result2 = page.extracted()

        assert result2 is result1
        assert result2.name == "Test"
        pdf.close()

    def test_to_dict(self):
        from natural_pdf import PDF

        pdf = PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]

        mock_client = self._make_mock_client(
            SimpleSchema,
            {
                "name": "Alice",
                "age": "30",
            },
        )

        result = page.extract(SimpleSchema, client=mock_client)
        assert result.to_dict() == {"name": "Alice", "age": "30"}
        pdf.close()

    def test_iteration(self):
        from natural_pdf import PDF

        pdf = PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]

        mock_client = self._make_mock_client(
            SimpleSchema,
            {
                "name": "Alice",
                "age": "30",
            },
        )

        result = page.extract(SimpleSchema, client=mock_client)
        fields = dict(result.items())
        assert "name" in fields and "age" in fields
        assert fields["name"].value == "Alice"
        assert fields["age"].value == "30"
        pdf.close()
