"""Tests for the documentation coverage tool."""

import json
import tempfile
from pathlib import Path

import pytest

from scripts.doc_coverage.analyzers import (
    MethodCall,
    build_api_catalog,
    calculate_coverage,
    extract_calls,
    resolve_call,
)
from scripts.doc_coverage.extractors import CodeSample, extract_from_markdown, extract_from_notebook


class TestCodeExtraction:
    """Tests for code sample extraction."""

    def test_extract_from_markdown(self, tmp_path):
        """Test extracting Python code blocks from markdown."""
        md_content = """
# Example

Here's some code:

```python
pdf = PDF("test.pdf")
page = pdf.pages[0]
text = page.extract_text()
```

And more text.

```python
region = page.find('text:bold')
```
"""
        md_file = tmp_path / "test.md"
        md_file.write_text(md_content)

        samples = extract_from_markdown(md_file)

        assert len(samples) == 2
        assert "pdf = PDF" in samples[0].source
        assert "region = page.find" in samples[1].source
        assert all(s.file_path == str(md_file) for s in samples)

    def test_extract_from_notebook(self, tmp_path):
        """Test extracting code from Jupyter notebooks."""
        nb_content = {
            "cells": [
                {"cell_type": "markdown", "source": ["# Title\n", "Some text"]},
                {"cell_type": "code", "source": ["pdf = PDF('test.pdf')\n", "page = pdf.pages[0]"]},
                {"cell_type": "code", "source": ["%matplotlib inline"]},  # Magic command
                {"cell_type": "code", "source": ["text = page.extract_text()"]},
            ]
        }
        nb_file = tmp_path / "test.ipynb"
        nb_file.write_text(json.dumps(nb_content))

        samples = extract_from_notebook(nb_file)

        # Should skip markdown and magic command cells
        assert len(samples) == 2
        assert "pdf = PDF" in samples[0].source
        assert "page.extract_text()" in samples[1].source


class TestAstWalker:
    """Tests for AST-based method call extraction."""

    def test_simple_method_call(self):
        """Test extracting simple method calls."""
        code = "page.find('text:bold')"
        calls = extract_calls(code)

        assert len(calls) == 1
        assert calls[0].receiver == "page"
        assert calls[0].method == "find"

    def test_chained_calls(self):
        """Test extracting chained method calls."""
        code = "page.find('text').below().extract_text()"
        calls = extract_calls(code)

        assert len(calls) == 3
        methods = [c.method for c in calls]
        assert "find" in methods
        assert "below" in methods
        assert "extract_text" in methods

    def test_subscript_receiver(self):
        """Test receiver with subscript access."""
        code = "pdf.pages[0].extract_text()"
        calls = extract_calls(code)

        # Should capture both the method call and the attribute access
        assert len(calls) == 2
        methods = [c.method for c in calls]
        assert "extract_text" in methods
        assert "pages" in methods  # Attribute access on pdf.pages

    def test_attribute_access(self):
        """Test capturing attribute access (not just calls)."""
        code = "width = page.width"
        calls = extract_calls(code)

        assert len(calls) == 1
        assert calls[0].receiver == "page"
        assert calls[0].method == "width"
        assert calls[0].is_call is False

    def test_invalid_syntax(self):
        """Test handling of invalid Python syntax."""
        code = "this is not valid python {"
        calls = extract_calls(code)

        assert calls == []


class TestMatcher:
    """Tests for method call resolution."""

    @pytest.fixture
    def sample_catalog(self):
        """Sample API catalog for testing."""
        return {
            "PDF": ["pages", "close"],
            "Page": ["find", "find_all", "extract_text", "apply_ocr"],
            "Region": ["extract_text", "show"],
            "Element": ["extract_text", "below"],
            "TextElement": ["extract_text", "below"],
            "ElementCollection": ["filter", "apply"],
        }

    def test_exact_variable_match(self, sample_catalog):
        """Test resolution with exact variable name match."""
        call = MethodCall(receiver="page", method="find", line=1)
        result = resolve_call(call, sample_catalog)

        assert result == "Page.find"

    def test_suffix_pattern_match(self, sample_catalog):
        """Test resolution with suffix-based matching."""
        call = MethodCall(receiver="header_region", method="extract_text", line=1)
        result = resolve_call(call, sample_catalog)

        assert result == "Region.extract_text"

    def test_method_chain_pattern(self, sample_catalog):
        """Test resolution for method chain results."""
        call = MethodCall(receiver="find()", method="extract_text", line=1)
        result = resolve_call(call, sample_catalog)

        # find() returns TextElement (not Element, which isn't in real API catalog)
        assert result == "TextElement.extract_text"

    def test_unique_method_fallback(self, sample_catalog):
        """Test fallback to unique method match."""
        call = MethodCall(receiver="unknown", method="apply_ocr", line=1)
        result = resolve_call(call, sample_catalog)

        # apply_ocr is unique to Page
        assert result == "Page.apply_ocr"

    def test_ambiguous_method(self, sample_catalog):
        """Test that ambiguous methods return None."""
        call = MethodCall(receiver="unknown", method="extract_text", line=1)
        result = resolve_call(call, sample_catalog)

        # extract_text exists on multiple classes
        assert result is None


class TestCoverageCalculation:
    """Tests for coverage calculation."""

    def test_basic_coverage(self):
        """Test basic coverage calculation."""
        catalog = {
            "Page": ["find", "extract_text"],
            "Region": ["show"],
        }
        samples = [
            CodeSample(source="page.find('text:bold')", file_path="test.md", location="line_1")
        ]

        result = calculate_coverage(samples, catalog)

        assert result.total_methods == 3
        assert result.covered_methods == 1
        assert result.coverage_percent == pytest.approx(33.33, rel=0.1)
        assert "Page.find" in result.method_hits
        assert result.method_hits["Page.find"] == 1
        assert "Page.extract_text" in result.uncovered
        assert "Region.show" in result.uncovered

    def test_multiple_hits(self):
        """Test counting multiple hits for same method."""
        catalog = {"Page": ["find"]}
        samples = [
            CodeSample(source="page.find('a')", file_path="a.md", location="1"),
            CodeSample(source="page.find('b')", file_path="b.md", location="1"),
        ]

        result = calculate_coverage(samples, catalog)

        assert result.method_hits["Page.find"] == 2


class TestApiCatalog:
    """Tests for API catalog building."""

    def test_builds_catalog(self):
        """Test that catalog builds for natural_pdf module."""
        catalog = build_api_catalog("natural_pdf")

        # Should have key classes
        assert "PDF" in catalog
        assert "Page" in catalog

        # Should have expected methods
        assert "find" in catalog.get("Page", [])
        assert "extract_text" in catalog.get("Page", [])

    def test_excludes_private_methods(self):
        """Test that private methods are excluded."""
        catalog = build_api_catalog("natural_pdf")

        for cls, methods in catalog.items():
            for method in methods:
                assert not method.startswith("_"), f"Private method {cls}.{method} included"
