"""Tests for PDF-level .ask() and .ask_batch() using the new extract-based flow."""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock, patch

from pydantic import BaseModel, Field, create_model

from natural_pdf.core.pdf import PDF
from natural_pdf.extraction.result import StructuredDataResult


def _make_answer_result(answer: str, model_used: str = "test") -> StructuredDataResult:
    """Build a StructuredDataResult with an 'answer' field."""
    schema = create_model("_QA", answer=(Optional[str], Field(None)))
    instance = schema(answer=answer)
    return StructuredDataResult(
        data=instance,
        success=True,
        error_message=None,
        raw_output=None,
        model_used=model_used,
    )


def test_pdf_ask_returns_structured_data_result(monkeypatch):
    """PDF.ask() should return StructuredDataResult via page-level extract."""
    pdf = PDF("pdfs/01-practice.pdf")

    fake_result = _make_answer_result("page-1")

    # Patch the page-level .extract() to return our fake result
    for page in pdf.pages:
        monkeypatch.setattr(page, "extract", lambda **kw: fake_result)

    try:
        result = pdf.ask("Which page?", pages=0)
    finally:
        pdf.close()

    assert isinstance(result, StructuredDataResult)
    assert result.answer == "page-1"
    assert result.success is True


def test_pdf_ask_batch_returns_list(monkeypatch):
    """PDF.ask_batch() should return a list of StructuredDataResult."""
    pdf = PDF("pdfs/01-practice.pdf")

    call_count = 0

    def fake_extract(**kwargs):
        nonlocal call_count
        call_count += 1
        return _make_answer_result(f"answer-{call_count}")

    for page in pdf.pages:
        monkeypatch.setattr(page, "extract", fake_extract)

    try:
        results = pdf.ask_batch(["first?", "second?"], pages=0)
    finally:
        pdf.close()

    assert len(results) == 2
    assert all(isinstance(r, StructuredDataResult) for r in results)
    assert results[0].answer == "answer-1"
    assert results[1].answer == "answer-2"


def test_pdf_ask_batch_resolves_pages_once(monkeypatch):
    """ask_batch should call _resolve_qa_pages only once, not per question."""
    pdf = PDF("pdfs/01-practice.pdf")

    resolve_count = 0
    original_resolve = pdf._resolve_qa_pages

    def counting_resolve(pages):
        nonlocal resolve_count
        resolve_count += 1
        return original_resolve(pages)

    monkeypatch.setattr(pdf, "_resolve_qa_pages", counting_resolve)

    fake_result = _make_answer_result("batch-answer")
    for page in pdf.pages:
        monkeypatch.setattr(page, "extract", lambda **kw: fake_result)

    try:
        results = pdf.ask_batch(["q1", "q2", "q3"], pages=0)
    finally:
        pdf.close()

    assert len(results) == 3
    assert resolve_count == 1, f"Expected 1 _resolve_qa_pages call, got {resolve_count}"


def test_pdf_ask_empty_pages():
    """PDF.ask() with no valid pages returns blank StructuredDataResult."""
    pdf = PDF("pdfs/01-practice.pdf")
    try:
        # Use an out-of-range page index to test error handling
        try:
            result = pdf.ask("test?", pages=9999)
            # If no error, result should be blank with success=False
            assert result.answer is None
            assert result.success is False
        except IndexError:
            # Expected for invalid page index
            pass
    finally:
        pdf.close()
