"""Tests for PageCollection.ask() using the new extract-based flow."""

from __future__ import annotations

from typing import Optional

from pydantic import Field, create_model

import natural_pdf as npdf
from natural_pdf.core.page_collection import PageCollection
from natural_pdf.extraction.result import StructuredDataResult


def _make_answer_result(answer: str, confidence: float = 0.5) -> StructuredDataResult:
    schema = create_model(
        "_QA",
        answer=(Optional[str], Field(None)),
        answer_confidence=(Optional[float], Field(None)),
    )
    instance = schema(answer=answer, answer_confidence=confidence)
    return StructuredDataResult(
        data=instance,
        success=True,
        error_message=None,
        raw_output=None,
        model_used="test",
    )


def test_page_collection_ask_returns_structured_data_result(monkeypatch):
    pdf = npdf.PDF("pdfs/01-practice.pdf")
    try:
        pages = PageCollection([pdf.pages[0]])

        fake_result = _make_answer_result("page-answer")
        monkeypatch.setattr(pdf.pages[0], "extract", lambda **kw: fake_result)

        result = pages.ask("Which page?", min_confidence=0.0)
    finally:
        pdf.close()

    assert isinstance(result, StructuredDataResult)
    assert result.answer == "page-answer"
