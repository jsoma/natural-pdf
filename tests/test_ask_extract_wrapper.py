"""Tests for .ask() → .extract() wrapper (QAService rewrite)."""

from __future__ import annotations

from typing import Optional

import pytest
from pydantic import Field, create_model

import natural_pdf as npdf
from natural_pdf.extraction.result import StructuredDataResult
from natural_pdf.services.qa_service import QAService


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


class TestAskExtractWrapper:
    """Verify that .ask() returns StructuredDataResult and uses .extract()."""

    def test_page_ask_returns_structured_data_result(self, monkeypatch):
        pdf = npdf.PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]

        fake_result = _make_answer_result("42")
        monkeypatch.setattr(page, "extract", lambda **kw: fake_result)

        try:
            result = page.ask("What is the answer?")
        finally:
            pdf.close()

        assert isinstance(result, StructuredDataResult)
        assert result.answer == "42"
        assert result.success is True

    def test_page_ask_with_client_kwarg(self, monkeypatch):
        """Passing client= should forward it to .extract()."""
        pdf = npdf.PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]

        captured_kwargs = {}

        def capture_extract(**kwargs):
            captured_kwargs.update(kwargs)
            return _make_answer_result("client-answer")

        monkeypatch.setattr(page, "extract", capture_extract)

        fake_client = object()
        try:
            result = page.ask("test?", client=fake_client, using="vision")
        finally:
            pdf.close()

        assert result.answer == "client-answer"
        assert captured_kwargs["client"] is fake_client
        assert captured_kwargs["using"] == "vision"

    def test_page_ask_with_engine_vlm(self, monkeypatch):
        """Passing engine='vlm' should forward it to .extract()."""
        pdf = npdf.PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]

        captured_kwargs = {}

        def capture_extract(**kwargs):
            captured_kwargs.update(kwargs)
            return _make_answer_result("vlm-answer")

        monkeypatch.setattr(page, "extract", capture_extract)

        try:
            result = page.ask("test?", engine="vlm")
        finally:
            pdf.close()

        assert result.answer == "vlm-answer"
        assert captured_kwargs["engine"] == "vlm"
        assert captured_kwargs["using"] == "vision"

    def test_page_ask_default_uses_doc_qa(self, monkeypatch):
        """Default .ask() (no client, no engine) should use engine='doc_qa'."""
        pdf = npdf.PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]

        captured_kwargs = {}

        def capture_extract(**kwargs):
            captured_kwargs.update(kwargs)
            return _make_answer_result("docqa-answer")

        monkeypatch.setattr(page, "extract", capture_extract)

        try:
            result = page.ask("test?")
        finally:
            pdf.close()

        assert result.answer == "docqa-answer"
        assert captured_kwargs["engine"] == "doc_qa"

    def test_batch_questions(self, monkeypatch):
        """Passing a list of questions returns a list of results."""
        pdf = npdf.PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]

        call_count = 0

        def counting_extract(**kwargs):
            nonlocal call_count
            call_count += 1
            return _make_answer_result(f"answer-{call_count}")

        monkeypatch.setattr(page, "extract", counting_extract)

        try:
            results = page.ask(["q1", "q2", "q3"])
        finally:
            pdf.close()

        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(r, StructuredDataResult) for r in results)
        assert results[0].answer == "answer-1"
        assert results[2].answer == "answer-3"


class TestQAServiceHelpers:
    """Test internal QAService helper methods."""

    def test_blank_structured_result(self):
        result = QAService._blank_structured_result("test question?")
        assert isinstance(result, StructuredDataResult)
        assert result.success is False
        assert result.error_message is not None
        assert result.answer is None

    def test_extract_confidence_from_answer_confidence(self):
        result = _make_answer_result("test", confidence=0.85)
        conf = QAService._extract_confidence(result)
        assert conf == 0.85

    def test_extract_confidence_from_fields(self):
        """Confidence from _fields should be preferred."""
        from unittest.mock import MagicMock

        result = _make_answer_result("test", confidence=0.5)
        field_info = MagicMock()
        field_info.confidence = 0.99
        result._fields = {"answer": field_info}
        conf = QAService._extract_confidence(result)
        assert conf == 0.99

    def test_extract_confidence_zero_when_answer_present(self):
        """An answer with no confidence data should return 0.0."""
        from pydantic import create_model

        schema = create_model("_QA", answer=(Optional[str], ...))
        instance = schema(answer="some answer")
        result = StructuredDataResult(data=instance, success=True, error_message=None)
        conf = QAService._extract_confidence(result)
        assert conf == 0.0

    def test_extract_confidence_neg_inf_no_answer(self):
        """No answer at all should return -inf."""
        from pydantic import create_model

        schema = create_model("_QA", answer=(Optional[str], ...))
        instance = schema(answer=None)
        result = StructuredDataResult(data=instance, success=True, error_message=None)
        conf = QAService._extract_confidence(result)
        assert conf == float("-inf")

    def test_extract_confidence_failed_result(self):
        result = StructuredDataResult(
            data=None,
            success=False,
            error_message="failed",
        )
        conf = QAService._extract_confidence(result)
        assert conf == float("-inf")

    def test_coerce_questions_single(self):
        questions, is_batch = QAService._coerce_questions("single")
        assert questions == ["single"]
        assert is_batch is False

    def test_coerce_questions_list(self):
        questions, is_batch = QAService._coerce_questions(["a", "b"])
        assert questions == ["a", "b"]
        assert is_batch is True

    def test_debug_forwarded_when_true(self, monkeypatch):
        """debug=True should be forwarded to .extract()."""
        import natural_pdf as npdf

        pdf = npdf.PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]
        captured = {}

        def capture_extract(**kwargs):
            captured.update(kwargs)
            return _make_answer_result("ok")

        monkeypatch.setattr(page, "extract", capture_extract)
        try:
            page.ask("test?", debug=True)
        finally:
            pdf.close()
        assert captured.get("debug") is True

    def test_debug_not_forwarded_when_false(self, monkeypatch):
        """debug=False (default) should not add debug to extract kwargs."""
        import natural_pdf as npdf

        pdf = npdf.PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]
        captured = {}

        def capture_extract(**kwargs):
            captured.update(kwargs)
            return _make_answer_result("ok")

        monkeypatch.setattr(page, "extract", capture_extract)
        try:
            page.ask("test?")
        finally:
            pdf.close()
        assert "debug" not in captured

    def test_citations_default_false(self, monkeypatch):
        """Default .ask() should pass citations=False to .extract()."""
        import natural_pdf as npdf

        pdf = npdf.PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]
        captured = {}

        def capture_extract(**kwargs):
            captured.update(kwargs)
            return _make_answer_result("ok")

        monkeypatch.setattr(page, "extract", capture_extract)
        try:
            page.ask("test?")
        finally:
            pdf.close()
        assert captured["citations"] is False

    def test_citations_passthrough(self, monkeypatch):
        """citations=True should be forwarded to .extract()."""
        import natural_pdf as npdf

        pdf = npdf.PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]
        captured = {}

        def capture_extract(**kwargs):
            captured.update(kwargs)
            return _make_answer_result("ok")

        monkeypatch.setattr(page, "extract", capture_extract)
        try:
            page.ask("test?", citations=True)
        finally:
            pdf.close()
        assert captured["citations"] is True

    def test_ask_uses_fixed_private_analysis_key_with_overwrite(self, monkeypatch):
        pdf = npdf.PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]
        captured_calls = []

        def capture_extract(**kwargs):
            captured_calls.append(kwargs)
            return _make_answer_result("ok")

        monkeypatch.setattr(page, "extract", capture_extract)
        try:
            page.ask("first?")
            page.ask("second?")
        finally:
            pdf.close()

        assert len(captured_calls) == 2
        assert all(call["analysis_key"] == "_qa" for call in captured_calls)
        assert all(call["overwrite"] is True for call in captured_calls)


class TestIntegration:
    """End-to-end integration test: .ask() → QAService → extract() → doc_qa."""

    def test_ask_full_delegation_chain(self, monkeypatch):
        """Verify the full .ask() → QAService → ExtractionService → doc_qa chain.

        Uses a real PDF but mocks the DocumentQA engine to avoid model deps.
        """
        from unittest.mock import MagicMock

        import natural_pdf as npdf

        pdf = npdf.PDF("pdfs/01-practice.pdf")
        page = pdf.pages[0]

        # Mock the DocumentQA engine at the extraction service level
        fake_qa_result = MagicMock()
        fake_qa_result.answer = "mocked answer"
        fake_qa_result.confidence = 0.95
        fake_qa_result.start = 0
        fake_qa_result.end = 10

        mock_engine = MagicMock()
        mock_engine.answer_question.return_value = [
            {"answer": "mocked answer", "score": 0.95, "start": 0, "end": 10}
        ]

        # Patch the extraction service's doc_qa engine resolution
        monkeypatch.setattr(
            "natural_pdf.qa.document_qa.get_qa_engine",
            lambda **kwargs: mock_engine,
        )

        try:
            result = page.ask("What is the title?")
        finally:
            pdf.close()

        assert isinstance(result, StructuredDataResult)
        # The result should have been processed through the full chain
        assert result.data is not None
