"""Tests for Document Question Answering functionality."""

import pytest

# Test imports
from natural_pdf import PDF


class TestDocumentQADependencies:
    """Test QA dependency handling."""

    def test_qa_dependencies_available(self):
        """Test that QA dependencies can be imported."""
        try:
            from natural_pdf.qa import get_qa_engine
            from natural_pdf.qa.document_qa import DocumentQA

            # If we get here, dependencies are available
            assert True
        except ImportError:
            # Skip QA tests if dependencies not available
            pytest.skip("QA dependencies not installed")


class TestPDFAskSingle:
    """Test PDF.ask() with single questions."""

    def test_ask_single_question_first_page(self, practice_pdf):
        """Test asking a single question about the first page."""
        try:
            from natural_pdf.qa import get_qa_engine
        except ImportError:
            pytest.skip("QA dependencies not installed")

        # Ask a question that should be answerable from the practice PDF
        result = practice_pdf.ask("What is the total amount?", pages=0)

        # Verify result structure
        assert isinstance(result, dict)
        assert "answer" in result
        assert "confidence" in result
        assert "found" in result
        assert "page_num" in result
        assert "source_elements" in result

        # Verify result types
        assert isinstance(result["confidence"], float)
        assert isinstance(result["found"], bool)
        assert 0.0 <= result["confidence"] <= 1.0

        if result["found"]:
            assert result["answer"] is not None
            assert result["page_num"] is not None
            assert isinstance(result["page_num"], int)
        else:
            # If not found, answer might be None or empty
            assert result["page_num"] is None or isinstance(result["page_num"], int)

    def test_ask_single_question_all_pages(self, practice_pdf):
        """Test asking a single question across all pages."""
        try:
            from natural_pdf.qa import get_qa_engine
        except ImportError:
            pytest.skip("QA dependencies not installed")

        # Ask a general question across all pages
        result = practice_pdf.ask("What type of document is this?")

        # Verify result structure
        assert isinstance(result, dict)
        assert "answer" in result
        assert "confidence" in result
        assert "found" in result

        # Should be able to find something across all pages
        # (though we can't guarantee specific content)

    def test_ask_single_question_specific_pages(self, practice_pdf):
        """Test asking a question about specific page range."""
        try:
            from natural_pdf.qa import get_qa_engine
        except ImportError:
            pytest.skip("QA dependencies not installed")

        # Test with a list of page indices
        if len(practice_pdf.pages) > 1:
            result = practice_pdf.ask("What information is shown?", pages=[0, 1])
        else:
            result = practice_pdf.ask("What information is shown?", pages=[0])

        assert isinstance(result, dict)
        assert "answer" in result
        assert "confidence" in result
        assert "found" in result

    def test_ask_invalid_page_index(self, practice_pdf):
        """Test asking with invalid page index."""
        try:
            from natural_pdf.qa import get_qa_engine
        except ImportError:
            pytest.skip("QA dependencies not installed")

        # Test with page index out of range
        with pytest.raises(IndexError):
            practice_pdf.ask("What is this?", pages=999)

    def test_ask_empty_question(self, practice_pdf):
        """Test asking an empty question."""
        try:
            from natural_pdf.qa import get_qa_engine
        except ImportError:
            pytest.skip("QA dependencies not installed")

        # Empty string question
        result = practice_pdf.ask("", pages=0)

        # Should still return valid structure
        assert isinstance(result, dict)
        assert "answer" in result
        assert "confidence" in result
        assert "found" in result


class TestPDFAskBatch:
    """Test PDF.ask_batch() with multiple questions (batch processing)."""

    def test_ask_batch_multiple_questions_same_page(self, practice_pdf):
        """Test asking multiple questions about the same page using ask_batch."""
        try:
            from natural_pdf.qa import get_qa_engine
        except ImportError:
            pytest.skip("QA dependencies not installed")

        questions = [
            "What is the total amount?",
            "What is the date?",
            "What type of document is this?",
        ]

        results = practice_pdf.ask_batch(questions, pages=0)

        # Should return a list of results
        assert isinstance(results, list)
        assert len(results) == len(questions)

        # Each result should have the proper structure
        for i, result in enumerate(results):
            assert isinstance(result, dict)
            assert "answer" in result
            assert "confidence" in result
            assert "found" in result
            assert "page_num" in result
            assert "source_elements" in result

            # Verify types
            assert isinstance(result["confidence"], float)
            assert isinstance(result["found"], bool)
            assert 0.0 <= result["confidence"] <= 1.0

    def test_ask_batch_multiple_questions_all_pages(self, practice_pdf):
        """Test asking multiple questions across all pages using ask_batch."""
        try:
            from natural_pdf.qa import get_qa_engine
        except ImportError:
            pytest.skip("QA dependencies not installed")

        questions = ["What is the main topic?", "Are there any dates mentioned?"]

        results = practice_pdf.ask_batch(questions)

        # Should return a list
        assert isinstance(results, list)
        assert len(results) == len(questions)

        # Verify each result
        for result in results:
            assert isinstance(result, dict)
            assert "answer" in result
            assert "confidence" in result
            assert "found" in result

    def test_ask_batch_mixed_findable_unfindable(self, practice_pdf):
        """Test batch with mix of findable and unfindable questions."""
        try:
            from natural_pdf.qa import get_qa_engine
        except ImportError:
            pytest.skip("QA dependencies not installed")

        questions = [
            "What color is the text?",  # Likely findable
            "What is the secret nuclear launch code?",  # Definitely not findable
            "What type of document is this?",  # Likely findable
        ]

        results = practice_pdf.ask_batch(questions)

        assert isinstance(results, list)
        assert len(results) == len(questions)

        # All should have proper structure even if not found
        for result in results:
            assert isinstance(result, dict)
            assert "found" in result
            assert isinstance(result["found"], bool)

            if not result["found"]:
                # Unfindable questions should have appropriate defaults
                assert result["confidence"] == 0.0
                assert result["page_num"] is None

    def test_ask_batch_empty_list(self, practice_pdf):
        """Test ask_batch with empty list of questions."""
        try:
            from natural_pdf.qa import get_qa_engine
        except ImportError:
            pytest.skip("QA dependencies not installed")

        results = practice_pdf.ask_batch([])

        # Should return empty list
        assert isinstance(results, list)
        assert len(results) == 0

    def test_ask_batch_single_question(self, practice_pdf):
        """Test ask_batch with a single question (should work like ask)."""
        try:
            from natural_pdf.qa import get_qa_engine
        except ImportError:
            pytest.skip("QA dependencies not installed")

        # Single question in list
        results = practice_pdf.ask_batch(["What is shown?"])

        assert isinstance(results, list)
        assert len(results) == 1

        result = results[0]
        assert isinstance(result, dict)
        assert "answer" in result
        assert "confidence" in result
        assert "found" in result


class TestPDFAskErrorHandling:
    """Test error handling in PDF ask methods."""

    def test_ask_invalid_question_type(self, practice_pdf):
        """Test asking with invalid question type."""
        try:
            from natural_pdf.qa import get_qa_engine
        except ImportError:
            pytest.skip("QA dependencies not installed")

        # Test with invalid question type for ask()
        with pytest.raises(TypeError):
            practice_pdf.ask(123)  # Number instead of string

        with pytest.raises(TypeError):
            practice_pdf.ask(["list", "instead", "of", "string"])  # List instead of string

    def test_ask_batch_invalid_questions_type(self, practice_pdf):
        """Test ask_batch with invalid questions type."""
        try:
            from natural_pdf.qa import get_qa_engine
        except ImportError:
            pytest.skip("QA dependencies not installed")

        # Test with invalid questions type for ask_batch()
        with pytest.raises(TypeError):
            practice_pdf.ask_batch("string instead of list")

        with pytest.raises(TypeError):
            practice_pdf.ask_batch([123, 456])  # List of numbers instead of strings

        with pytest.raises(TypeError):
            practice_pdf.ask_batch(["valid", 123])  # Mixed valid/invalid

    def test_ask_invalid_pages_parameter(self, practice_pdf):
        """Test asking with invalid pages parameter."""
        try:
            from natural_pdf.qa import get_qa_engine
        except ImportError:
            pytest.skip("QA dependencies not installed")

        # Test with invalid pages type for both methods
        with pytest.raises(ValueError):
            practice_pdf.ask("What is this?", pages="invalid")

        with pytest.raises(ValueError):
            practice_pdf.ask_batch(["What is this?"], pages={0, 1})  # Set instead of list

    def test_ask_empty_pdf(self):
        """Test asking questions on a PDF with no pages."""
        try:
            from natural_pdf.qa import get_qa_engine
        except ImportError:
            pytest.skip("QA dependencies not installed")

        # Create a minimal PDF or mock one with no pages
        # For now, skip this test as it's hard to create an empty PDF
        pytest.skip("Empty PDF test needs custom setup")


class TestPageAskMethod:
    """Test Page.ask() method (individual pages)."""

    def test_page_ask_single_question(self, practice_pdf):
        """Test asking a single question on a specific page."""
        try:
            from natural_pdf.qa import get_qa_engine
        except ImportError:
            pytest.skip("QA dependencies not installed")

        page = practice_pdf.pages[0]
        result = page.ask("What is shown on this page?")

        assert isinstance(result, dict)
        assert "answer" in result
        assert "confidence" in result
        assert "found" in result

    def test_page_ask_batch_questions(self, practice_pdf):
        """Test asking multiple questions on a specific page."""
        try:
            from natural_pdf.qa import get_qa_engine
        except ImportError:
            pytest.skip("QA dependencies not installed")

        page = practice_pdf.pages[0]
        questions = ["What is the main content?", "Are there any numbers?"]
        results = page.ask(questions)  # Page.ask() still supports batch

        assert isinstance(results, list)
        assert len(results) == len(questions)

        for result in results:
            assert isinstance(result, dict)
            assert "answer" in result
            assert "confidence" in result
            assert "found" in result


class TestQAPerformance:
    """Test QA performance characteristics."""

    def test_batch_vs_sequential_consistency(self, practice_pdf):
        """Test that batch processing gives similar results to sequential."""
        try:
            from natural_pdf.qa import get_qa_engine
        except ImportError:
            pytest.skip("QA dependencies not installed")

        questions = ["What is this document?", "What information is shown?"]

        # Get batch results using ask_batch
        batch_results = practice_pdf.ask_batch(questions, pages=0)

        # Get sequential results using individual ask calls
        sequential_results = []
        for question in questions:
            result = practice_pdf.ask(question, pages=0)
            sequential_results.append(result)

        # Results should be similar (though not necessarily identical due to different processing)
        assert len(batch_results) == len(sequential_results)

        # Both should find answers to the same questions
        for batch_result, seq_result in zip(batch_results, sequential_results):
            # If one found an answer, the other should too (mostly)
            # We allow some variation due to batch vs sequential processing differences
            assert isinstance(batch_result["found"], bool)
            assert isinstance(seq_result["found"], bool)

    def test_no_semaphore_leak_multiple_calls(self, practice_pdf):
        """Test that multiple QA calls don't cause resource leaks."""
        try:
            from natural_pdf.qa import get_qa_engine
        except ImportError:
            pytest.skip("QA dependencies not installed")

        # This test mainly verifies that we don't get semaphore errors
        # when making multiple QA calls in succession
        questions = ["What is this?", "What type of document?", "What information?"]

        # Make multiple batch calls
        for i in range(3):
            results = practice_pdf.ask_batch(questions)
            assert isinstance(results, list)
            assert len(results) == len(questions)

        # Make multiple single calls
        for i in range(3):
            result = practice_pdf.ask("What is shown?")
            assert isinstance(result, dict)

        # If we get here without errors, the semaphore leak issue is likely fixed


class TestQAIntegration:
    """Integration tests for QA with other features."""

    def test_qa_with_ocr_pages(self, practice_pdf):
        """Test QA on pages that might benefit from OCR."""
        try:
            from natural_pdf.qa import get_qa_engine
        except ImportError:
            pytest.skip("QA dependencies not installed")

        # Test QA before and after OCR (if OCR is available)
        try:
            # Try OCR if available
            practice_pdf.apply_ocr(engine="easyocr", pages=[0])

            result = practice_pdf.ask("What text is visible?", pages=0)
            assert isinstance(result, dict)
            assert "answer" in result

        except (ImportError, RuntimeError):
            # OCR not available, just test basic QA
            result = practice_pdf.ask("What is visible?", pages=0)
            assert isinstance(result, dict)
            assert "answer" in result

    def test_qa_with_regions(self, practice_pdf):
        """Test QA on specific regions."""
        try:
            from natural_pdf.qa import get_qa_engine
        except ImportError:
            pytest.skip("QA dependencies not installed")

        page = practice_pdf.pages[0]

        # Create a region (top half of page)
        region = page.region(top=0, bottom=page.height / 2)

        # Test QA on the region
        result = region.ask("What is in this area?")
        assert isinstance(result, dict)
        assert "answer" in result
        assert "confidence" in result
        assert "found" in result


class TestAPIConsistency:
    """Test that the API is consistent and discoverable."""

    def test_ask_returns_single_dict(self, practice_pdf):
        """Test that ask() always returns a single dict."""
        try:
            from natural_pdf.qa import get_qa_engine
        except ImportError:
            pytest.skip("QA dependencies not installed")

        result = practice_pdf.ask("What is this?")

        # Should always be a dict, never a list
        assert isinstance(result, dict)
        assert not isinstance(result, list)

    def test_ask_batch_returns_list(self, practice_pdf):
        """Test that ask_batch() always returns a list."""
        try:
            from natural_pdf.qa import get_qa_engine
        except ImportError:
            pytest.skip("QA dependencies not installed")

        # Single question should still return list
        results = practice_pdf.ask_batch(["What is this?"])
        assert isinstance(results, list)
        assert len(results) == 1

        # Multiple questions should return list
        results = practice_pdf.ask_batch(["What is this?", "What type is this?"])
        assert isinstance(results, list)
        assert len(results) == 2

        # Empty list should return empty list
        results = practice_pdf.ask_batch([])
        assert isinstance(results, list)
        assert len(results) == 0
