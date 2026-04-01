"""Tests for text analysis utilities."""

from natural_pdf.utils.text_utils import detect_repetition


class TestDetectRepetition:
    def test_short_text_always_false(self):
        assert detect_repetition("hello world") is False

    def test_normal_text_not_repetitive(self):
        # Real document text should not trigger
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a sample document with various words and sentences. "
            "It contains different phrases and vocabulary to simulate "
            "natural language text that might appear in a PDF document. "
        ) * 5
        assert detect_repetition(text) is False

    def test_obvious_repetition_detected(self):
        # VLM hallucination loop: same phrase repeated hundreds of times
        text = "the total amount is $100. " * 200
        assert detect_repetition(text) is True

    def test_single_word_loop(self):
        text = "word " * 500
        assert detect_repetition(text) is True

    def test_html_content_not_repetitive(self):
        # Dense HTML from a VLM should not false-positive
        text = "".join(
            f"<tr><td>Item {i}</td><td>${i * 10:.2f}</td><td>{i} units</td></tr>\n"
            for i in range(100)
        )
        assert detect_repetition(text) is False

    def test_threshold_customizable(self):
        text = "repeat " * 200
        # With a very low threshold, even repetitive text passes
        assert detect_repetition(text, threshold=0.01) is False
        # With default threshold, it's caught
        assert detect_repetition(text) is True

    def test_min_length_respected(self):
        text = "word " * 500  # 2500 chars, very repetitive
        assert detect_repetition(text, min_length=5000) is False
        assert detect_repetition(text, min_length=500) is True
