"""Vectorize decoration detection with per-call arrays."""

from contextlib import contextmanager

METADATA = {
    "track": "vector",
    "candidate": "decorations_ephemeral",
    "cache_only": False,
    "hypothesis": "Columnar per-call arrays can preserve decoration behavior while reducing Python loop cost.",
}


@contextmanager
def install():
    from experiments.performance.patches.vector._decorations import (
        annotate_chars_vectorized,
        propagate_to_words_vectorized,
    )
    from natural_pdf.core.decoration_detector import DecorationDetector

    original_annotate = DecorationDetector.annotate_chars
    original_propagate = DecorationDetector.propagate_to_words

    def patched_annotate_chars(self, char_dicts):
        return annotate_chars_vectorized(self, char_dicts, reuse=False)

    def patched_propagate_to_words(self, word_elements, prepared_char_dicts):
        return propagate_to_words_vectorized(self, word_elements, prepared_char_dicts, reuse=False)

    DecorationDetector.annotate_chars = patched_annotate_chars
    DecorationDetector.propagate_to_words = patched_propagate_to_words
    try:
        yield
    finally:
        DecorationDetector.annotate_chars = original_annotate
        DecorationDetector.propagate_to_words = original_propagate
