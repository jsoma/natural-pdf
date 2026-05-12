"""Skip decoration detection to estimate materialization overhead."""

from contextlib import contextmanager

METADATA = {
    "track": "page_materialization",
    "candidate": "defer_decorations",
    "cache_only": False,
    "hypothesis": "Decoration detection is a measurable eager page-materialization cost.",
}


@contextmanager
def install():
    from natural_pdf.core.decoration_detector import DecorationDetector

    original_annotate = DecorationDetector.annotate_chars
    original_propagate = DecorationDetector.propagate_to_words

    def patched_annotate_chars(self, char_dicts):
        for ch in char_dicts:
            ch.setdefault("strike", False)
            ch.setdefault("underline", False)
            ch.setdefault("highlight", False)

    def patched_propagate_to_words(self, word_elements, prepared_char_dicts):
        for word in word_elements:
            word._obj.setdefault("strike", False)
            word._obj.setdefault("underline", False)
            word._obj.setdefault("highlight", False)

    DecorationDetector.annotate_chars = patched_annotate_chars
    DecorationDetector.propagate_to_words = patched_propagate_to_words
    try:
        yield
    finally:
        DecorationDetector.annotate_chars = original_annotate
        DecorationDetector.propagate_to_words = original_propagate
