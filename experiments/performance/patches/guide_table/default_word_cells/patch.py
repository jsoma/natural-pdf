"""Default Guides.extract_table to batched word-cell extraction when safe."""

from contextlib import contextmanager

METADATA = {
    "track": "guide_table",
    "candidate": "default_word_cells",
    "cache_only": False,
    "hypothesis": "Guide-built indexed cells should use fast word assignment by default when no OCR or custom callback is requested.",
}


@contextmanager
def install():
    from natural_pdf.analyzers.guides import Guides

    original_extract_table = Guides.extract_table

    def patched_extract_table(self, *args, **kwargs):
        if (
            "cell_extract" not in kwargs
            and not kwargs.get("use_ocr", False)
            and kwargs.get("cell_extraction_func") is None
        ):
            kwargs = dict(kwargs)
            kwargs["cell_extract"] = "words"
            kwargs.setdefault("cell_newlines", False)
        return original_extract_table(self, *args, **kwargs)

    Guides.extract_table = patched_extract_table
    try:
        yield
    finally:
        Guides.extract_table = original_extract_table
