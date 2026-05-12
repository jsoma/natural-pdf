"""Apply stricter pdfplumber text-strategy defaults for comparison."""

from contextlib import contextmanager

METADATA = {
    "track": "tiny_text",
    "candidate": "tuned_text_strategy",
    "cache_only": False,
    "hypothesis": "Stricter text-strategy settings reduce pdfplumber tiny-text table search work.",
}


@contextmanager
def install():
    from natural_pdf.tables.utils import plumber

    original_inject = plumber.inject_text_tolerances

    def patched_inject_text_tolerances(region, table_settings):
        original_inject(region, table_settings)
        uses_text = "text" in (
            table_settings.get("vertical_strategy"),
            table_settings.get("horizontal_strategy"),
        )
        if not uses_text:
            return
        table_settings.setdefault("min_words_vertical", 2)
        table_settings.setdefault("min_words_horizontal", 1)
        table_settings.setdefault("intersection_tolerance", 1)
        table_settings.setdefault("edge_min_length", 1)

    plumber.inject_text_tolerances = patched_inject_text_tolerances
    try:
        yield
    finally:
        plumber.inject_text_tolerances = original_inject
