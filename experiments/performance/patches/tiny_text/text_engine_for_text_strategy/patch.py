"""Route pdfplumber text-strategy table calls to Natural PDF's text engine."""

from contextlib import contextmanager

METADATA = {
    "track": "tiny_text",
    "candidate": "text_engine_for_text_strategy",
    "cache_only": False,
    "hypothesis": "Natural PDF text-table extraction can avoid pdfplumber char-in-cell blowups for text strategies.",
}


@contextmanager
def install():
    from natural_pdf.services.table_service import TableService

    original_extract_table = TableService.extract_table

    def patched_extract_table(self, host, *args, **kwargs):
        method = kwargs.get("method")
        table_settings = kwargs.get("table_settings") or {}
        uses_text_strategy = (
            table_settings.get("vertical_strategy") == "text"
            and table_settings.get("horizontal_strategy") == "text"
        )
        if method == "pdfplumber" and uses_text_strategy:
            kwargs = dict(kwargs)
            kwargs["method"] = "text"
            kwargs.setdefault("cell_extract", "words")
            kwargs.setdefault("cell_newlines", False)
        return original_extract_table(self, host, *args, **kwargs)

    TableService.extract_table = patched_extract_table
    try:
        yield
    finally:
        TableService.extract_table = original_extract_table
