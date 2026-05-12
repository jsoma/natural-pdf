"""Prefilter pdfplumber chars to the target region before table extraction."""

from contextlib import contextmanager

METADATA = {
    "track": "tiny_text",
    "candidate": "prefilter_pdfplumber_chars",
    "cache_only": False,
    "hypothesis": "Filtering chars before pdfplumber text table extraction reduces cell-assignment scans.",
}


@contextmanager
def install():
    from natural_pdf.tables.utils import plumber

    original_filter_page = plumber.filter_page_for_exclusions

    def patched_filter_page_for_exclusions(region, base_page, *, apply_exclusions: bool):
        page = original_filter_page(region, base_page, apply_exclusions=apply_exclusions)
        bbox = getattr(region, "bbox", None)
        if bbox is None:
            return page

        def keep_in_region(obj):
            if obj.get("object_type") != "char":
                return True
            return (
                obj.get("x1", 0) >= bbox[0]
                and obj.get("x0", 0) <= bbox[2]
                and obj.get("bottom", 0) >= bbox[1]
                and obj.get("top", 0) <= bbox[3]
            )

        try:
            return page.filter(keep_in_region)
        except Exception:
            return page

    plumber.filter_page_for_exclusions = patched_filter_page_for_exclusions
    try:
        yield
    finally:
        plumber.filter_page_for_exclusions = original_filter_page
