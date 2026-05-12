"""Skip table alt-text region scanning to estimate its cost."""

from contextlib import contextmanager

METADATA = {
    "track": "guide_table",
    "candidate": "skip_alt_text_scan",
    "cache_only": False,
    "hypothesis": "Alt-text scanning is unnecessary overhead for common guide/table extraction cases.",
}


@contextmanager
def install():
    from natural_pdf.tables.utils import plumber

    original_has_alt_text_regions = plumber._has_alt_text_regions
    plumber._has_alt_text_regions = lambda region: False
    try:
        yield
    finally:
        plumber._has_alt_text_regions = original_has_alt_text_regions
