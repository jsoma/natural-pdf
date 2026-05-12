"""Prefer vector line detection in guides to avoid pixel rendering."""

from contextlib import contextmanager

METADATA = {
    "track": "guide_table",
    "candidate": "vector_lines_only",
    "cache_only": False,
    "hypothesis": "Vector line detection avoids render cost when PDFs already expose line objects.",
}


@contextmanager
def install():
    from natural_pdf.analyzers.guides.base import Guides, GuidesList

    original_list_from_lines = GuidesList.from_lines
    original_guides_from_lines = Guides.from_lines

    def patched_list_from_lines(self, *args, **kwargs):
        kwargs = dict(kwargs)
        kwargs.setdefault("detection_method", "vector")
        return original_list_from_lines(self, *args, **kwargs)

    def patched_guides_from_lines(cls, *args, **kwargs):
        kwargs = dict(kwargs)
        kwargs.setdefault("detection_method", "vector")
        return original_guides_from_lines.__func__(cls, *args, **kwargs)

    GuidesList.from_lines = patched_list_from_lines
    Guides.from_lines = classmethod(patched_guides_from_lines)
    try:
        yield
    finally:
        GuidesList.from_lines = original_list_from_lines
        Guides.from_lines = original_guides_from_lines
