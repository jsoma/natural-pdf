"""Regression test: ElementCollection.viewer() must return a working widget.

It used to pass arguments to a zero-arg Page.viewer() and swallow the
TypeError into a silent None.
"""

import pytest

from natural_pdf import PDF


@pytest.fixture
def practice_pdf():
    pdf = PDF("pdfs/01-practice.pdf")
    try:
        yield pdf
    finally:
        pdf.close()


def test_collection_viewer_returns_widget(practice_pdf):
    from natural_pdf.widgets.viewer import InteractiveViewerWidget

    page = practice_pdf.pages[0]
    words = page.find_all("text")
    assert len(words) > 0

    widget = words.viewer()
    assert isinstance(widget, InteractiveViewerWidget)
    # Only the collection's elements are rendered (plus no chars)
    assert len(widget.pdf_data["elements"]) == len(
        [w for w in words if str(getattr(w, "type", "")).lower() != "char"]
    )


def test_collection_viewer_empty_raises(practice_pdf):
    page = practice_pdf.pages[0]
    empty = page.find_all('text:contains("zzz-no-such-string-zzz")')
    with pytest.raises(ValueError, match="empty collection"):
        empty.viewer()


def test_page_viewer_includes_excluded_elements(practice_pdf):
    """The viewer is a debugging view: exclusions must NOT hide elements."""
    page = practice_pdf.pages[0]
    total = len([e for e in page.find_all("*", apply_exclusions=False)])

    # Exclude the top half of the page
    practice_pdf.add_exclusion(lambda p: p.create_region(0, 0, p.width, p.height / 2))

    widget = page.viewer()
    rendered = len(widget.pdf_data["elements"])
    non_char_total = len(
        [
            e
            for e in page.find_all("*", apply_exclusions=False)
            if str(getattr(e, "type", "")).lower() != "char"
        ]
    )
    assert rendered == non_char_total, "viewer must show excluded elements too"
    assert total > 0
