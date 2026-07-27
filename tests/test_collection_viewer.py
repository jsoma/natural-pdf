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


def test_viewer_html_keeps_divs_out_of_svg(practice_pdf):
    """Element divs belong in the elements-layer; only rects belong in the svg.

    A <div> inside <svg> is invalid HTML — browsers abort SVG parsing at the
    div, so the highlight JS querying `svg rect` finds nothing.
    """
    page = practice_pdf.pages[0]
    widget = page.viewer()
    html = widget._build_html()
    n = len(widget.pdf_data["elements"])
    assert n > 0

    svg_start = html.index("<svg")
    svg_end = html.index("</svg>")
    svg_inner = html[svg_start:svg_end]

    # No HTML elements inside the SVG; all rects inside it
    assert "<div" not in svg_inner
    assert svg_inner.count("<rect") == n

    # All element divs live in the elements-layer, before the svg opens
    layer_start = html.index("elements-layer")
    assert layer_start < svg_start
    layer_segment = html[layer_start:svg_start]
    assert layer_segment.count('class="pdf-element"') == n
    assert html.count('class="pdf-element"') == n

    # data-element-id hooks preserved on both layers for the JS
    assert 'data-element-id="0"' in layer_segment
    assert 'data-element-id="0"' in svg_inner


def test_collection_viewer_multipage_raises():
    from natural_pdf.elements.element_collection import ElementCollection

    pdf = PDF("pdfs/Atlanta_Public_Schools_GA_sample.pdf")
    try:
        first = list(pdf.pages[0].find_all("text"))
        second = list(pdf.pages[1].find_all("text"))
        assert first and second
        combined = ElementCollection(first + second)
        with pytest.raises(ValueError, match="spans multiple pages"):
            combined.viewer()
    finally:
        pdf.close()


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
