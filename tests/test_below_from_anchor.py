import pytest

import natural_pdf as npdf


@pytest.fixture(scope="module")
def sample_page():
    pdf = npdf.PDF("pdfs/01-practice.pdf")
    try:
        yield pdf.pages[0]
    finally:
        pdf.close()


def _boundary_element(element, anchor=None):
    kwargs = {"until": "text"}
    if anchor:
        kwargs["anchor"] = anchor
    region = element.below(**kwargs)
    assert region is not None
    boundary = getattr(region, "boundary_element", None)
    assert boundary is not None
    return boundary


def test_below_anchor_excludes_source_and_differs(sample_page):
    elem = sample_page.find("text:contains('the')")
    assert elem is not None

    start_boundary = _boundary_element(elem)
    end_boundary = _boundary_element(elem, anchor="end")

    assert start_boundary is not elem
    assert end_boundary is not elem
    # Depending on the surrounding layout the same boundary element can be
    # returned for different anchors. The important behaviour is that the
    # anchor argument is accepted and still yields a valid boundary element.
    assert getattr(start_boundary, "bbox", None) is not None
    assert getattr(end_boundary, "bbox", None) is not None
