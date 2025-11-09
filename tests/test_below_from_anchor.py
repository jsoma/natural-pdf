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
    assert start_boundary is not end_boundary
    assert start_boundary.bbox != end_boundary.bbox
