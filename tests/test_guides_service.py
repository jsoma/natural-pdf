from natural_pdf import PDF
from natural_pdf.analyzers.guides import Guides
from natural_pdf.flows.flow import Flow
from natural_pdf.flows.region import FlowRegion


def _close(pdf: PDF) -> None:
    try:
        pdf.close()
    except Exception:
        pass


def test_page_guides_returns_guides():
    pdf = PDF("pdfs/01-practice.pdf")
    try:
        page = pdf.pages[0]
        guides = page.guides(verticals=[0, page.width], horizontals=[0, 50])
        assert isinstance(guides, Guides)
        assert guides.context is page
    finally:
        _close(pdf)


def test_region_guides_uses_region_context():
    pdf = PDF("pdfs/01-practice.pdf")
    try:
        page = pdf.pages[0]
        region = page.region(left=0, width=page.width / 2)
        guides = region.guides(verticals=[region.x0, region.x1])
        assert isinstance(guides, Guides)
        assert guides.context is region
    finally:
        _close(pdf)


def test_flowregion_guides():
    pdf = PDF("pdfs/01-practice.pdf")
    try:
        page = pdf.pages[0]
        flow = Flow([page], arrangement="vertical")
        region = page.region(left=0, width=page.width)
        flow_region = FlowRegion(flow, [region])
        guides = flow_region.guides(verticals=[0, page.width])
        assert isinstance(guides, Guides)
        assert guides.context is flow_region
    finally:
        _close(pdf)
