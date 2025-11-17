from natural_pdf import PDF
from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.flows.flow import Flow


class _DummyAnalyzer:
    def __init__(self, host):
        self.host = host

    def analyze_layout(self, **kwargs):
        return []


def _patch_layout(monkeypatch):
    monkeypatch.setattr(
        "natural_pdf.analyzers.layout.layout_analyzer.LayoutAnalyzer",
        _DummyAnalyzer,
    )


def _close_pdf(pdf: PDF) -> None:
    try:
        pdf.close()
    except Exception:
        pass


def test_page_analyze_layout_uses_service(monkeypatch):
    _patch_layout(monkeypatch)
    pdf = PDF("pdfs/01-practice.pdf")
    try:
        page = pdf.pages[0]
        result = page.analyze_layout(engine="mock")
    finally:
        _close_pdf(pdf)

    assert isinstance(result, ElementCollection)


def test_flow_analyze_layout(monkeypatch):
    _patch_layout(monkeypatch)
    pdf = PDF("pdfs/01-practice.pdf")
    try:
        page = pdf.pages[0]
        flow = Flow([page], arrangement="vertical")
        result = flow.analyze_layout(engine="mock")
    finally:
        _close_pdf(pdf)

    assert isinstance(result, ElementCollection)


def test_page_collection_analyze_layout(monkeypatch):
    _patch_layout(monkeypatch)
    pdf = PDF("pdfs/01-practice.pdf")
    try:
        result = pdf.pages.analyze_layout(show_progress=False)
    finally:
        _close_pdf(pdf)

    assert isinstance(result, ElementCollection)


def test_pdf_analyze_layout(monkeypatch):
    _patch_layout(monkeypatch)
    pdf = PDF("pdfs/01-practice.pdf")
    try:
        result = pdf.analyze_layout(show_progress=False)
    finally:
        _close_pdf(pdf)

    assert isinstance(result, ElementCollection)
