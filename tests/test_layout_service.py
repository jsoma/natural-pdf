from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.flows.flow import Flow


class _DummyAnalyzer:
    def __init__(self, host):
        self.host = host

    def analyze_layout(self, **kwargs):
        assert kwargs.get("existing") == "replace"
        return []


def _patch_layout(monkeypatch):
    monkeypatch.setattr(
        "natural_pdf.analyzers.layout.layout_analyzer.LayoutAnalyzer",
        _DummyAnalyzer,
    )


def test_page_analyze_layout_uses_service(monkeypatch, practice_pdf_fresh):
    _patch_layout(monkeypatch)
    page = practice_pdf_fresh.pages[0]
    result = page.analyze_layout(engine="mock")
    assert isinstance(result, ElementCollection)


def test_page_analyze_layout_accepts_positional_engine(monkeypatch, practice_pdf_fresh):
    _patch_layout(monkeypatch)
    page = practice_pdf_fresh.pages[0]
    result = page.analyze_layout("mock")
    assert isinstance(result, ElementCollection)


def test_flow_analyze_layout(monkeypatch, practice_pdf_fresh):
    _patch_layout(monkeypatch)
    page = practice_pdf_fresh.pages[0]
    flow = Flow([page], arrangement="vertical")
    result = flow.analyze_layout(engine="mock")
    assert isinstance(result, ElementCollection)


def test_page_collection_analyze_layout(monkeypatch, practice_pdf_fresh):
    _patch_layout(monkeypatch)
    result = practice_pdf_fresh.pages.analyze_layout(show_progress=False)
    assert isinstance(result, ElementCollection)


def test_pdf_analyze_layout(monkeypatch, practice_pdf_fresh):
    _patch_layout(monkeypatch)
    result = practice_pdf_fresh.analyze_layout(show_progress=False)
    assert isinstance(result, ElementCollection)


def test_pdf_analyze_layout_positional_engine(monkeypatch, practice_pdf_fresh):
    _patch_layout(monkeypatch)
    result = practice_pdf_fresh.analyze_layout("mock", show_progress=False)
    assert isinstance(result, ElementCollection)
