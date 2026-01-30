from __future__ import annotations

import natural_pdf as npdf
from natural_pdf.core.context import PDFContext
from natural_pdf.engine_provider import get_provider
from natural_pdf.flows.flow import Flow
from natural_pdf.selectors import register_pseudo, unregister_pseudo
from natural_pdf.selectors.selector_provider import SelectorResult


class _StubSelectorEngine:
    def __init__(self):
        self.calls = 0

    def query(self, *, context, selector, options):  # pragma: no cover - exercised via tests
        from natural_pdf.elements.element_collection import ElementCollection

        self.calls += 1
        return SelectorResult(elements=ElementCollection([]))


def _register_stub_engine(name: str) -> _StubSelectorEngine:
    provider = get_provider()
    stub = _StubSelectorEngine()
    provider.register("selectors", name, lambda **_: stub, replace=True)
    return stub


def test_page_find_all_uses_registered_selector_engine(practice_pdf):
    stub_engine = _register_stub_engine("test-selectors-page")
    page = practice_pdf.pages[0]
    page.find_all("text", engine="test-selectors-page")
    assert stub_engine.calls == 1


def test_clause_pack_registration_enables_custom_pseudo(practice_pdf):
    @register_pseudo("always-match", replace=True)
    def _always_handler(pseudo, ctx):
        return {"name": ":always", "func": lambda _el: True}

    try:
        page = practice_pdf.pages[0]
        results = page.find_all("text:always-match()")
        assert results
    finally:
        unregister_pseudo("always-match")


def test_regex_pseudo_via_clause_registry(practice_pdf):
    page = practice_pdf.pages[0]
    matches = page.find_all("text:regex('Total')")
    assert matches is not None


def test_contains_clause_still_operational(practice_pdf):
    page = practice_pdf.pages[0]
    matches = page.find_all("text:contains('Total')")
    assert matches is not None


def test_region_find_all_passes_engine_to_page_selector(practice_pdf):
    stub_engine = _register_stub_engine("test-selectors-region")
    page = practice_pdf.pages[0]
    region = page.create_region(0, 0, page.width / 2, page.height / 2)
    region.find_all("text", engine="test-selectors-region")
    assert stub_engine.calls >= 1


def test_flow_find_all_passes_engine_to_pages(practice_pdf):
    stub_engine = _register_stub_engine("test-selectors-flow")
    flow = Flow([practice_pdf.pages[0]], arrangement="vertical", alignment="start")
    flow.find_all("text", engine="test-selectors-flow")
    assert stub_engine.calls >= 1


def test_first_post_pseudo_returns_single_result(practice_pdf):
    page = practice_pdf.pages[0]
    matches = page.find_all("text:first")
    assert len(matches.elements if matches else []) <= 1


def test_above_relational_pseudo_executes(practice_pdf):
    page = practice_pdf.pages[0]
    matches = page.find_all("text:above(text:last)")
    assert matches is not None


def test_context_default_selector_engine(pdf_factory):
    # This test needs fresh PDF because it uses a custom context
    stub_engine = _register_stub_engine("context-selectors")
    context = PDFContext(options={"selector": {"engine": "context-selectors"}})
    pdf = pdf_factory("pdfs/01-practice.pdf")
    # Note: pdf_factory caches, so we create context-aware PDF separately
    pdf_with_context = npdf.PDF("pdfs/01-practice.pdf", context=context)
    try:
        page = pdf_with_context.pages[0]
        page.find_all("text")
    finally:
        pdf_with_context.close()

    assert stub_engine.calls == 1


def test_above_relational_pseudo_executes_standalone(practice_pdf):
    page = practice_pdf.pages[0]
    matches = page.find_all("text:above(text:last)")
    assert matches is not None
