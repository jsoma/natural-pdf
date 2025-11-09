from __future__ import annotations

import natural_pdf as npdf
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


def test_page_find_all_uses_registered_selector_engine():
    stub_engine = _register_stub_engine("test-selectors-page")

    pdf = npdf.PDF("pdfs/01-practice.pdf")
    try:
        page = pdf.pages[0]
        page.find_all("text", engine="test-selectors-page")
    finally:
        pdf.close()

    assert stub_engine.calls == 1


def test_clause_pack_registration_enables_custom_pseudo():
    pdf = npdf.PDF("pdfs/01-practice.pdf")

    @register_pseudo("always-match", replace=True)
    def _always_handler(pseudo, ctx):
        return {"name": ":always", "func": lambda _el: True}

    try:
        page = pdf.pages[0]
        results = page.find_all("text:always-match()")
        assert results
    finally:
        unregister_pseudo("always-match")
        pdf.close()


def test_regex_pseudo_via_clause_registry():
    pdf = npdf.PDF("pdfs/01-practice.pdf")
    try:
        page = pdf.pages[0]
        matches = page.find_all("text:regex('Total')")
        assert matches is not None
    finally:
        pdf.close()


def test_contains_clause_still_operational():
    pdf = npdf.PDF("pdfs/01-practice.pdf")
    try:
        page = pdf.pages[0]
        matches = page.find_all("text:contains('Total')")
        assert matches is not None
    finally:
        pdf.close()


def test_region_find_all_passes_engine_to_page_selector():
    stub_engine = _register_stub_engine("test-selectors-region")
    pdf = npdf.PDF("pdfs/01-practice.pdf")
    try:
        page = pdf.pages[0]
        region = page.create_region(0, 0, page.width / 2, page.height / 2)
        region.find_all("text", engine="test-selectors-region")
    finally:
        pdf.close()

    assert stub_engine.calls >= 1


def test_flow_find_all_passes_engine_to_pages():
    stub_engine = _register_stub_engine("test-selectors-flow")
    pdf = npdf.PDF("pdfs/01-practice.pdf")
    try:
        flow = Flow([pdf.pages[0]], arrangement="vertical", alignment="start")
        flow.find_all("text", engine="test-selectors-flow")
    finally:
        pdf.close()

    assert stub_engine.calls >= 1


def test_first_post_pseudo_returns_single_result():
    pdf = npdf.PDF("pdfs/01-practice.pdf")
    try:
        page = pdf.pages[0]
        matches = page.find_all("text:first")
        assert len(matches.elements if matches else []) <= 1
    finally:
        pdf.close()


def test_above_relational_pseudo_executes():
    pdf = npdf.PDF("pdfs/01-practice.pdf")
    try:
        page = pdf.pages[0]
        matches = page.find_all("text:above(text:last)")
        assert matches is not None
    finally:
        pdf.close()
