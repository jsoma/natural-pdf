from __future__ import annotations

import natural_pdf as npdf
from natural_pdf.engine_provider import get_provider
from natural_pdf.selectors import register_pseudo, unregister_pseudo
from natural_pdf.selectors.selector_provider import SelectorResult


class _StubSelectorEngine:
    def __init__(self):
        self.calls = 0

    def query(self, *, context, selector, options):  # pragma: no cover - exercised via tests
        from natural_pdf.elements.element_collection import ElementCollection

        self.calls += 1
        return SelectorResult(elements=ElementCollection([]))


def test_page_find_all_uses_registered_selector_engine():
    provider = get_provider()
    stub_engine = _StubSelectorEngine()
    provider.register("selectors", "test-selectors", lambda **_: stub_engine, replace=True)

    pdf = npdf.PDF("pdfs/01-practice.pdf")
    try:
        page = pdf.pages[0]
        page.find_all("text", engine="test-selectors")
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
