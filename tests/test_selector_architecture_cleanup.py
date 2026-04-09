from __future__ import annotations

from types import SimpleNamespace

from natural_pdf.core.page_collection import PageCollection
from natural_pdf.core.pdf_collection import PDFCollection
from natural_pdf.core.selector_utils import execute_parsed_selector
from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.selectors.parser import build_text_contains_selector, parse_selector
from natural_pdf.services.selector_service import SelectorService


def test_execute_parsed_selector_preserves_stable_first_appearance_order(practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]
    selector = parse_selector("text:first|text:first|text:last")

    matches = execute_parsed_selector(page, selector)

    first = page.find_all("text:first").elements[0]
    last = page.find_all("text:last").elements[0]
    assert [id(el) for el in matches.elements] == [id(first), id(last)]


def test_collection_hosts_normalize_text_shortcuts(monkeypatch, practice_pdf):
    service = SelectorService(practice_pdf._context)
    expected_selector = build_text_contains_selector(["Total", "Date"])
    captured = {}

    def _fake_find_all_page(self, page, **kwargs):
        captured["page"] = kwargs
        return ElementCollection([])

    def _fake_find_all_pdf(self, pdf, **kwargs):
        captured["pdf"] = kwargs
        return ElementCollection([])

    monkeypatch.setattr(SelectorService, "_find_all_page", _fake_find_all_page)
    monkeypatch.setattr(SelectorService, "_find_all_pdf", _fake_find_all_pdf)

    pdf = practice_pdf
    pages = PageCollection([practice_pdf.pages[0]], context=practice_pdf._context)
    pdfs = PDFCollection([practice_pdf])

    service._find_all_pdf(pdf, text=["Total", "Date"])
    service._find_all_page_collection(pages, text=["Total", "Date"])
    service._find_all_pdf_collection(pdfs, text=["Total", "Date"])

    assert captured["page"]["selector"] == expected_selector
    assert "text" not in captured["page"]
    assert captured["pdf"]["selector"] == expected_selector
    assert "text" not in captured["pdf"]


def test_execute_selector_query_uses_provider_only_for_non_native_engine(monkeypatch, practice_pdf):
    page = practice_pdf.pages[0]
    provider_calls = {"count": 0}
    native_calls = {"count": 0}

    def _fake_provider(*args, **kwargs):
        provider_calls["count"] += 1
        return ElementCollection([])

    def _fake_native(*args, **kwargs):
        native_calls["count"] += 1
        return ElementCollection([])

    monkeypatch.setattr(
        "natural_pdf.selectors.selector_provider.run_selector_engine", _fake_provider
    )
    monkeypatch.setattr("natural_pdf.core.selector_utils._run_native_selector", _fake_native)

    page.find_all("text", engine="native")
    page.find_all("text", engine="test-selectors-provider")

    assert native_calls["count"] == 1
    assert provider_calls["count"] == 1


def test_execute_selector_query_uses_resolved_provider_engine(monkeypatch, practice_pdf):
    page = practice_pdf.pages[0]
    provider_calls = {"count": 0}
    native_calls = {"count": 0}

    def _fake_provider(*args, **kwargs):
        provider_calls["count"] += 1
        return ElementCollection([])

    def _fake_native(*args, **kwargs):
        native_calls["count"] += 1
        return ElementCollection([])

    monkeypatch.setattr(
        "natural_pdf.selectors.selector_provider.resolve_selector_engine_name",
        lambda host, requested: "resolved-provider",
    )
    monkeypatch.setattr(
        "natural_pdf.selectors.selector_provider.run_selector_engine", _fake_provider
    )
    monkeypatch.setattr("natural_pdf.core.selector_utils._run_native_selector", _fake_native)

    page.find_all("text")

    assert provider_calls["count"] == 1
    assert native_calls["count"] == 0
