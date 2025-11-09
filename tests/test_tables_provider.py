"""Tests for the provider-based table extraction flow."""

from __future__ import annotations

import natural_pdf as npdf
import natural_pdf.engine_provider as provider_module
from natural_pdf.engine_provider import EngineProvider
from natural_pdf.tables import TableResult
from natural_pdf.tables.engines import pdfplumber as pdfplumber_mod
from natural_pdf.tables.table_provider import PdfPlumberTablesEngine, normalize_table_settings
from natural_pdf.tables.utils import plumber as plumber_utils


def test_region_extract_tables_delegates_to_provider(monkeypatch):
    provider = EngineProvider()
    provider._entry_points_loaded = True
    monkeypatch.setattr(provider_module, "_PROVIDER", provider)

    class _StubEngine:
        def extract_tables(self, *, context, region, table_settings=None, **kwargs):
            assert region is context
            assert table_settings == {"foo": "bar"}
            return [[["provider"]]]

    provider.register("tables", "pdfplumber_auto", lambda **_: _StubEngine(), replace=True)

    pdf = npdf.PDF("pdfs/01-practice.pdf")
    region = pdf.pages[0].to_region()
    tables = region.extract_tables(table_settings={"foo": "bar"})
    assert tables == [[["provider"]]]
    pdf.close()


def test_pdfplumber_auto_engine_falls_back_to_stream(monkeypatch):
    calls = []

    def fake_extract(region, table_settings, apply_exclusions):
        calls.append(table_settings.copy())
        if table_settings.get("vertical_strategy") == "lines":
            return [[[""]]]  # No meaningful content
        return [[["data"]]]

    monkeypatch.setattr(pdfplumber_mod, "extract_tables_plumber", fake_extract, raising=False)

    engine = PdfPlumberTablesEngine("auto")
    tables = engine.extract_tables(context=None, region=object(), table_settings={"snap": 1})

    assert len(calls) == 2, "Auto engine should attempt lattice then stream"
    assert tables == [[["data"]]]
    assert calls[0]["vertical_strategy"] == "lines"
    assert calls[1]["vertical_strategy"] == "text"


def test_normalize_table_settings_returns_copy():
    original = {"vertical_strategy": "text"}
    normalized = normalize_table_settings(original)
    assert normalized == original
    assert normalized is not original


def test_page_extract_table_delegates_to_provider(monkeypatch):
    provider = EngineProvider()
    provider._entry_points_loaded = True
    monkeypatch.setattr(provider_module, "_PROVIDER", provider)

    class _StubEngine:
        def extract_tables(self, *, context, region, table_settings=None, **kwargs):
            return [
                [["short"]],
                [["r1c1", "r1c2"], ["r2c1", "r2c2"]],
                [["x"]],
            ]

    provider.register("tables", "stream", lambda **_: _StubEngine(), replace=True)

    pdf = npdf.PDF("pdfs/01-practice.pdf")
    table = pdf.pages[0].extract_table(method="stream", table_settings={"foo": "bar"})
    assert isinstance(table, TableResult)
    assert list(table) == [["r1c1", "r1c2"], ["r2c1", "r2c2"]]
    pdf.close()


def test_region_text_method_routes_through_provider(monkeypatch):
    provider = EngineProvider()
    provider._entry_points_loaded = True
    monkeypatch.setattr(provider_module, "_PROVIDER", provider)

    sentry = object()

    class _StubTextEngine:
        def extract_tables(self, *, context, region, cell_extraction_func=None, **kwargs):
            assert context is region
            assert cell_extraction_func is sentry
            assert kwargs["text_options"] == {"foo": "bar"}
            return [[["text"]]]

    provider.register("tables", "text", lambda **_: _StubTextEngine(), replace=True)

    pdf = npdf.PDF("pdfs/01-practice.pdf")
    region = pdf.pages[0].to_region()
    table = region.extract_table(
        method="text",
        text_options={"foo": "bar"},
        cell_extraction_func=sentry,
    )
    assert isinstance(table, TableResult)
    assert list(table) == [["text"]]
    pdf.close()


def test_region_tatr_method_routes_through_provider(monkeypatch):
    provider = EngineProvider()
    provider._entry_points_loaded = True
    monkeypatch.setattr(provider_module, "_PROVIDER", provider)

    class _StubTATREngine:
        def extract_tables(self, *, use_ocr=False, **kwargs):
            assert use_ocr is True
            return [[["tatr"]]]

    provider.register("tables", "tatr", lambda **_: _StubTATREngine(), replace=True)

    pdf = npdf.PDF("pdfs/01-practice.pdf")
    region = pdf.pages[0].to_region()
    table = region.extract_table(method="tatr", use_ocr=True)
    assert isinstance(table, TableResult)
    assert list(table) == [["tatr"]]
    pdf.close()
