"""Tests for the provider-based table extraction flow."""

from __future__ import annotations

import natural_pdf as npdf
import natural_pdf.engine_provider as provider_module
from natural_pdf.engine_provider import EngineProvider
from natural_pdf.tables import TableResult
from natural_pdf.tables.table_provider import PdfPlumberTablesEngine, normalize_table_settings


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

    class DummyRegion:
        bbox = (0, 0, 1, 1)

        def _extract_tables_plumber(self, settings):
            calls.append(settings.copy())
            if len(calls) == 1:
                return [[[""]]]  # No meaningful content
            return [[["data"]]]

    engine = PdfPlumberTablesEngine("auto")
    tables = engine.extract_tables(context=None, region=DummyRegion(), table_settings={"snap": 1})

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
