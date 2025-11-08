"""Integration tests for table extraction using real PDFs."""

from __future__ import annotations

from contextlib import contextmanager

import natural_pdf as npdf
from natural_pdf.flows.flow import Flow
from natural_pdf.tables import TableResult


@contextmanager
def _load_pdf(path: str = "pdfs/01-practice.pdf"):
    pdf = npdf.PDF(path)
    try:
        yield pdf
    finally:
        pdf.close()


def test_page_extract_table_real_pdf():
    with _load_pdf() as pdf:
        page = pdf.pages[0]
        single_table = page.extract_table(
            method="stream", table_settings={"vertical_strategy": "text"}
        )
        assert isinstance(single_table, TableResult)

        tables = page.extract_tables(method="lattice")
        assert isinstance(tables, list)


def test_flow_extract_table_real_pdf():
    with _load_pdf() as pdf:
        flow = Flow([pdf.pages[0]], arrangement="vertical", alignment="start")
        table = flow.extract_table(method="pdfplumber")
        assert isinstance(table, TableResult)

        tables = flow.extract_tables(method="pdfplumber")
        assert isinstance(tables, list)
