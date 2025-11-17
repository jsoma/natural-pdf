from __future__ import annotations

from typing import List

from natural_pdf.core.pdf import PDF
from natural_pdf.core.pdf_collection import PDFCollection
from natural_pdf.qa.qa_result import QAResult


def _fake_result(question: str, page_number: int) -> QAResult:
    return QAResult(
        {
            "question": question,
            "answer": f"page-{page_number}",
            "confidence": float(page_number),
            "found": True,
            "page_num": page_number,
            "source_elements": [],
        }
    )


def test_pdf_ask_extractive_uses_service(monkeypatch):
    pdf = PDF("pdfs/Atlanta_Public_Schools_GA_sample.pdf")
    calls: List[int] = []

    def fake_run_document_qa(*, region, question, **kwargs):
        page_number = getattr(region.page, "number", None)
        calls.append(page_number)
        return _fake_result(question, page_number)

    monkeypatch.setattr(
        "natural_pdf.services.qa_service.run_document_qa",
        fake_run_document_qa,
    )

    try:
        result = pdf.ask("Which page?", pages=0)
    finally:
        pdf.close()

    assert result["answer"] == "page-1"
    assert calls == [1]


def test_pdf_ask_batch_extractive(monkeypatch):
    pdf = PDF("pdfs/Atlanta_Public_Schools_GA_sample.pdf")
    calls: List[int] = []

    def fake_run_document_qa(*, region, question, **kwargs):
        page_number = getattr(region.page, "number", None)
        calls.append(page_number)
        return _fake_result(question, page_number)

    monkeypatch.setattr(
        "natural_pdf.services.qa_service.run_document_qa",
        fake_run_document_qa,
    )

    try:
        results = pdf.ask_batch(["first?", "second?"], pages=[0, 1])
    finally:
        pdf.close()

    assert len(results) == 2
    assert all(res["found"] for res in results)
    # Two questions across two pages => four invocations
    assert calls == [1, 2, 1, 2]


def test_pdf_collection_ask(monkeypatch):
    pdf1 = PDF("pdfs/01-practice.pdf")
    pdf2 = PDF("pdfs/Atlanta_Public_Schools_GA_sample.pdf")
    collection = PDFCollection([pdf1, pdf2])

    calls: List[int] = []

    def fake_run_document_qa(*, region, question, **kwargs):
        page_number = getattr(region.page, "number", None) or 0
        calls.append(page_number)
        return _fake_result(question, page_number)

    monkeypatch.setattr(
        "natural_pdf.services.qa_service.run_document_qa",
        fake_run_document_qa,
    )

    try:
        result = collection.ask("Where?", min_confidence=0.0)
    finally:
        pdf1.close()
        pdf2.close()

    assert result["found"] is True
    total_pages = sum(len(pdf.pages) for pdf in (pdf1, pdf2))
    assert len(calls) == total_pages
