from __future__ import annotations

import natural_pdf as npdf
from natural_pdf.core.page_collection import PageCollection
from natural_pdf.qa.qa_result import QAResult


def test_page_collection_ask_uses_qa_service(monkeypatch):
    pdf = npdf.PDF("pdfs/Atlanta_Public_Schools_GA_sample.pdf")
    try:
        pages = PageCollection([pdf.pages[0], pdf.pages[1]])

        calls = []

        def fake_run_document_qa(*, context, region, question, **kwargs):
            page_num = getattr(region.page, "number", None)
            calls.append(page_num)
            return QAResult(
                question=question,
                answer=f"page-{page_num}",
                confidence=float(page_num or 0),
                found=True,
            )

        monkeypatch.setattr(
            "natural_pdf.services.qa_service.run_document_qa",
            fake_run_document_qa,
        )

        result = pages.ask("Which page?", min_confidence=0.0)
    finally:
        pdf.close()

    assert result["answer"] == "page-2"
    assert calls == [1, 2]
