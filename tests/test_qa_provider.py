from __future__ import annotations

import natural_pdf as npdf
import natural_pdf.engine_provider as provider_module
from natural_pdf.engine_provider import EngineProvider
from natural_pdf.qa.qa_result import QAResult


def test_page_ask_uses_provider(monkeypatch):
    provider = EngineProvider()
    provider._entry_points_loaded = True
    monkeypatch.setattr(provider_module, "_PROVIDER", provider)

    class _StubQAEngine:
        def ask_region(self, *, question, **kwargs):
            if isinstance(question, (list, tuple)):
                return [
                    QAResult(question=q, answer="A", confidence=0.9, found=True) for q in question
                ]
            return QAResult(question=question, answer="A", confidence=0.9, found=True)

    provider.register("qa.document", "layoutlm", lambda **_: _StubQAEngine(), replace=True)

    pdf = npdf.PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    result = page.ask("What?", min_confidence=0.1)
    assert isinstance(result, QAResult)
    assert result.answer == "A"
    pdf.close()
