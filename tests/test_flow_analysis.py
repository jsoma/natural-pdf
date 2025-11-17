from typing import Any, Dict

import natural_pdf as npdf
from natural_pdf.flows.flow import Flow
from natural_pdf.qa.qa_result import QAResult


class _StubAnalysisRegion:
    def __init__(self):
        self.calls = []

    def apply_ocr(self, *args, **kwargs):
        self.calls.append(("apply_ocr", args, kwargs))

    def extract_ocr_elements(self, *args, **kwargs):
        self.calls.append(("extract_ocr_elements", args, kwargs))
        return ["ocr"]

    def remove_ocr_elements(self):
        self.calls.append(("remove_ocr_elements", (), {}))
        return 3

    def clear_text_layer(self):
        self.calls.append(("clear_text_layer", (), {}))
        return (1, 2)

    def create_text_elements_from_ocr(self, *args, **kwargs):
        self.calls.append(("create_text_elements_from_ocr", args, kwargs))
        return ["text"]


def _make_flow():
    pdf = npdf.PDF("pdfs/01-practice.pdf")
    flow = Flow([pdf.pages[0]], arrangement="vertical")
    return pdf, flow


def test_flow_analysis_methods_delegate(monkeypatch):
    pdf, flow = _make_flow()
    stub = _StubAnalysisRegion()
    monkeypatch.setattr(flow, "_analysis_region", lambda: stub)

    try:
        assert flow.apply_ocr(engine="stub") is flow
        assert flow.extract_ocr_elements() == ["ocr"]
        assert flow.remove_ocr_elements() == 3
        assert flow.clear_text_layer() == (1, 2)
        assert flow.create_text_elements_from_ocr("results") == ["text"]
    finally:
        pdf.close()

    recorded = [call[0] for call in stub.calls]
    assert recorded == [
        "apply_ocr",
        "extract_ocr_elements",
        "remove_ocr_elements",
        "clear_text_layer",
        "create_text_elements_from_ocr",
    ]


def test_flow_ask_uses_qa_service(monkeypatch):
    pdf, flow = _make_flow()
    capture: Dict[str, Any] = {}

    def fake_run_document_qa(*, context, region, question, **kwargs):
        capture["region"] = region
        capture["question"] = question
        return QAResult(question=question, answer="flow", confidence=0.42, found=True)

    monkeypatch.setattr(
        "natural_pdf.services.qa_service.run_document_qa",
        fake_run_document_qa,
    )

    try:
        result = flow.ask("What?", min_confidence=0.0)
    finally:
        pdf.close()

    assert result["answer"] == "flow"
    assert result["found"] is True
    assert capture["region"] is flow._analysis_region().constituent_regions[0]
