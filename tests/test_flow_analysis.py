from typing import Any, Dict, Optional

from pydantic import Field, create_model

import natural_pdf as npdf
from natural_pdf.extraction.result import StructuredDataResult
from natural_pdf.flows.flow import Flow


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


def _make_answer_result(answer: str) -> StructuredDataResult:
    schema = create_model("_QA", answer=(Optional[str], Field(None)))
    instance = schema(answer=answer)
    return StructuredDataResult(
        data=instance,
        success=True,
        error_message=None,
        raw_output=None,
        model_used="test",
    )


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


def test_flow_ask_returns_structured_data_result(monkeypatch):
    pdf, flow = _make_flow()

    fake_result = _make_answer_result("flow")

    # Patch the constituent region's extract method
    region = flow._analysis_region().constituent_regions[0]
    monkeypatch.setattr(region, "extract", lambda **kw: fake_result)

    try:
        result = flow.ask("What?", min_confidence=0.0)
    finally:
        pdf.close()

    assert isinstance(result, StructuredDataResult)
    assert result.answer == "flow"
