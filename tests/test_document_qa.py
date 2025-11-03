"""Minimal coverage for Document QA public API."""

from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.qa, pytest.mark.qa_remote, pytest.mark.slow]


def _require_qa():
    """Skip if qa extras are unavailable."""

    pytest.importorskip("natural_pdf.qa")


class _DummyCompletions:
    def __init__(self, answer: str):
        self.answer = answer
        self.last_kwargs = None

    def parse(self, *args, **kwargs):
        self.last_kwargs = {"args": args, "kwargs": kwargs}
        response_format = kwargs["response_format"]
        parsed = response_format(answer=self.answer)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))])


class DummyLLMClient:
    def __init__(self, answer: str = "stub answer"):
        completions = _DummyCompletions(answer=answer)
        self.beta = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        self.completions = completions


def _assert_result_schema(result: dict) -> None:
    assert isinstance(result, dict)
    assert set(result).issuperset({"answer", "confidence", "found"})
    assert isinstance(result["confidence"], float)
    assert isinstance(result["found"], bool)


def test_qa_dependencies_available():
    _require_qa()


def test_pdf_ask_and_batch_smoke(practice_pdf):
    _require_qa()

    single = practice_pdf.ask("What is the total amount?", pages=0)
    _assert_result_schema(single)

    batch = practice_pdf.ask_batch(
        [
            "What is the total amount?",
            "What type of document is this?",
        ],
        pages=0,
    )
    assert len(batch) == 2
    for result in batch:
        _assert_result_schema(result)


def test_pdf_ask_invalid_inputs(practice_pdf):
    _require_qa()

    with pytest.raises(TypeError):
        practice_pdf.ask(123)
    with pytest.raises(ValueError):
        practice_pdf.ask("Too wide", pages="invalid")
    with pytest.raises(TypeError):
        practice_pdf.ask_batch("not a list")


def test_pdf_page_ask(practice_pdf):
    _require_qa()
    page = practice_pdf.pages[0]
    result = page.ask("What is shown on this page?")
    if isinstance(result, list):
        assert result  # page.ask still supports batches
        result = result[0]
    _assert_result_schema(result)


def test_pdf_ask_generative_mode(practice_pdf):
    _require_qa()
    client = DummyLLMClient(answer="Generated summary")
    result = practice_pdf.ask(
        "Provide a short summary.",
        mode="generative",
        llm_client=client,
    )
    assert result["answer"] == "Generated summary"
    assert result["confidence"] is None


def test_pdf_ask_with_ocr_page(practice_pdf):
    _require_qa()
    try:
        practice_pdf.apply_ocr(engine="easyocr", pages=[0])
    except RuntimeError as exc:
        pytest.skip(f"EasyOCR unavailable: {exc}")

    result = practice_pdf.ask("Is OCR text available?", pages=0)
    _assert_result_schema(result)
