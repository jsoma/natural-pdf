from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

from natural_pdf.core.context import PDFContext
from natural_pdf.core.page import Page
from natural_pdf.core.page_collection import PageCollection
from natural_pdf.core.pdf import PDF
from natural_pdf.core.pdf_collection import PDFCollection
from natural_pdf.ocr.unified_dispatch import EngineEntry, _run_classic


def test_temporary_text_settings_propagates_body_exception_without_changes():
    page = Page.__new__(Page)
    page._config = {"x_tolerance": 3}
    page._element_mgr = MagicMock()

    with pytest.raises(RuntimeError, match="body failed"):
        with page._temporary_text_settings(text_tolerance={"x_tolerance": 3}) as changed:
            assert changed is False
            raise RuntimeError("body failed")

    page._element_mgr.invalidate_cache.assert_not_called()


def test_empty_pdf_collection_is_fully_initialized_with_requested_context():
    context = PDFContext.with_defaults()
    collection = PDFCollection([], recursive=False, context=context, password="secret")

    assert len(collection) == 0
    assert list(collection) == []
    assert collection._iter_index == 0
    assert collection._recursive is False
    assert collection._pdf_options == {"context": context, "password": "secret"}
    assert collection._context is context
    assert collection.services._context is context


def test_pdf_collection_slice_preserves_protocol_context_services_and_options():
    context = PDFContext.with_defaults()
    first = PDF.__new__(PDF)
    first._context = context
    second = PDF.__new__(PDF)
    second._context = PDFContext.with_defaults()
    collection = PDFCollection([first, second], recursive=False, password="secret")

    sliced = collection[1:]

    assert len(sliced) == 1
    assert list(sliced) == [second]
    assert sliced[0] is second
    assert sliced._iter_index == 0
    assert sliced._recursive is False
    assert sliced._pdf_options == {"password": "secret"}
    assert sliced._context is context
    assert sliced.services._context is context


def test_page_collection_extract_text_forwards_the_aggregate_contract():
    page = MagicMock()
    page.extract_text.return_value = "text"
    pages = PageCollection([page], context=PDFContext.with_defaults())

    assert (
        pages.extract_text(
            apply_exclusions=False,
            newlines=False,
            whitespace="normalize",
            strip=False,
        )
        == "text"
    )
    page.extract_text.assert_called_once_with(
        layout=False,
        apply_exclusions=False,
        newlines=False,
        whitespace="normalize",
        strip=False,
        bidi=True,
        content_filter=None,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"preserve_whitespace": False},
        {"keep_blank_chars": False},
        {"use_exclusions": False},
        {"strip_final": False},
        {"strip_empty": False},
    ],
)
def test_page_collection_extract_text_rejects_removed_aliases(kwargs):
    pages = PageCollection([], context=PDFContext.with_defaults())

    with pytest.raises(TypeError):
        pages.extract_text(**kwargs)


def test_pdf_apply_ocr_forwards_supported_page_controls(monkeypatch):
    page = SimpleNamespace(_execute_ocr_request=MagicMock())
    pdf = PDF.__new__(PDF)
    pdf._closed = False
    pdf._pdf = object()
    pdf._config = {"resolution": 72}
    pdf._get_target_pages = MagicMock(return_value=[page])
    monkeypatch.setattr("natural_pdf.core.pdf.normalize_ocr_options", lambda value, **kwargs: value)
    monkeypatch.setattr("natural_pdf.core.pdf.resolve_ocr_engine_name", lambda **kwargs: "vlm")
    monkeypatch.setattr(
        "natural_pdf.core.pdf.resolve_ocr_languages", lambda *args, **kwargs: ["fr", "en"]
    )
    monkeypatch.setattr("natural_pdf.core.pdf.resolve_ocr_min_confidence", lambda *a, **k: 0.5)
    monkeypatch.setattr("natural_pdf.core.pdf.resolve_ocr_device", lambda *a, **k: "cpu")
    client = object()

    result = PDF.apply_ocr(
        pdf,
        model="model",
        client=client,
        instructions="instructions",
        prompt="prompt",
        max_new_tokens=123,
        layout="rapidocr",
        preserve_markup=True,
        show_progress=False,
    )

    assert result is pdf
    request = page._execute_ocr_request.call_args.args[0]
    assert request.engine == "vlm"
    assert request.options is None
    assert request.languages == ("fr", "en")
    assert request.min_confidence == 0.5
    assert request.device == "cpu"
    assert request.resolution == 72
    assert request.apply_exclusions is True
    assert request.replace == "ocr"
    assert request.model == "model"
    assert request.client is client
    assert request.instructions == "instructions"
    assert request.prompt == "prompt"
    assert request.max_new_tokens == 123
    assert request.layout == "rapidocr"
    assert request.preserve_markup is True


def test_pdf_collection_apply_ocr_forwards_supported_controls():
    class StubPDF:
        path = "mock.pdf"

        def __init__(self):
            self.prepared = []

        def _prepare_pdf_ocr_request(self, request, *, pages):
            self.prepared.append((request, pages))
            return request, []

        def _execute_prepared_pdf_ocr_request(self, request, targets, *, show_progress):
            return None

    pdf = StubPDF()
    collection = PDFCollection.__new__(PDFCollection)
    collection._pdfs = [pdf]
    client = object()

    result = collection.apply_ocr(
        model="model",
        client=client,
        instructions="instructions",
        prompt="prompt",
        max_new_tokens=123,
        layout=True,
        preserve_markup=True,
        show_progress=False,
    )

    assert result is collection
    request, pages = pdf.prepared[0]
    assert pages is None
    assert request.mode == "recognition"
    assert request.model == "model"
    assert request.client is client
    assert request.instructions == "instructions"
    assert request.prompt == "prompt"
    assert request.max_new_tokens == 123
    assert request.layout is True
    assert request.preserve_markup is True


def test_pdf_add_exclusion_rejects_invalid_method_immediately():
    pdf = SimpleNamespace(_pages=[], _exclusions=[])

    with pytest.raises(ValueError, match="Exclusion method must be 'region' or 'element'"):
        PDF.add_exclusion(pdf, lambda page: None, method="invalid")

    assert pdf._exclusions == []


def test_classic_dispatch_preserves_language_order_in_cache_and_engine(monkeypatch):
    seen = {}

    class SpyEngine:
        def _initialize_model(self, languages, device, options):
            seen["initialized"] = languages

        def process_image(self, image, *, languages, **kwargs):
            seen["processed"] = languages
            return []

    class SpyCache:
        def get_or_create(
            self,
            *,
            engine_name,
            languages,
            device,
            init_key,
            factory,
            provider_identity=None,
        ):
            seen["cache_key"] = languages
            seen["provider_identity"] = provider_identity
            return factory()

    monkeypatch.setattr("natural_pdf.ocr.unified_dispatch._engine_cache", SpyCache())

    _run_classic(
        image=Image.new("RGB", (10, 10)),
        engine_name="spy",
        entry=EngineEntry(engine_type="classic", provider=SpyEngine),
        languages=["fr", "en"],
        min_confidence=None,
        device="cpu",
        detect_only=False,
        options=None,
    )

    assert seen == {
        "cache_key": ("fr", "en"),
        "provider_identity": id(SpyEngine),
        "initialized": ["fr", "en"],
        "processed": ["fr", "en"],
    }
