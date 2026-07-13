import inspect

import pytest

from natural_pdf.core.ocr_mixin import PDFCollectionOCRMixin, PDFOCRMixin
from natural_pdf.core.pdf import PDF
from natural_pdf.core.pdf_collection import PDFCollection


class _Page:
    def __init__(self, region):
        self.region = region
        self.requests = []

    def _execute_ocr_request(self, request):
        self.requests.append(request)
        if hasattr(request, "function"):
            request.function(self.region)


def _pdf_with_pages(pages):
    pdf = PDF.__new__(PDF)
    pdf._config = {"resolution": 72}
    pdf._closed = False
    pdf._pdf = object()
    pdf._get_target_pages = lambda selection: (
        list(pages) if selection is None else [pages[index] for index in selection]
    )
    return pdf


def _stub_pdf(monkeypatch, pages):
    pdf = _pdf_with_pages(pages)
    monkeypatch.setattr("natural_pdf.core.pdf.normalize_ocr_options", lambda value, **kwargs: value)
    monkeypatch.setattr(
        "natural_pdf.core.pdf.resolve_ocr_engine_name", lambda **kwargs: "resolved-engine"
    )
    monkeypatch.setattr(
        "natural_pdf.core.pdf.resolve_ocr_languages", lambda *args, **kwargs: ["fr", "en"]
    )
    monkeypatch.setattr(
        "natural_pdf.core.pdf.resolve_ocr_min_confidence", lambda *args, **kwargs: 0.6
    )
    monkeypatch.setattr("natural_pdf.core.pdf.resolve_ocr_device", lambda *args, **kwargs: "cpu")
    return pdf


def test_pdf_request_forwards_resolved_controls_to_selected_pages(monkeypatch):
    pages = [_Page("first"), _Page("second")]
    pdf = _stub_pdf(monkeypatch, pages)

    assert (
        pdf.apply_ocr(
            "requested-engine",
            options={"mode": "fast"},
            languages=["de"],
            min_confidence=0.1,
            device="cuda",
            resolution=300,
            replace="none",
            use_cache=False,
            pages=[1],
            show_progress=False,
        )
        is pdf
    )

    assert pages[0].requests == []
    request = pages[1].requests[0]
    assert request.engine == "resolved-engine"
    assert request.options == {"mode": "fast"}
    assert request.languages == ("fr", "en")
    assert request.min_confidence == 0.6
    assert request.device == "cpu"
    assert request.resolution == 300
    assert request.replace == "none"
    assert request.use_cache is False


def test_pdf_progress_can_be_suppressed_and_function_runs_once_per_selected_page(monkeypatch):
    pages = [_Page("first"), _Page("second"), _Page("third")]
    pdf = _stub_pdf(monkeypatch, pages)
    calls = []
    monkeypatch.setattr(
        "natural_pdf.core.pdf.tqdm",
        lambda *args, **kwargs: pytest.fail("progress must not be created"),
    )

    assert (
        pdf.apply_ocr(
            function=lambda region: calls.append(region),
            pages=[0, 2],
            show_progress=False,
        )
        is pdf
    )

    assert calls == ["first", "third"]


def test_pdf_model_or_client_selects_vlm_before_pdf_default_resolution(monkeypatch):
    page = _Page("first")
    pdf = _pdf_with_pages([page])
    requested_engines = []
    monkeypatch.setattr("natural_pdf.core.pdf.normalize_ocr_options", lambda value, **kwargs: value)

    def resolve_engine(**kwargs):
        requested_engines.append(kwargs["requested"])
        return kwargs["requested"]

    monkeypatch.setattr("natural_pdf.core.pdf.resolve_ocr_engine_name", resolve_engine)
    monkeypatch.setattr(
        "natural_pdf.core.pdf.resolve_ocr_languages", lambda *args, **kwargs: ["en"]
    )
    monkeypatch.setattr(
        "natural_pdf.core.pdf.resolve_ocr_min_confidence", lambda *args, **kwargs: None
    )
    monkeypatch.setattr("natural_pdf.core.pdf.resolve_ocr_device", lambda *args, **kwargs: "cpu")

    client = object()
    pdf.apply_ocr(model="vision-model", client=client, show_progress=False)

    assert requested_engines == ["vlm"]
    assert page.requests[0].engine == "vlm"
    assert page.requests[0].model == "vision-model"
    assert page.requests[0].client is client


def test_pdf_rejects_engine_invalid_options_before_page_execution(monkeypatch):
    page = _Page("first")
    pdf = _pdf_with_pages([page])
    monkeypatch.setattr("natural_pdf.core.pdf.resolve_ocr_engine_name", lambda **kwargs: "rapidocr")
    monkeypatch.setattr(
        "natural_pdf.core.pdf.resolve_ocr_languages", lambda *args, **kwargs: ["en"]
    )
    monkeypatch.setattr(
        "natural_pdf.core.pdf.resolve_ocr_min_confidence", lambda *args, **kwargs: None
    )
    monkeypatch.setattr("natural_pdf.core.pdf.resolve_ocr_device", lambda *args, **kwargs: "cpu")

    with pytest.raises(TypeError, match="Unsupported option.*rapidocr"):
        pdf.apply_ocr(
            "rapidocr",
            options={"definitely_invalid": True},
            show_progress=False,
        )

    assert page.requests == []


class _CollectionPDF:
    def __init__(self, path, targets):
        self.path = path
        self.targets = targets
        self.prepared = []
        self.executed = []

    def _prepare_pdf_ocr_request(self, request, *, pages):
        self.prepared.append((request, pages))
        selected = self.targets if pages is None else [self.targets[index] for index in pages]
        return request, selected

    def _execute_prepared_pdf_ocr_request(self, request, targets, *, show_progress):
        self.executed.append((request, list(targets), show_progress))
        if hasattr(request, "function"):
            for target in targets:
                request.function(target)


def _collection(pdfs):
    collection = PDFCollection.__new__(PDFCollection)
    collection._pdfs = pdfs
    return collection


@pytest.mark.parametrize("max_workers", [None, 2])
def test_collection_returns_self_and_forwards_page_selection(max_workers):
    first = _CollectionPDF("first.pdf", ["first-0", "first-1"])
    second = _CollectionPDF("second.pdf", ["second-0", "second-1"])
    collection = _collection([first, second])
    calls = []

    assert (
        collection.apply_ocr(
            function=lambda region: calls.append(region),
            pages=[1],
            max_workers=max_workers,
            show_progress=False,
        )
        is collection
    )

    assert [prepared[1] for prepared in first.prepared + second.prepared] == [(1,), (1,)]
    assert sorted(calls) == ["first-1", "second-1"]
    assert all(executed[2] is False for pdf in (first, second) for executed in pdf.executed)


def test_collection_validates_all_pdfs_before_starting_workers(monkeypatch):
    collection = _collection([_CollectionPDF("first.pdf", []), _CollectionPDF("bad.pdf", [])])

    def reject_request(request, *, pages):
        raise ValueError("invalid page selection")

    collection._pdfs[1]._prepare_pdf_ocr_request = reject_request
    monkeypatch.setattr(
        "natural_pdf.core.pdf_collection.concurrent.futures.ThreadPoolExecutor",
        lambda *args, **kwargs: pytest.fail("workers must not start before validation"),
    )

    with pytest.raises(ValueError, match="invalid page selection"):
        collection.apply_ocr(max_workers=2, show_progress=False)


def test_collection_rejects_closed_cached_pdf_before_any_worker_or_mutation(monkeypatch):
    first_page = _Page("first")
    first = _stub_pdf(monkeypatch, [first_page])
    closed = _pdf_with_pages([_Page("closed")])
    closed._closed = True
    closed._pdf = None
    collection = _collection([first, closed])
    monkeypatch.setattr(
        "natural_pdf.core.pdf_collection.concurrent.futures.ThreadPoolExecutor",
        lambda *args, **kwargs: pytest.fail("workers must not start before closed-PDF validation"),
    )

    with pytest.raises(RuntimeError, match="PDF has been closed"):
        collection.apply_ocr(max_workers=2, show_progress=False)

    assert first_page.requests == []


def test_collection_progress_can_be_suppressed(monkeypatch):
    collection = _collection([_CollectionPDF("first.pdf", [])])
    monkeypatch.setattr(
        "natural_pdf.core.pdf_collection.tqdm",
        lambda *args, **kwargs: pytest.fail("progress must not be created"),
    )

    assert collection.apply_ocr(show_progress=False) is collection


def test_pdf_and_collection_ocr_signatures_belong_to_mixins():
    assert PDF.apply_ocr is PDFOCRMixin.apply_ocr
    assert PDFCollection.apply_ocr is PDFCollectionOCRMixin.apply_ocr

    for host in (PDF, PDFCollection):
        parameters = inspect.signature(host.apply_ocr).parameters
        assert parameters["engine"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert all(
            parameter.kind is inspect.Parameter.KEYWORD_ONLY
            for name, parameter in parameters.items()
            if name not in {"self", "engine"}
        )
        assert "ocr_function" not in parameters
        assert not any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        )
