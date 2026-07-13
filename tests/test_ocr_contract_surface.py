"""Focused checks for the public OCR contract.

These tests intentionally exercise the API surface rather than an OCR backend;
the backend-free fake keeps failures attributable to contract wiring.
"""

import inspect
from typing import Any

import pytest
from typing_extensions import get_overloads

from natural_pdf.analyzers.guides.base import Guides
from natural_pdf.core.ocr_mixin import OCRScopeMixin, PDFCollectionOCRMixin, PDFOCRMixin
from natural_pdf.core.page import Page
from natural_pdf.core.page_collection import PageCollection
from natural_pdf.core.pdf import PDF
from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.elements.region import Region
from natural_pdf.flows.flow import Flow
from natural_pdf.flows.region import FlowRegion


@pytest.mark.parametrize(
    "host", [Page, Region, Flow, FlowRegion, PageCollection, ElementCollection]
)
def test_spatial_hosts_share_the_ocr_method(host: type[Any]) -> None:
    """All direct/aggregate spatial hosts expose one canonical implementation."""

    assert host.apply_ocr is OCRScopeMixin.apply_ocr


def test_ocr_signature_is_keyword_strict_and_has_engine_only_positional() -> None:
    signature = inspect.signature(OCRScopeMixin.apply_ocr)
    parameters = list(signature.parameters.values())
    assert parameters[1].name == "engine"
    assert parameters[1].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert all(parameter.kind is not inspect.Parameter.VAR_KEYWORD for parameter in parameters)
    assert signature.parameters["function"].kind is inspect.Parameter.KEYWORD_ONLY
    assert "ocr_function" not in signature.parameters


def test_pdf_contract_exposes_page_selection_progress_and_workers() -> None:
    pdf_signature = inspect.signature(PDFOCRMixin.apply_ocr)
    collection_signature = inspect.signature(PDFCollectionOCRMixin.apply_ocr)
    assert {"pages", "show_progress"} <= pdf_signature.parameters.keys()
    assert {"pages", "show_progress", "max_workers"} <= collection_signature.parameters.keys()
    assert all(
        p.kind is not inspect.Parameter.VAR_KEYWORD for p in pdf_signature.parameters.values()
    )
    assert all(
        p.kind is not inspect.Parameter.VAR_KEYWORD
        for p in collection_signature.parameters.values()
    )


@pytest.mark.parametrize(
    ("mixin", "batch_controls"),
    [
        (PDFOCRMixin, {"pages", "show_progress"}),
        (PDFCollectionOCRMixin, {"pages", "max_workers", "show_progress"}),
    ],
)
def test_pdf_ocr_contract_has_three_mode_specific_overloads(
    mixin: type[Any], batch_controls: set[str]
) -> None:
    """PDF surfaces retain the same mode help as direct OCR targets."""

    overloads = get_overloads(mixin.apply_ocr)

    assert len(overloads) == 3
    for overload in overloads:
        signature = inspect.signature(overload)
        assert batch_controls <= signature.parameters.keys()
        assert signature.parameters["engine"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD

    recognition, detection, function = (inspect.signature(overload) for overload in overloads)
    assert recognition.parameters["detect_only"].default is False
    assert detection.parameters["detect_only"].default is inspect.Parameter.empty
    assert function.parameters["function"].default is inspect.Parameter.empty
    assert "This method has three validated modes" in mixin.apply_ocr.__doc__


class _FakeScope(OCRScopeMixin):
    def _iter_ocr_hosts(self, request):
        return ()


def test_contract_returns_concrete_self_and_rejects_legacy_keyword() -> None:
    fake = _FakeScope()
    assert fake.apply_ocr(detect_only=True) is fake
    with pytest.raises(TypeError):
        fake.apply_ocr(ocr_function=lambda region: "text")


def test_guides_ocr_is_scoped_to_views_not_guides_itself() -> None:
    assert not hasattr(Guides, "apply_ocr")
    assert hasattr(Guides, "cells")
    assert hasattr(Guides, "rows")
    assert hasattr(Guides, "columns")
    # The view contract owns both planning and execution.
    guide_stub = object.__new__(Guides)
    cells = Guides.cells.__get__(guide_stub, Guides)
    rows = Guides.rows.__get__(guide_stub, Guides)
    columns = Guides.columns.__get__(guide_stub, Guides)
    assert hasattr(cells, "apply_ocr")
    assert hasattr(cells, "plan_ocr")
    assert hasattr(rows, "apply_ocr")
    assert hasattr(columns, "apply_ocr")
