from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from natural_pdf.core.context import PDFContext

if TYPE_CHECKING:
    from natural_pdf.services.checkbox_service import CheckboxDetectionService
    from natural_pdf.services.classification_service import ClassificationService
    from natural_pdf.services.describe_service import DescribeService
    from natural_pdf.services.exclusion_service import ExclusionService
    from natural_pdf.services.extraction_service import ExtractionService
    from natural_pdf.services.guides_service import GuidesService
    from natural_pdf.services.layout_service import LayoutService
    from natural_pdf.services.navigation_service import NavigationService
    from natural_pdf.services.ocr_service import OCRService
    from natural_pdf.services.qa_service import QAService
    from natural_pdf.services.selector_service import SelectorService
    from natural_pdf.services.shape_detection_service import ShapeDetectionService
    from natural_pdf.services.table_service import TableService
    from natural_pdf.services.text_service import TextService
    from natural_pdf.services.vision_service import VisualSearchService


class ServiceNamespace:
    """Typed namespace for accessing services."""

    def __init__(self, context: PDFContext):
        self._context = context

    @property
    def checkbox(self) -> CheckboxDetectionService:
        return self._context.get_service("checkbox")

    @property
    def classification(self) -> ClassificationService:
        return self._context.get_service("classification")

    @property
    def describe(self) -> DescribeService:
        return self._context.get_service("describe")

    @property
    def exclusion(self) -> ExclusionService:
        return self._context.get_service("exclusion")

    @property
    def extraction(self) -> ExtractionService:
        return self._context.get_service("extraction")

    @property
    def guides(self) -> GuidesService:
        return self._context.get_service("guides")

    @property
    def layout(self) -> LayoutService:
        return self._context.get_service("layout")

    @property
    def navigation(self) -> NavigationService:
        return self._context.get_service("navigation")

    @property
    def ocr(self) -> OCRService:
        return self._context.get_service("ocr")

    @property
    def qa(self) -> QAService:
        return self._context.get_service("qa")

    @property
    def selector(self) -> SelectorService:
        return self._context.get_service("selector")

    @property
    def shapes(self) -> ShapeDetectionService:
        return self._context.get_service("shapes")

    @property
    def table(self) -> TableService:
        return self._context.get_service("table")

    @property
    def text(self) -> TextService:
        return self._context.get_service("text")

    @property
    def vision(self) -> VisualSearchService:
        return self._context.get_service("vision")


class ServiceHostMixin:
    """Provides helpers for objects that access services via PDFContext."""

    _context: PDFContext
    services: ServiceNamespace

    def _init_service_host(self, context: PDFContext) -> None:
        self._context = context
        self.services = ServiceNamespace(context)

    def _get_service(self, capability: str) -> Any:
        # Deprecated: Use self.services.[capability] instead
        return self._context.get_service(capability)


def resolve_service(host: Any, capability: str) -> Any:
    """Return a service for hosts that may or may not inherit ServiceHostMixin."""

    if hasattr(host, "services") and isinstance(host.services, ServiceNamespace):
        return getattr(host.services, capability)

    attrs = getattr(host, "__dict__", {})
    if "_context" in attrs:
        # Fallback for hosts that have context but not the new namespace yet
        return host._context.get_service(capability)

    context = getattr(resolve_service, "_fallback_context", None)
    if context is None:
        context = PDFContext.with_defaults()
        setattr(resolve_service, "_fallback_context", context)
    return context.get_service(capability)
