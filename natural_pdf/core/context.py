from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional

ServiceFactory = Callable[["PDFContext"], Any]


@dataclass
class PDFContext:
    """Holds shared services/configuration for a PDF and its descendants."""

    service_factories: Optional[Mapping[str, ServiceFactory]] = None
    options: Optional[Mapping[str, Mapping[str, Any]]] = None
    _shared_services: MutableMapping[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        factories = dict(self.service_factories or {})
        if "text" not in factories:
            factories["text"] = self._default_text_factory()
        if "navigation" not in factories:
            factories["navigation"] = self._default_navigation_factory()
        if "ocr" not in factories:
            factories["ocr"] = self._default_ocr_factory()
        if "table" not in factories:
            factories["table"] = self._default_table_factory()
        if "extraction" not in factories:
            factories["extraction"] = self._default_extraction_factory()
        if "classification" not in factories:
            factories["classification"] = self._default_classification_factory()
        if "qa" not in factories:
            factories["qa"] = self._default_qa_factory()
        if "exclusion" not in factories:
            factories["exclusion"] = self._default_exclusion_factory()
        if "selector" not in factories:
            factories["selector"] = self._default_selector_factory()
        if "describe" not in factories:
            factories["describe"] = self._default_describe_factory()
        if "vision" not in factories:
            factories["vision"] = self._default_vision_factory()
        if "shapes" not in factories:
            factories["shapes"] = self._default_shape_factory()
        if "checkbox" not in factories:
            factories["checkbox"] = self._default_checkbox_factory()
        if "guides" not in factories:
            factories["guides"] = self._default_guides_factory()
        if "layout" not in factories:
            factories["layout"] = self._default_layout_factory()
        if "rendering" not in factories:
            factories["rendering"] = self._default_rendering_factory()
        self._service_factories: Dict[str, ServiceFactory] = factories
        self._options: Dict[str, Mapping[str, Any]] = dict(self.options or {})

    @classmethod
    def with_defaults(cls) -> "PDFContext":
        return cls()

    def get_service(self, capability: str) -> Any:
        if capability in self._shared_services:
            return self._shared_services[capability]
        if capability not in self._service_factories:
            raise KeyError(f"Unknown service capability '{capability}'")
        factory = self._service_factories[capability]
        service = factory(self)
        self._shared_services[capability] = service
        return service

    def get_option(
        self,
        capability: str,
        key: str,
        *,
        host: Optional[Any] = None,
        default: Any = None,
        scope: str = "region",
    ) -> Any:
        capability_options = self._options.get(capability)
        if capability_options and key in capability_options:
            return capability_options[key]
        if host is not None:
            getter = getattr(host, "get_config", None)
            if callable(getter):
                try:
                    value = getter(key, default, scope=scope)
                except TypeError:
                    value = getter(key, default)
                return value
        return default

    @staticmethod
    def _default_text_factory() -> ServiceFactory:
        from natural_pdf.services.text_service import TextService

        def factory(context: "PDFContext") -> TextService:
            return TextService(context)

        return factory

    @staticmethod
    def _default_navigation_factory() -> ServiceFactory:
        from natural_pdf.services.navigation_service import NavigationService

        def factory(context: "PDFContext") -> NavigationService:
            return NavigationService(context)

        return factory

    @staticmethod
    def _default_ocr_factory() -> ServiceFactory:
        from natural_pdf.services.ocr_service import OCRService

        def factory(context: "PDFContext") -> OCRService:
            return OCRService(context)

        return factory

    @staticmethod
    def _default_table_factory() -> ServiceFactory:
        from natural_pdf.services.table_service import TableService

        def factory(context: "PDFContext") -> TableService:
            return TableService(context)

        return factory

    @staticmethod
    def _default_extraction_factory() -> ServiceFactory:
        from natural_pdf.services.extraction_service import ExtractionService

        def factory(context: "PDFContext") -> ExtractionService:
            return ExtractionService(context)

        return factory

    @staticmethod
    def _default_classification_factory() -> ServiceFactory:
        from natural_pdf.services.classification_service import ClassificationService

        def factory(context: "PDFContext") -> ClassificationService:
            return ClassificationService(context)

        return factory

    @staticmethod
    def _default_qa_factory() -> ServiceFactory:
        from natural_pdf.services.qa_service import QAService

        def factory(context: "PDFContext") -> QAService:
            return QAService(context)

        return factory

    @staticmethod
    def _default_exclusion_factory() -> ServiceFactory:
        from natural_pdf.services.exclusion_service import ExclusionService

        def factory(context: "PDFContext") -> ExclusionService:
            return ExclusionService(context)

        return factory

    @staticmethod
    def _default_selector_factory() -> ServiceFactory:
        from natural_pdf.services.selector_service import SelectorService

        def factory(context: "PDFContext") -> SelectorService:
            return SelectorService(context)

        return factory

    @staticmethod
    def _default_describe_factory() -> ServiceFactory:
        from natural_pdf.services.describe_service import DescribeService

        def factory(context: "PDFContext") -> DescribeService:
            return DescribeService(context)

        return factory

    @staticmethod
    def _default_vision_factory() -> ServiceFactory:
        from natural_pdf.services.vision_service import VisualSearchService

        def factory(context: "PDFContext") -> VisualSearchService:
            return VisualSearchService(context)

        return factory

    @staticmethod
    def _default_shape_factory() -> ServiceFactory:
        from natural_pdf.services.shape_detection_service import ShapeDetectionService

        def factory(context: "PDFContext") -> ShapeDetectionService:
            return ShapeDetectionService(context)

        return factory

    @staticmethod
    def _default_checkbox_factory() -> ServiceFactory:
        from natural_pdf.services.checkbox_service import CheckboxDetectionService

        def factory(context: "PDFContext") -> CheckboxDetectionService:
            return CheckboxDetectionService(context)

        return factory

    @staticmethod
    def _default_guides_factory() -> ServiceFactory:
        from natural_pdf.services.guides_service import GuidesService

        def factory(context: "PDFContext") -> GuidesService:
            return GuidesService(context)

        return factory

    @staticmethod
    def _default_layout_factory() -> ServiceFactory:
        from natural_pdf.services.layout_service import LayoutService

        def factory(context: "PDFContext") -> LayoutService:
            return LayoutService(context)

        return factory

    @staticmethod
    def _default_rendering_factory() -> ServiceFactory:
        from natural_pdf.services.rendering_service import RenderingService

        def factory(context: "PDFContext") -> RenderingService:
            return RenderingService(context)

        return factory
