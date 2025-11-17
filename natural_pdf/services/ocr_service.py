from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Protocol

from natural_pdf.ocr.ocr_manager import (
    normalize_ocr_options,
    resolve_ocr_device,
    resolve_ocr_engine_name,
    resolve_ocr_languages,
    resolve_ocr_min_confidence,
    run_ocr_apply,
    run_ocr_extract,
)
from natural_pdf.services.registry import register_delegate

logger = logging.getLogger(__name__)


class SupportsOCRElementManager(Protocol):
    def _ocr_element_manager(self): ...


class OCRService:
    """Shared OCR helpers extracted from OCRMixin."""

    def __init__(self, context):
        self._context = context

    @staticmethod
    def _scope(host) -> str:
        scope_getter = getattr(host, "_ocr_scope", None)
        if callable(scope_getter):
            try:
                scope = scope_getter()
            except TypeError:
                scope = scope_getter
            if isinstance(scope, str) and scope:
                return scope
        return "page"

    @staticmethod
    def _render_kwargs(host, *, apply_exclusions: bool) -> Dict[str, Any]:
        hook = getattr(host, "_ocr_render_kwargs", None)
        if callable(hook):
            try:
                kwargs = hook(apply_exclusions=apply_exclusions)
            except TypeError:
                kwargs = hook()
            if isinstance(kwargs, dict):
                return kwargs
        return {"apply_exclusions": apply_exclusions}

    @staticmethod
    def _resolve_resolution(host, requested: Optional[int], scope: str) -> int:
        if requested is not None:
            return requested

        getter = getattr(host, "get_config", None)
        if callable(getter):
            sentinel = object()
            value = sentinel
            try:
                value = getter("resolution", sentinel, scope=scope)
            except TypeError:
                try:
                    value = getter("resolution", sentinel)
                except TypeError:
                    value = sentinel
            if value is not sentinel and value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass

        pdf_obj = getattr(host, "pdf", getattr(host, "_parent", None))
        if pdf_obj is not None:
            cfg = getattr(pdf_obj, "_config", None)
            if isinstance(cfg, dict):
                pdf_resolution = cfg.get("resolution")
                if pdf_resolution is not None:
                    try:
                        return int(pdf_resolution)
                    except (TypeError, ValueError):
                        pass

        page_obj = getattr(host, "page", None)
        if page_obj is not None:
            cfg = getattr(page_obj, "_config", None)
            if isinstance(cfg, dict):
                page_resolution = cfg.get("resolution")
                if page_resolution is not None:
                    try:
                        return int(page_resolution)
                    except (TypeError, ValueError):
                        pass

        return 150

    @register_delegate("ocr", "remove_ocr_elements")
    def remove_ocr_elements(self, host: SupportsOCRElementManager) -> int:
        mgr = host._ocr_element_manager()
        return int(mgr.remove_ocr_elements())

    @register_delegate("ocr", "clear_text_layer")
    def clear_text_layer(self, host: SupportsOCRElementManager):
        mgr = host._ocr_element_manager()
        return mgr.clear_text_layer()

    @register_delegate("ocr", "create_text_elements_from_ocr")
    def create_text_elements_from_ocr(
        self,
        host: SupportsOCRElementManager,
        ocr_results: Any,
        scale_x: Optional[float] = None,
        scale_y: Optional[float] = None,
    ):
        mgr = host._ocr_element_manager()
        return mgr.create_text_elements_from_ocr(ocr_results, scale_x=scale_x, scale_y=scale_y)

    @register_delegate("ocr", "apply_ocr")
    def apply_ocr(
        self,
        host,
        *,
        engine: Optional[str] = None,
        options: Optional[Any] = None,
        languages: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        device: Optional[str] = None,
        resolution: Optional[int] = None,
        detect_only: bool = False,
        apply_exclusions: bool = True,
        replace: bool = True,
        **kwargs,
    ):
        normalized_options = normalize_ocr_options(options)
        scope = self._scope(host)
        engine_name = resolve_ocr_engine_name(
            context=host, requested=engine, options=normalized_options, scope=scope
        )
        resolved_languages = resolve_ocr_languages(host, languages, scope=scope)
        resolved_min_conf = resolve_ocr_min_confidence(host, min_confidence, scope=scope)
        resolved_device = resolve_ocr_device(host, device, scope=scope)

        if replace:
            removed = self.remove_ocr_elements(host)
            if removed:
                logger.info("Removed %d OCR elements before new OCR run.", removed)

        final_resolution = self._resolve_resolution(host, resolution, scope)
        render_kwargs = self._render_kwargs(host, apply_exclusions=apply_exclusions)

        ocr_payload = run_ocr_apply(
            target=host,
            context=host,
            engine_name=engine_name,
            resolution=final_resolution,
            languages=resolved_languages,
            min_confidence=resolved_min_conf,
            device=resolved_device,
            detect_only=detect_only,
            options=normalized_options,
            render_kwargs=render_kwargs,
        )

        image_width, image_height = ocr_payload.image_size
        if not image_width or not image_height:
            logger.error("OCR payload missing image dimensions.")
            return host

        width = (
            getattr(host, "width", None) or getattr(getattr(host, "page", None), "width", None) or 0
        )
        height = (
            getattr(host, "height", None)
            or getattr(getattr(host, "page", None), "height", None)
            or 0
        )
        scale_x = width / image_width if width else 1.0
        scale_y = height / image_height if height else 1.0
        created_elements = self.create_text_elements_from_ocr(
            host, ocr_payload.results, scale_x=scale_x, scale_y=scale_y
        )
        logger.info("Added %d OCR elements using '%s'.", len(created_elements), engine_name)
        return host

    @register_delegate("ocr", "extract_ocr_elements")
    def extract_ocr_elements(
        self,
        host,
        *,
        engine: Optional[str] = None,
        options: Optional[Any] = None,
        languages: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        device: Optional[str] = None,
        resolution: Optional[int] = None,
    ):
        normalized_options = normalize_ocr_options(options)
        scope = self._scope(host)
        engine_name = resolve_ocr_engine_name(
            context=host, requested=engine, options=normalized_options, scope=scope
        )
        resolved_languages = resolve_ocr_languages(host, languages, scope=scope)
        resolved_min_conf = resolve_ocr_min_confidence(host, min_confidence, scope=scope)
        resolved_device = resolve_ocr_device(host, device, scope=scope)

        final_resolution = self._resolve_resolution(host, resolution, scope)
        render_kwargs = self._render_kwargs(host, apply_exclusions=True)

        ocr_payload = run_ocr_extract(
            target=host,
            context=host,
            engine_name=engine_name,
            resolution=final_resolution,
            languages=resolved_languages,
            min_confidence=resolved_min_conf,
            device=resolved_device,
            detect_only=False,
            options=normalized_options,
            render_kwargs=render_kwargs,
        )

        results = ocr_payload.results
        image_width, image_height = ocr_payload.image_size
        if not image_width or not image_height:
            logger.error("OCR payload missing image dimensions.")
            return []

        width = (
            getattr(host, "width", None) or getattr(getattr(host, "page", None), "width", None) or 0
        )
        height = (
            getattr(host, "height", None)
            or getattr(getattr(host, "page", None), "height", None)
            or 0
        )
        scale_x = width / image_width if width else 1.0
        scale_y = height / image_height if height else 1.0
        return self.create_text_elements_from_ocr(host, results, scale_x=scale_x, scale_y=scale_y)
