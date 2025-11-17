from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Tuple, cast

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


class _OCRElementManager(Protocol):
    def remove_ocr_elements(self) -> int: ...

    def clear_text_layer(self) -> Tuple[int, int]: ...

    def create_text_elements_from_ocr(
        self,
        ocr_results: Any,
        scale_x: Optional[float] = None,
        scale_y: Optional[float] = None,
    ) -> List[Any]: ...


class SupportsOCRElementManager(Protocol):
    def _ocr_element_manager(self) -> _OCRElementManager: ...


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
            if value is not sentinel:
                coerced = OCRService._coerce_int(value)
                if coerced is not None:
                    return coerced

        pdf_obj = getattr(host, "pdf", getattr(host, "_parent", None))
        if pdf_obj is not None:
            cfg = getattr(pdf_obj, "_config", None)
            if isinstance(cfg, dict):
                pdf_resolution = cfg.get("resolution")
                coerced = OCRService._coerce_int(pdf_resolution)
                if coerced is not None:
                    return coerced

        page_obj = getattr(host, "page", None)
        if page_obj is not None:
            cfg = getattr(page_obj, "_config", None)
            if isinstance(cfg, dict):
                page_resolution = cfg.get("resolution")
                coerced = OCRService._coerce_int(page_resolution)
                if coerced is not None:
                    return coerced

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

    @register_delegate("ocr", "apply_custom_ocr")
    def apply_custom_ocr(
        self,
        host,
        *,
        ocr_function: Callable[[Any], Optional[str]],
        source_label: str = "custom-ocr",
        replace: bool = True,
        confidence: Optional[float] = None,
        add_to_page: bool = True,
    ):
        if not callable(ocr_function):
            raise TypeError("ocr_function must be callable.")

        if replace:
            self._remove_custom_ocr_elements(host, source_label=source_label)

        logger.debug("Running custom OCR function for %s", host)
        ocr_text = ocr_function(host)
        if ocr_text is not None and not isinstance(ocr_text, str):
            raise TypeError(
                f"Custom OCR function returned {type(ocr_text).__name__}; expected str or None."
            )

        if ocr_text is None:
            logger.debug("Custom OCR function returned None; no elements created.")
            return host

        to_text_element = getattr(host, "to_text_element", None)
        if not callable(to_text_element):
            raise AttributeError(
                f"{host.__class__.__name__} must implement to_text_element() for custom OCR."
            )

        to_text_element(
            text_content=ocr_text,
            source_label=source_label,
            confidence=confidence,
            add_to_page=add_to_page,
        )
        logger.info(
            "Created custom OCR text element (%d chars) via %s.",
            len(ocr_text),
            source_label,
        )
        return host

    @staticmethod
    def _coerce_int(value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                return int(float(stripped))
            except (ValueError, TypeError):
                return None
        return None

    def _remove_custom_ocr_elements(self, host, *, source_label: str) -> None:
        page = getattr(host, "page", None)
        if page is None:
            return
        get_elements = getattr(page, "get_elements_by_type", None)
        remove_element = getattr(page, "remove_element", None)
        if not callable(get_elements) or not callable(remove_element):
            return

        intersects = getattr(host, "intersects", None)
        bbox = getattr(host, "bbox", None)

        def _overlaps(candidate: Any) -> bool:
            if callable(intersects):
                try:
                    return bool(intersects(candidate))
                except Exception:
                    return False
            if bbox is None:
                return True
            candidate_bbox = self._element_bbox(candidate)
            if candidate_bbox is None:
                return False
            x0, top, x1, bottom = candidate_bbox
            hx0, htop, hx1, hbottom = bbox
            return not (x1 < hx0 or x0 > hx1 or bottom < htop or top > hbottom)

        def _safe_remove(element: Any):
            etype = getattr(element, "object_type", None)
            try:
                remove_element(element, element_type=etype)
            except Exception:
                return

        removed = 0
        word_iter = get_elements("words") or []
        for word in list(cast(Iterable[Any], word_iter)):
            source = getattr(word, "source", "")
            if source not in {"ocr", source_label}:
                continue
            if _overlaps(word):
                _safe_remove(word)
                removed += 1

        char_iter = get_elements("chars") or []
        for char in list(cast(Iterable[Any], char_iter)):
            char_source = (
                char.get("source") if isinstance(char, dict) else getattr(char, "source", None)
            )
            if char_source not in {"ocr", source_label}:
                continue
            if _overlaps(char):
                _safe_remove(char)
                removed += 1

        if removed:
            logger.info("Removed %d existing OCR element(s) before custom OCR.", removed)

    @staticmethod
    def _element_bbox(element: Any) -> Optional[Tuple[float, float, float, float]]:
        bbox = getattr(element, "bbox", None)
        if (
            isinstance(bbox, tuple)
            and len(bbox) == 4
            and all(isinstance(coord, (int, float)) for coord in bbox)
        ):
            return float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        if isinstance(element, dict):
            try:
                return (
                    float(element.get("x0", 0)),
                    float(element.get("top", 0)),
                    float(element.get("x1", 0)),
                    float(element.get("bottom", 0)),
                )
            except (TypeError, ValueError):
                return None
        if hasattr(element, "x0") and hasattr(element, "x1") and hasattr(element, "top"):
            try:
                return (
                    float(getattr(element, "x0")),
                    float(getattr(element, "top")),
                    float(getattr(element, "x1")),
                    float(getattr(element, "bottom", getattr(element, "top"))),
                )
            except (TypeError, ValueError):
                return None
        return None

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
