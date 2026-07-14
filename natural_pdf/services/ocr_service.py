from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Tuple, cast

from natural_pdf.core.ocr_converter import validate_classic_ocr_results, validate_ocr_image_size
from natural_pdf.core.ocr_execution import protected_ocr_artifact_ids, register_ocr_artifacts
from natural_pdf.exceptions import OCRError
from natural_pdf.ocr.ocr_manager import (
    normalize_ocr_options,
    resolve_ocr_device,
    resolve_ocr_languages,
    resolve_ocr_min_confidence,
)
from natural_pdf.ocr.replacement import OCRReplaceMode, normalize_ocr_replace_mode
from natural_pdf.ocr.unified_dispatch import get_registry, run_ocr
from natural_pdf.services.registry import register_delegate

logger = logging.getLogger(__name__)


class _OCRElementManager(Protocol):
    def remove_ocr_elements(
        self, bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> int: ...

    def clear_text_layer(
        self, bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> Tuple[int, int]: ...

    def remove_text_elements_in_bbox(
        self,
        bbox: Optional[Tuple[float, float, float, float]],
        *,
        sources: Optional[Iterable[str]] = None,
        predicate: Optional[Callable[[Any], bool]] = None,
    ) -> Tuple[int, int]: ...

    def create_text_elements_from_ocr(
        self,
        ocr_results: Any,
        scale_x: Optional[float] = None,
        scale_y: Optional[float] = None,
        engine_name: Optional[str] = None,
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

    def _render_kwargs(self, host, *, apply_exclusions: bool) -> Dict[str, Any]:
        hook = getattr(host, "_ocr_render_kwargs", None)
        if callable(hook):
            try:
                kwargs = hook(apply_exclusions=apply_exclusions)
            except TypeError:
                kwargs = hook()
            if isinstance(kwargs, dict):
                resolved = dict(kwargs)
            else:
                resolved = {}
        else:
            resolved = {}

        resolved["apply_exclusions"] = apply_exclusions
        crop_bbox = self._resolve_crop_bbox(host, resolved)
        if crop_bbox is not None:
            if crop_bbox[2] <= crop_bbox[0] or crop_bbox[3] <= crop_bbox[1]:
                raise ValueError(
                    "OCR crop has no area within the page bounds: "
                    f"requested {resolved.get('crop_bbox')!r}, resolved {crop_bbox!r}."
                )
            resolved["crop_bbox"] = crop_bbox
        if apply_exclusions:
            resolved["_ocr_exclusion_bboxes"] = self._resolve_exclusion_bboxes(
                host, crop_bbox=crop_bbox
            )
        return resolved

    @staticmethod
    def _resolve_crop_bbox(
        host, render_kwargs: Optional[Dict[str, Any]]
    ) -> Optional[Tuple[float, float, float, float]]:
        if not render_kwargs:
            return None

        crop_bbox = render_kwargs.get("crop_bbox")
        if (
            isinstance(crop_bbox, (list, tuple))
            and len(crop_bbox) == 4
            and all(isinstance(coord, (int, float)) for coord in crop_bbox)
        ):
            resolved = tuple(float(coord) for coord in crop_bbox)

            page = getattr(host, "page", host)
            page_width = getattr(page, "width", None)
            page_height = getattr(page, "height", None)
            if isinstance(page_width, (int, float)) and isinstance(page_height, (int, float)):
                x0, y0, x1, y1 = resolved
                return (
                    max(0.0, min(float(page_width), x0)),
                    max(0.0, min(float(page_height), y0)),
                    max(0.0, min(float(page_width), x1)),
                    max(0.0, min(float(page_height), y1)),
                )
            return resolved

        if render_kwargs.get("crop"):
            bbox = getattr(host, "bbox", None)
            if (
                isinstance(bbox, (list, tuple))
                and len(bbox) == 4
                and all(isinstance(coord, (int, float)) for coord in bbox)
            ):
                return OCRService._resolve_crop_bbox(host, {"crop_bbox": bbox})

        return None

    @classmethod
    def _resolve_exclusion_bboxes(
        cls,
        host,
        *,
        crop_bbox: Optional[Tuple[float, float, float, float]],
    ) -> Tuple[Tuple[float, float, float, float], ...]:
        """Evaluate, clip, normalize, and deterministically order OCR masks."""
        getter = getattr(host, "_get_exclusion_regions", None)
        if not callable(getter):
            return ()

        try:
            regions = getter(include_callable=True)
        except TypeError:
            regions = getter()

        exclusion_items: List[Any] = list(regions or ())
        exclusion_items.extend(cls._resolve_element_exclusion_items(host))

        page = getattr(host, "page", host)
        page_width = float(getattr(page, "width", 0.0) or 0.0)
        page_height = float(getattr(page, "height", 0.0) or 0.0)
        clip = crop_bbox or (0.0, 0.0, page_width, page_height)
        clip_x0, clip_y0, clip_x1, clip_y1 = clip

        normalized = set()
        for item in exclusion_items:
            bbox = cls._element_bbox(item)
            if bbox is None:
                continue
            x0 = max(clip_x0, bbox[0])
            y0 = max(clip_y0, bbox[1])
            x1 = min(clip_x1, bbox[2])
            y1 = min(clip_y1, bbox[3])
            if x1 <= x0 or y1 <= y0:
                continue
            normalized.add(tuple(round(value, 6) for value in (x0, y0, x1, y1)))

        return tuple(sorted(normalized))

    @classmethod
    def _resolve_element_exclusion_items(cls, host) -> List[Any]:
        """Resolve exact ``method='element'`` exclusions for OCR masking only.

        Ordinary exclusion-region evaluation deliberately leaves these entries
        to the selector filtering stage. OCR rasterization has no such later
        stage, so it must collect their exact boxes without re-evaluating any
        callable exclusions.
        """

        def normalize(specs):
            return [
                (spec[0], spec[1], spec[2] if len(spec) == 3 else "region") for spec in specs or ()
            ]

        page = getattr(host, "page", host)
        contextual_entries: List[Tuple[Any, Tuple[Any, Any, str]]] = []

        # Region-local exclusions are additive; they do not shadow page entries.
        if page is not host:
            contextual_entries.extend(
                (host, entry) for entry in normalize(getattr(host, "_exclusions", ()))
            )

        page_entries = normalize(getattr(page, "_exclusions", ()))
        contextual_entries.extend((page, entry) for entry in page_entries)

        # Match Page._get_exclusion_regions label-shadowing rules for PDF-level
        # entries so the geometry here is the geometry effective for the page.
        parent = getattr(page, "_parent", None)
        parent_entries = normalize(getattr(parent, "_exclusions", ()))
        page_labels = {label for _, label, _ in page_entries if label}
        contextual_entries.extend(
            (page, entry) for entry in parent_entries if not (entry[1] and entry[1] in page_labels)
        )

        resolved: List[Any] = []
        for context, (item, _label, method) in contextual_entries:
            if method != "element" or callable(item):
                continue
            if isinstance(item, str):
                finder = getattr(context, "find_all", None)
                if not callable(finder):
                    continue
                matches = finder(item, apply_exclusions=False)
                resolved.extend(getattr(matches, "elements", matches) or ())
                continue
            elements = getattr(item, "elements", None)
            if elements is not None:
                resolved.extend(elements)
            else:
                resolved.append(item)

        return resolved

    @staticmethod
    def _exclusion_fingerprint(
        render_kwargs: Optional[Dict[str, Any]],
    ) -> str:
        bboxes = (render_kwargs or {}).get("_ocr_exclusion_bboxes") or ()
        return ";".join(",".join(f"{float(coord):.6f}" for coord in bbox) for bbox in bboxes)

    @staticmethod
    def _target_dimensions(
        host,
        crop_bbox: Optional[Tuple[float, float, float, float]],
    ) -> Tuple[float, float]:
        if crop_bbox is not None:
            return crop_bbox[2] - crop_bbox[0], crop_bbox[3] - crop_bbox[1]
        page = getattr(host, "page", host)
        return (
            float(getattr(host, "width", None) or getattr(page, "width", 0.0) or 0.0),
            float(getattr(host, "height", None) or getattr(page, "height", 0.0) or 0.0),
        )

    @classmethod
    def _resolve_offsets(cls, host, render_kwargs: Optional[Dict[str, Any]]) -> Tuple[float, float]:
        crop_bbox = cls._resolve_crop_bbox(host, render_kwargs)
        if crop_bbox is None:
            return 0.0, 0.0
        return crop_bbox[0], crop_bbox[1]

    def _resolve_resolution(self, host, requested: Optional[int], scope: str) -> int:
        if requested is not None:
            return requested

        option_value = self._context.get_option(
            "ocr",
            "resolution",
            host=host,
            default=None,
            scope=scope,
        )
        if option_value is not None:
            coerced = self._coerce_int(option_value)
            if coerced is not None:
                return coerced

        return 150

    @staticmethod
    def _host_bbox(host) -> Optional[Tuple[float, float, float, float]]:
        """Return a single-page host's replacement geometry, if it has one."""

        bbox = getattr(host, "bbox", None)
        if (
            isinstance(bbox, (list, tuple))
            and len(bbox) == 4
            and all(isinstance(coord, (int, float)) for coord in bbox)
        ):
            return cast(Tuple[float, float, float, float], tuple(float(coord) for coord in bbox))
        return None

    @staticmethod
    def _center_in_bbox(element: Any, bbox: Optional[Tuple[float, float, float, float]]) -> bool:
        if bbox is None:
            return True
        element_bbox = OCRService._element_bbox(element)
        if element_bbox is None:
            return False
        x0, top, x1, bottom = element_bbox
        bx0, btop, bx1, bbottom = bbox
        return bx0 <= (x0 + x1) / 2.0 <= bx1 and btop <= (top + bottom) / 2.0 <= bbottom

    @register_delegate("ocr", "remove_ocr_elements")
    def remove_ocr_elements(self, host: SupportsOCRElementManager) -> int:
        mgr = host._ocr_element_manager()
        bbox = self._host_bbox(host)
        if bbox is not None:
            removed = int(mgr.remove_ocr_elements(bbox))
        else:
            # Compatibility for third-party page-like managers. Real Region
            # managers provide the geometry-aware method above.
            try:
                removed = int(mgr.remove_ocr_elements(None))
            except TypeError:
                removed = int(mgr.remove_ocr_elements())
        return removed

    @register_delegate("ocr", "clear_text_layer")
    def clear_text_layer(self, host: SupportsOCRElementManager):
        mgr = host._ocr_element_manager()
        bbox = self._host_bbox(host)
        if bbox is not None:
            removed = mgr.clear_text_layer(bbox)
        else:
            try:
                removed = mgr.clear_text_layer(None)
            except TypeError:
                removed = mgr.clear_text_layer()
        return removed

    @register_delegate("ocr", "create_text_elements_from_ocr")
    def create_text_elements_from_ocr(
        self,
        host: SupportsOCRElementManager,
        ocr_results: Any,
        scale_x: Optional[float] = None,
        scale_y: Optional[float] = None,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        engine_name: Optional[str] = None,
    ):
        ocr_results = validate_classic_ocr_results(ocr_results)
        mgr = host._ocr_element_manager()
        created = mgr.create_text_elements_from_ocr(
            ocr_results,
            scale_x=scale_x,
            scale_y=scale_y,
            offset_x=offset_x,
            offset_y=offset_y,
            engine_name=engine_name,
        )
        return created

    def _resolve_engine_name(
        self,
        host,
        requested: Optional[str],
        options: Optional[Any],
        scope: str,
    ) -> str:
        """Resolve engine name using the unified registry."""
        registry = get_registry()
        if requested is not None:
            normalized = requested.strip().lower()
            if normalized in registry:
                return normalized

        # Fall back to config/global defaults for classic engines
        from natural_pdf.ocr.ocr_manager import resolve_ocr_engine_name

        return resolve_ocr_engine_name(
            context=host, requested=requested, options=options, scope=scope
        )

    @staticmethod
    def _as_detection_results(results: Iterable[Any]) -> List[Dict[str, Any]]:
        return [
            {
                **result,
                "text": "",
                "source_category": "detection",
                "_ocr_detection_only": True,
            }
            for result in results
            if isinstance(result, Mapping)
        ]

    @staticmethod
    def _normalize_payload_image_size(ocr_payload: Any) -> Tuple[float, float]:
        dimensions = validate_ocr_image_size(getattr(ocr_payload, "image_size", None))
        ocr_payload.image_size = dimensions
        return dimensions

    @staticmethod
    def _normalize_classic_payload_results(
        ocr_payload: Any,
        *,
        detection_only: bool | None = None,
    ) -> List[Mapping[str, Any]]:
        results = validate_classic_ocr_results(
            getattr(ocr_payload, "results", None),
            detection_only=detection_only,
        )
        ocr_payload.results = results
        return results

    @staticmethod
    def _scale_vlm_payload_results(
        ocr_payload: Any,
        *,
        image_width: float,
        image_height: float,
        target_width: float,
        target_height: float,
        offset_x: float,
        offset_y: float,
        min_confidence: Optional[float],
        detection_only: bool,
    ) -> List[Mapping[str, Any]]:
        """Materialize, scale, and validate a complete VLM payload."""

        from natural_pdf.ocr.vlm_ocr import scale_ocr_results, validate_table_ocr_results

        raw_results = getattr(ocr_payload, "results", None)
        if isinstance(raw_results, (str, bytes, bytearray, Mapping)):
            raise OCRError("Invalid VLM OCR payload: results must be an iterable of mappings.")
        try:
            materialized_results = list(raw_results)
        except TypeError as exc:
            raise OCRError(
                "Invalid VLM OCR payload: results must be an iterable of mappings."
            ) from exc

        # Preserve one-shot provider output for preflight, cache, and apply
        # paths that may inspect the same payload more than once.
        ocr_payload.results = materialized_results
        try:
            scaled = scale_ocr_results(
                materialized_results,
                image_width=image_width,
                image_height=image_height,
                page_width=target_width,
                page_height=target_height,
                offset_x=offset_x,
                offset_y=offset_y,
            )
        except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
            raise OCRError("Invalid VLM OCR payload: results could not be scaled.") from exc

        normalized = validate_classic_ocr_results(
            scaled,
            detection_only=detection_only,
        )
        if not validate_table_ocr_results(normalized):
            raise OCRError("Invalid VLM OCR payload: malformed table result.")

        if min_confidence is not None:
            normalized = [
                result
                for result in normalized
                if result.get("confidence") is None or float(result["confidence"]) >= min_confidence
            ]
        return normalized

    @staticmethod
    def _mark_detection_elements(elements: Iterable[Any]) -> None:
        for element in elements:
            metadata = getattr(element, "metadata", None)
            if isinstance(metadata, dict):
                metadata["ocr_detection_only"] = True
            element.is_ocr_detection = True

    @staticmethod
    def _is_detection_element(element: Any) -> bool:
        """Return whether an element is a persistent detection-only artifact."""

        if bool(getattr(element, "is_ocr_detection", False)):
            return True
        metadata = getattr(element, "metadata", None)
        if isinstance(metadata, dict) and bool(metadata.get("ocr_detection_only", False)):
            return True
        obj = getattr(element, "_obj", None)
        return isinstance(obj, dict) and bool(obj.get("ocr_detection_only", False))

    def _refresh_detection_artifacts(
        self,
        host: Any,
        bbox: Optional[Tuple[float, float, float, float]],
        created: Iterable[Any],
    ) -> Tuple[int, int]:
        """Remove superseded detections in scope while preserving new results."""

        created_ids = {id(element) for element in created}
        protected_ids = protected_ocr_artifact_ids() | created_ids
        manager = host._ocr_element_manager()
        return manager.remove_text_elements_in_bbox(
            bbox,
            sources={"ocr"},
            predicate=lambda element: id(element) not in protected_ids
            and self._is_detection_element(element),
        )

    def _process_ocr_payload(
        self,
        host,
        ocr_payload,
        engine_name,
        min_confidence,
        offset_x,
        offset_y,
        crop_bbox=None,
        detect_only: bool = False,
    ):
        """Scale OCR results and create text elements.

        Returns the created element list. Invalid payloads raise :class:`OCRError`.
        Used by both cache-hit and fresh-OCR paths to avoid duplication.
        """
        image_width, image_height = self._normalize_payload_image_size(ocr_payload)

        width, height = self._target_dimensions(host, crop_bbox)
        scale_x = width / image_width if width else 1.0
        scale_y = height / image_height if height else 1.0

        if ocr_payload.engine_type == "vlm":
            from natural_pdf.ocr.vlm_ocr import create_table_regions_from_ocr

            scaled = self._scale_vlm_payload_results(
                ocr_payload,
                image_width=image_width,
                image_height=image_height,
                target_width=width,
                target_height=height,
                offset_x=offset_x,
                offset_y=offset_y,
                min_confidence=min_confidence,
                detection_only=detect_only,
            )

            if detect_only:
                detection_results = validate_classic_ocr_results(
                    self._as_detection_results(scaled),
                    detection_only=True,
                )
                created = self.create_text_elements_from_ocr(
                    host,
                    detection_results,
                    scale_x=1.0,
                    scale_y=1.0,
                    offset_x=0.0,
                    offset_y=0.0,
                    engine_name=engine_name,
                )
                self._mark_detection_elements(created)
                return created

            text_results = [
                result
                for result in scaled
                if str(result.get("source_category", "")).lower() != "table"
            ]

            # Validate every non-table result before table Regions are
            # registered. A malformed later text entry must not leave an
            # earlier table partially attached to the page.
            page_obj = getattr(host, "page", host)
            _ignored_text_results, table_regions = create_table_regions_from_ocr(
                page_obj,
                scaled,
                source_label=engine_name,
            )
            if table_regions:
                for region in table_regions:
                    metadata = getattr(region, "metadata", None)
                    if isinstance(metadata, dict):
                        metadata["_natural_pdf_ocr_generated"] = True
                        metadata["ocr_engine"] = engine_name
                    region._natural_pdf_ocr_generated = True
                register_ocr_artifacts(*table_regions)

            try:
                return self.create_text_elements_from_ocr(
                    host,
                    text_results,
                    scale_x=1.0,
                    scale_y=1.0,
                    offset_x=0.0,
                    offset_y=0.0,
                    engine_name=engine_name,
                )
            except Exception:
                # Text construction is normally atomic in ElementManager, but
                # third-party managers can still fail after table registration.
                # Do not leave the table half of one VLM payload attached.
                remover = getattr(page_obj, "remove_regions", None)
                if callable(remover) and table_regions:
                    table_ids = {id(region) for region in table_regions}
                    remover(predicate=lambda region: id(region) in table_ids)
                raise
        else:
            results = ocr_payload.results
            if detect_only:
                # Detection-only output remains selector-visible geometry, but
                # is explicitly labelled and never replaces recognized text.
                results = self._as_detection_results(results)
            created = self.create_text_elements_from_ocr(
                host,
                results,
                scale_x=scale_x,
                scale_y=scale_y,
                offset_x=offset_x,
                offset_y=offset_y,
                engine_name=engine_name,
            )
            if detect_only:
                self._mark_detection_elements(created)
            return created

    def _payload_can_refresh_detection(
        self,
        host: Any,
        ocr_payload: Any,
        engine_name: str,
        min_confidence: Optional[float],
        offset_x: float,
        offset_y: float,
        crop_bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> bool:
        """Validate detection geometry before replacing prior detections.

        A genuinely empty result is a successful refresh. A non-empty payload
        whose entries cannot create any bounded detection element is malformed
        and must leave the previous detection state untouched.
        """

        try:
            image_width, image_height = self._normalize_payload_image_size(ocr_payload)
        except (AttributeError, OCRError):
            return False

        width, height = self._target_dimensions(host, crop_bbox)
        scale_x = width / image_width if width else 1.0
        scale_y = height / image_height if height else 1.0
        staged_offset_x = offset_x
        staged_offset_y = offset_y

        if getattr(ocr_payload, "engine_type", None) != "vlm":
            try:
                staged_results = validate_classic_ocr_results(
                    getattr(ocr_payload, "results", None),
                    detection_only=True,
                )
            except OCRError:
                return False
            ocr_payload.results = staged_results
        else:
            try:
                staged_results = self._scale_vlm_payload_results(
                    ocr_payload,
                    image_width=image_width,
                    image_height=image_height,
                    target_width=width,
                    target_height=height,
                    offset_x=offset_x,
                    offset_y=offset_y,
                    min_confidence=min_confidence,
                    detection_only=True,
                )
            except OCRError:
                return False
            scale_x = scale_y = 1.0
            staged_offset_x = staged_offset_y = 0.0

        if not staged_results:
            return True

        detection_results = self._as_detection_results(staged_results)
        if not detection_results:
            return False

        manager = host._ocr_element_manager()
        converter = getattr(manager, "_ocr_converter", None)
        if converter is None:
            from natural_pdf.core.ocr_converter import OCRConverter

            converter = OCRConverter(getattr(host, "page", host))
        try:
            words, _chars = converter.convert(
                detection_results,
                scale_x=scale_x,
                scale_y=scale_y,
                offset_x=staged_offset_x,
                offset_y=staged_offset_y,
                engine_name=engine_name,
            )
        except Exception:
            logger.debug("Failed to validate detection payload.", exc_info=True)
            return False
        return bool(words)

    def _remove_generated_table_regions(
        self,
        host,
        bbox: Optional[Tuple[float, float, float, float]],
    ) -> int:
        """Remove only VLM table Regions created by an earlier OCR apply."""

        page = getattr(host, "page", host)
        remover = getattr(page, "remove_regions", None)
        if not callable(remover):
            return 0

        def is_generated_table(region: Any) -> bool:
            metadata = getattr(region, "metadata", None)
            generated = bool(getattr(region, "_natural_pdf_ocr_generated", False)) or (
                isinstance(metadata, dict)
                and bool(metadata.get("_natural_pdf_ocr_generated", False))
            )
            return (
                generated
                and id(region) not in protected_ocr_artifact_ids()
                and getattr(region, "region_type", None) == "table"
                and self._center_in_bbox(region, bbox)
            )

        return int(remover(predicate=is_generated_table))

    def _remove_for_replace(
        self,
        host,
        replace: OCRReplaceMode,
        bbox: Optional[Tuple[float, float, float, float]],
    ) -> Tuple[int, int, int]:
        """Apply a validated replacement mode inside one page geometry."""

        if replace == "none":
            return 0, 0, 0

        mgr = host._ocr_element_manager()
        sources = None if replace == "all" else {"ocr"}
        protected_ids = protected_ocr_artifact_ids()
        remove_scoped = getattr(mgr, "remove_text_elements_in_bbox", None)
        if callable(remove_scoped):
            words, chars = remove_scoped(
                bbox,
                sources=sources,
                predicate=lambda element: id(element) not in protected_ids,
            )
        elif bbox is None:
            if replace == "all":
                words, chars = mgr.clear_text_layer()
            else:
                removed = int(mgr.remove_ocr_elements())
                words, chars = removed, 0
        else:  # pragma: no cover - only third-party incomplete managers
            raise TypeError("Region OCR replacement requires a geometry-aware element manager")

        tables = self._remove_generated_table_regions(host, bbox)
        return int(words), int(chars), tables

    def _convert_ocr_payload(
        self,
        host,
        ocr_payload,
        engine_name,
        min_confidence,
        offset_x,
        offset_y,
        crop_bbox=None,
    ):
        """Convert an OCR payload to detached text elements without registration."""
        image_width, image_height = self._normalize_payload_image_size(ocr_payload)

        width, height = self._target_dimensions(host, crop_bbox)
        scale_x = width / image_width if width else 1.0
        scale_y = height / image_height if height else 1.0
        results = ocr_payload.results

        if ocr_payload.engine_type == "vlm":
            results = self._scale_vlm_payload_results(
                ocr_payload,
                image_width=image_width,
                image_height=image_height,
                target_width=width,
                target_height=height,
                offset_x=offset_x,
                offset_y=offset_y,
                min_confidence=min_confidence,
                detection_only=False,
            )
            # Table payloads are represented as registered Regions only in the
            # mutating apply path. Extraction must not leak selector-visible
            # artifacts into the page.
            results = [
                result
                for result in results
                if str(result.get("source_category", "")).lower() != "table"
            ]
            scale_x = scale_y = 1.0
            offset_x = offset_y = 0.0

        manager = host._ocr_element_manager()
        converter = getattr(manager, "_ocr_converter", None)
        if converter is None:
            from natural_pdf.core.ocr_converter import OCRConverter

            converter = OCRConverter(getattr(host, "page", host))

        words, _chars = converter.convert(
            results,
            scale_x=scale_x,
            scale_y=scale_y,
            offset_x=offset_x,
            offset_y=offset_y,
            engine_name=engine_name,
        )
        return list(words)

    def _payload_can_replace_text(
        self,
        host,
        ocr_payload,
        engine_name,
        min_confidence,
        offset_x,
        offset_y,
        crop_bbox=None,
    ) -> bool:
        validated_classic_results = None
        if getattr(ocr_payload, "engine_type", None) != "vlm":
            try:
                validated_classic_results = self._normalize_classic_payload_results(
                    ocr_payload,
                    detection_only=False,
                )
            except (AttributeError, OCRError):
                return False

        try:
            image_width, image_height = self._normalize_payload_image_size(ocr_payload)
        except (AttributeError, OCRError):
            return False

        width, height = self._target_dimensions(host, crop_bbox)
        scale_x = width / image_width if width else 1.0
        scale_y = height / image_height if height else 1.0

        staged_results = (
            validated_classic_results
            if validated_classic_results is not None
            else ocr_payload.results
        )
        staged_scale_x = scale_x
        staged_scale_y = scale_y
        staged_offset_x = offset_x
        staged_offset_y = offset_y

        if ocr_payload.engine_type == "vlm":
            try:
                scaled = self._scale_vlm_payload_results(
                    ocr_payload,
                    image_width=image_width,
                    image_height=image_height,
                    target_width=width,
                    target_height=height,
                    offset_x=offset_x,
                    offset_y=offset_y,
                    min_confidence=min_confidence,
                    detection_only=False,
                )
            except OCRError:
                logger.debug("Failed to scale OCR payload before replacement.", exc_info=True)
                return False

            staged_results = [
                result
                for result in scaled
                if str(result.get("source_category", "")).lower() != "table"
            ]
            has_table_results = any(
                str(result.get("source_category", "")).lower() == "table"
                and bool(result.get("text"))
                for result in scaled
            )
            staged_scale_x = 1.0
            staged_scale_y = 1.0
            staged_offset_x = 0.0
            staged_offset_y = 0.0
        else:
            has_table_results = False

        if not staged_results:
            return has_table_results

        try:
            staged_results = validate_classic_ocr_results(
                staged_results,
                detection_only=False,
            )
        except OCRError:
            return False

        mgr = host._ocr_element_manager()
        converter = getattr(mgr, "_ocr_converter", None)
        if converter is None:
            return True

        try:
            staged_words, staged_chars = converter.convert(
                staged_results,
                scale_x=staged_scale_x,
                scale_y=staged_scale_y,
                offset_x=staged_offset_x,
                offset_y=staged_offset_y,
                engine_name=engine_name,
            )
        except Exception:
            logger.debug("Failed to validate OCR payload before replacement.", exc_info=True)
            return False

        return bool(staged_words or staged_chars)

    def _payload_should_cache(
        self,
        host,
        ocr_payload,
        engine_name,
        min_confidence,
        offset_x,
        offset_y,
        crop_bbox=None,
        *,
        detection_only: bool | None = None,
    ) -> bool:
        """Return whether an OCR payload is safe to persist in the disk cache."""
        try:
            self._normalize_payload_image_size(ocr_payload)
        except (AttributeError, OCRError):
            return False

        if getattr(ocr_payload, "engine_type", None) != "vlm":
            try:
                self._normalize_classic_payload_results(
                    ocr_payload,
                    detection_only=detection_only,
                )
            except (AttributeError, OCRError):
                return False
            return True

        if not getattr(ocr_payload, "results", None):
            return False

        return self._payload_can_replace_text(
            host,
            ocr_payload,
            engine_name,
            min_confidence,
            offset_x,
            offset_y,
            crop_bbox,
        )

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
        replace: OCRReplaceMode = "ocr",
        use_cache: bool = True,
        # VLM params:
        model: Optional[str] = None,
        client: Optional[Any] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        layout: Optional[bool | str] = None,
        preserve_markup: bool = False,
        **kwargs,
    ):
        replace_mode = normalize_ocr_replace_mode(replace)
        if detect_only and replace_mode != "ocr":
            raise ValueError(
                "detect_only=True refreshes detection artifacts and cannot use "
                f"recognition replace={replace_mode!r}"
            )
        normalized_options = normalize_ocr_options(options)
        scope = self._scope(host)

        # If model= or client= provided without engine, default to "vlm"
        if engine is None and (model is not None or client is not None):
            engine = "vlm"

        engine_name = self._resolve_engine_name(host, engine, normalized_options, scope)
        # Resolve deferred mapping options against the selected engine before
        # rendering, cache lookup, or mutation. Invalid options must be a
        # side-effect-free public-boundary failure.
        normalized_options = normalize_ocr_options(
            normalized_options,
            engine_name=engine_name,
        )
        resolved_languages = resolve_ocr_languages(host, languages, scope=scope)
        resolved_min_conf = resolve_ocr_min_confidence(host, min_confidence, scope=scope)
        resolved_device = resolve_ocr_device(host, device, scope=scope)

        final_resolution = self._resolve_resolution(host, resolution, scope)
        render_kwargs = self._render_kwargs(host, apply_exclusions=apply_exclusions)
        crop_bbox = self._resolve_crop_bbox(host, render_kwargs)
        offset_x, offset_y = self._resolve_offsets(host, render_kwargs)

        # --- OCR result cache ---
        from natural_pdf.ocr.ocr_cache import (
            compute_cache_key,
            compute_render_kwargs_cache_key,
            get_default_cache,
            resolve_ocr_cache_identity,
        )

        cache = get_default_cache() if use_cache else None
        cache_key = None

        page_obj = getattr(host, "page", host)
        pdf_obj = getattr(page_obj, "pdf", getattr(page_obj, "_parent", None))
        pdf_path = getattr(pdf_obj, "_resolved_path", None)

        execution_identity = None
        render_kwargs_cache_key = None
        if cache is not None and pdf_path and pdf_path != "<stream>":
            execution_identity = resolve_ocr_cache_identity(
                engine_name=engine_name,
                device=resolved_device,
                model=model,
                client=client,
            )
            render_kwargs_cache_key = compute_render_kwargs_cache_key(render_kwargs)
            if execution_identity is None:
                logger.debug(
                    "Skipping persistent OCR result cache for %s: backend identity is not "
                    "explicitly cacheable.",
                    engine_name,
                )
                cache = None
            elif render_kwargs_cache_key is None:
                logger.debug(
                    "Skipping persistent OCR result cache for %s: render kwargs do not have "
                    "a stable JSON identity.",
                    engine_name,
                )
                cache = None

        if cache is not None and pdf_path and pdf_path != "<stream>":
            assert execution_identity is not None
            assert render_kwargs_cache_key is not None
            try:
                file_stat = os.stat(pdf_path)
                page_index = getattr(page_obj, "index", 0)
                options_key = (
                    normalized_options._cache_key()
                    if hasattr(normalized_options, "_cache_key")
                    else ""
                )
                if options_key is None:
                    logger.debug(
                        "Skipping persistent OCR result cache for %s: options do not have "
                        "a canonical cache identity.",
                        engine_name,
                    )
                else:
                    exclusion_key = self._exclusion_fingerprint(render_kwargs)
                    cache_key = compute_cache_key(
                        pdf_path=pdf_path,
                        file_mtime_ns=file_stat.st_mtime_ns,
                        file_size=file_stat.st_size,
                        page_index=page_index,
                        engine_name=engine_name,
                        languages=tuple(resolved_languages or ["en"]),
                        resolution=final_resolution,
                        detect_only=detect_only,
                        device=execution_identity["device"],
                        options_cache_key=options_key,
                        apply_exclusions=apply_exclusions,
                        min_confidence=resolved_min_conf,
                        model=model,
                        prompt=prompt,
                        instructions=instructions,
                        max_new_tokens=max_new_tokens,
                        layout=layout,
                        preserve_markup=preserve_markup,
                        crop_bbox=crop_bbox,
                        exclusion_geometry_key=exclusion_key,
                        render_kwargs_cache_key=render_kwargs_cache_key,
                        execution_identity=execution_identity,
                    )

                cached = cache.get(cache_key) if cache_key is not None else None
                if cached is not None:
                    logger.info("OCR cache hit for page %d (%s)", page_index, engine_name)
                    if not self._payload_should_cache(
                        host,
                        cached,
                        engine_name,
                        resolved_min_conf,
                        offset_x,
                        offset_y,
                        crop_bbox,
                        detection_only=detect_only,
                    ):
                        cache.delete(cache_key)
                        logger.warning(
                            "Ignoring invalid cached OCR payload for page %d (%s); retrying OCR.",
                            page_index,
                            engine_name,
                        )
                        cached = None
                    if (
                        cached is not None
                        and detect_only
                        and not self._payload_can_refresh_detection(
                            host,
                            cached,
                            engine_name,
                            resolved_min_conf,
                            offset_x,
                            offset_y,
                            crop_bbox,
                        )
                    ):
                        cache.delete(cache_key)
                        logger.warning("Ignoring malformed cached detection payload; retrying OCR.")
                        cached = None
                    destructive = not detect_only and replace_mode != "none"
                    if (
                        cached is not None
                        and destructive
                        and not self._payload_can_replace_text(
                            host,
                            cached,
                            engine_name,
                            resolved_min_conf,
                            offset_x,
                            offset_y,
                            crop_bbox,
                        )
                    ):
                        cache.delete(cache_key)
                        logger.warning(
                            "Ignoring cached OCR payload that cannot safely replace text; "
                            "retrying OCR."
                        )
                        cached = None
                    if cached is not None:
                        if destructive:
                            words, chars, tables = self._remove_for_replace(
                                host, replace_mode, crop_bbox
                            )
                            if words or chars or tables:
                                logger.info(
                                    "Removed %d words, %d chars, and %d OCR table regions "
                                    "before cached OCR.",
                                    words,
                                    chars,
                                    tables,
                                )
                        created = self._process_ocr_payload(
                            host,
                            cached,
                            engine_name,
                            resolved_min_conf,
                            offset_x,
                            offset_y,
                            crop_bbox,
                            detect_only=detect_only,
                        )
                        if created is not None:
                            register_ocr_artifacts(*created)
                            if detect_only:
                                self._refresh_detection_artifacts(host, crop_bbox, created)
                            logger.info(
                                "Added %d OCR elements from cache using '%s'.",
                                len(created),
                                engine_name,
                            )
                            return host
            except OSError:
                pass  # Can't stat file — skip cache

        ocr_payload = run_ocr(
            target=host,
            engine_name=engine_name,
            resolution=final_resolution,
            languages=resolved_languages,
            min_confidence=resolved_min_conf,
            device=resolved_device,
            detect_only=detect_only,
            options=normalized_options,
            render_kwargs=render_kwargs,
            context=host,
            model=model,
            client=client,
            prompt=prompt,
            instructions=instructions,
            max_new_tokens=max_new_tokens,
            layout=layout,
            preserve_markup=preserve_markup,
        )

        image_width, image_height = self._normalize_payload_image_size(ocr_payload)

        if getattr(ocr_payload, "engine_type", None) != "vlm":
            self._normalize_classic_payload_results(
                ocr_payload,
                detection_only=detect_only,
            )
        else:
            # Fresh provider output is a public-boundary failure when malformed.
            # Boolean validation below is reserved for cache eviction and other
            # non-throwing preflight decisions.
            target_width, target_height = self._target_dimensions(host, crop_bbox)
            self._scale_vlm_payload_results(
                ocr_payload,
                image_width=image_width,
                image_height=image_height,
                target_width=target_width,
                target_height=target_height,
                offset_x=offset_x,
                offset_y=offset_y,
                min_confidence=resolved_min_conf,
                detection_only=detect_only,
            )

        detection_valid = not detect_only or self._payload_can_refresh_detection(
            host,
            ocr_payload,
            engine_name,
            resolved_min_conf,
            offset_x,
            offset_y,
            crop_bbox,
        )
        if not detection_valid:
            logger.warning(
                "Skipping detection refresh because the OCR payload contained no valid geometry."
            )
            return host

        # Cache only valid VLM payloads; malformed/empty VLM responses should
        # not poison later retries. Classic engines may legitimately return
        # empty pages, so keep their existing cache behavior.
        if cache_key is not None and self._payload_should_cache(
            host,
            ocr_payload,
            engine_name,
            resolved_min_conf,
            offset_x,
            offset_y,
            crop_bbox,
            detection_only=detect_only,
        ):
            page_index = getattr(page_obj, "index", 0)
            cache.put(cache_key, ocr_payload, engine_name, page_index)

        destructive = not detect_only and replace_mode != "none"
        if destructive:
            if not self._payload_can_replace_text(
                host,
                ocr_payload,
                engine_name,
                resolved_min_conf,
                offset_x,
                offset_y,
                crop_bbox,
            ):
                logger.warning(
                    "Skipping replacement OCR because the OCR payload could not be validated."
                )
                return host
            words, chars, tables = self._remove_for_replace(host, replace_mode, crop_bbox)
            if words or chars or tables:
                logger.info(
                    "Removed %d words, %d chars, and %d OCR table regions before OCR.",
                    words,
                    chars,
                    tables,
                )
        created_elements = self._process_ocr_payload(
            host,
            ocr_payload,
            engine_name,
            resolved_min_conf,
            offset_x,
            offset_y,
            crop_bbox,
            detect_only=detect_only,
        )
        if created_elements is None:
            return host

        register_ocr_artifacts(*created_elements)
        if detect_only:
            self._refresh_detection_artifacts(host, crop_bbox, created_elements)

        logger.info("Added %d OCR elements using '%s'.", len(created_elements), engine_name)
        return host

    @register_delegate("ocr", "apply_custom_ocr")
    def apply_custom_ocr(
        self,
        host,
        *,
        ocr_function: Callable[[Any], Optional[str]],
        source_label: str = "custom-ocr",
        replace: OCRReplaceMode = "ocr",
        confidence: Optional[float] = None,
        add_to_page: bool = True,
    ):
        replace_mode = normalize_ocr_replace_mode(replace)
        if not callable(ocr_function):
            raise TypeError("ocr_function must be callable.")

        logger.debug("Running custom OCR function for %s", host)
        ocr_text = ocr_function(host)
        if ocr_text is not None and not isinstance(ocr_text, str):
            raise TypeError(
                f"Custom OCR function returned {type(ocr_text).__name__}; expected str or None."
            )

        if ocr_text is None or not ocr_text.strip():
            logger.debug("Custom OCR function returned no recognized text; no elements created.")
            return host

        to_text_element = getattr(host, "to_text_element", None)
        if not callable(to_text_element):
            raise AttributeError(
                f"{host.__class__.__name__} must implement to_text_element() for custom OCR."
            )

        if add_to_page and replace_mode != "none":
            self._remove_for_replace(
                host,
                replace_mode,
                self._host_bbox(host),
            )

        created = to_text_element(
            text_content=ocr_text,
            # Keep the public source category stable so selectors, correction,
            # removal, export, and subsequent engine OCR all see function OCR.
            source_label="ocr",
            confidence=confidence,
            add_to_page=add_to_page,
        )
        if created is not None:
            obj = getattr(created, "_obj", None)
            if isinstance(obj, dict):
                obj["ocr_engine"] = source_label
                obj["ocr_source_label"] = source_label
                obj["_natural_pdf_ocr_generated"] = True
            created._natural_pdf_ocr_generated = True
            if add_to_page:
                register_ocr_artifacts(created)
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
        apply_exclusions: bool = True,
        model: Optional[str] = None,
        client: Optional[Any] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        layout: Optional[bool | str] = None,
        preserve_markup: bool = False,
    ):
        normalized_options = normalize_ocr_options(options)
        scope = self._scope(host)
        engine_name = self._resolve_engine_name(
            host,
            engine,
            normalized_options,
            scope,
        )
        resolved_languages = resolve_ocr_languages(host, languages, scope=scope)
        resolved_min_conf = resolve_ocr_min_confidence(host, min_confidence, scope=scope)
        resolved_device = resolve_ocr_device(host, device, scope=scope)

        final_resolution = self._resolve_resolution(host, resolution, scope)
        render_kwargs = self._render_kwargs(host, apply_exclusions=apply_exclusions)
        crop_bbox = self._resolve_crop_bbox(host, render_kwargs)
        offset_x, offset_y = self._resolve_offsets(host, render_kwargs)

        ocr_payload = run_ocr(
            target=host,
            engine_name=engine_name,
            resolution=final_resolution,
            languages=resolved_languages,
            min_confidence=resolved_min_conf,
            device=resolved_device,
            detect_only=False,
            options=normalized_options,
            render_kwargs=render_kwargs,
            context=host,
            model=model,
            client=client,
            prompt=prompt,
            instructions=instructions,
            max_new_tokens=max_new_tokens,
            layout=layout,
            preserve_markup=preserve_markup,
        )

        self._normalize_payload_image_size(ocr_payload)

        if getattr(ocr_payload, "engine_type", None) != "vlm":
            self._normalize_classic_payload_results(
                ocr_payload,
                detection_only=False,
            )

        created = self._convert_ocr_payload(
            host,
            ocr_payload,
            engine_name,
            resolved_min_conf,
            offset_x,
            offset_y,
            crop_bbox,
        )
        return created or []
