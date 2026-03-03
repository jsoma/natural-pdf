"""Unified checkbox detection orchestrator.

Handles engine selection (auto or explicit), image rendering,
coordinate conversion, classification, and region storage.
"""

import dataclasses
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from natural_pdf.elements.region import Region
from natural_pdf.engine_provider import get_provider

from .base import DetectionContext
from .checkbox_manager import engine_name_for_options, get_options_class_for_engine
from .checkbox_options import BaseCheckboxOptions
from .classifier import CheckboxClassifier

logger = logging.getLogger(__name__)

# Unicode ballot box characters
_UNICODE_CHECKED = frozenset("\u2611\u2612")  # ☑ ☒
_UNICODE_UNCHECKED = frozenset("\u2610")  # ☐
_UNICODE_CHECKBOXES = _UNICODE_CHECKED | _UNICODE_UNCHECKED


class CheckboxAnalyzer:
    """Unified orchestrator for checkbox detection."""

    def __init__(self, page):
        self._page = page
        self._provider = get_provider()

    def detect_checkboxes(
        self,
        engine: Optional[str] = None,
        options: Optional[Union[BaseCheckboxOptions, Dict[str, Any]]] = None,
        confidence: Optional[float] = None,
        resolution: Optional[int] = None,
        device: Optional[str] = None,
        existing: str = "replace",
        limit: Optional[int] = None,
        classify: Optional[bool] = None,
        classify_with: Optional[Any] = None,
        **kwargs,
    ) -> List[Region]:
        """Detect checkboxes on the page.

        Args:
            engine: Engine name or "auto" (default).
            options: Full options object or dict.
            confidence: Minimum confidence threshold.
            resolution: DPI for rendering.
            device: Device for inference.
            existing: "replace" or "append".
            limit: Max checkboxes to return.
            classify: Run classification (default True).
            classify_with: Judge instance for classification.
            **kwargs: Engine-specific arguments.

        Returns:
            List of Region objects.
        """
        # Build final options
        final_options = self._build_options(
            engine,
            options,
            confidence,
            resolution,
            device,
            classify,
            classify_with,
            **kwargs,
        )
        engine_name = engine or "auto"

        if limit is None:
            limit = getattr(final_options, "limit", None)

        logger.info("Detecting checkboxes (engine=%s)", engine_name)

        # Remove old checkbox regions before detection so their alt_text
        # doesn't interfere with the text-rejection filter on new candidates.
        if existing.lower() == "replace":
            self._page.remove_regions(source="checkbox", region_type="checkbox")

        if engine_name == "auto":
            detections = self._detect_auto(final_options)
        else:
            detections = self._detect_with_engine(engine_name, final_options)

        if not detections:
            logger.info("No checkboxes detected")
            return self._store_results([], existing)

        # Convert image-space detections to PDF coordinates
        regions = self._convert_to_regions(detections, final_options)

        # Text rejection filter
        if getattr(final_options, "reject_with_text", True):
            regions = self._filter_text_regions(regions)

        # Classification
        do_classify = getattr(final_options, "classify", True) if classify is None else classify
        judge = classify_with or getattr(final_options, "classify_with", None)

        # Force classification for detection-only engines (is_checked=None)
        has_unknown = any(getattr(r, "checkbox_state", "unknown") == "unknown" for r in regions)
        if has_unknown or do_classify or judge is not None:
            CheckboxClassifier.classify_regions(
                regions,
                self._page,
                judge=judge,
                classify=do_classify or has_unknown,
            )

        # Set alt_text based on classification result
        from natural_pdf import options as _npdf_options

        _alt_cfg = getattr(_npdf_options, "alt_text", None)
        for region in regions:
            if region.is_checked is True:
                region.alt_text = getattr(_alt_cfg, "checkbox_checked", "[CHECKED]")
            elif region.is_checked is False:
                region.alt_text = getattr(_alt_cfg, "checkbox_unchecked", "[UNCHECKED]")
            # is_checked=None → leave alt_text as None

        # Detect Unicode checkbox characters
        unicode_regions = self._detect_unicode_checkboxes()
        if unicode_regions:
            # Dedup against visual results
            unicode_regions = self._dedup_unicode(unicode_regions, regions)
            # Set alt_text on Unicode regions
            for region in unicode_regions:
                if region.is_checked is True:
                    region.alt_text = getattr(_alt_cfg, "checkbox_checked", "[CHECKED]")
                elif region.is_checked is False:
                    region.alt_text = getattr(_alt_cfg, "checkbox_unchecked", "[UNCHECKED]")
            regions.extend(unicode_regions)

        # Apply limit
        if limit is not None and len(regions) > limit:
            regions = sorted(regions, key=lambda r: r.confidence, reverse=True)[:limit]

        return self._store_results(regions, existing)

    def _detect_auto(self, options: BaseCheckboxOptions) -> List[Dict[str, Any]]:
        """Auto-strategy: vector first for native PDFs, ONNX model for scanned."""
        page_type = self._probe_page_type()

        # Native PDFs: try vector detection first (instant, no rendering)
        if page_type in ("vector", "mixed"):
            vector_results = self._try_engine("vector", options)
            if vector_results:
                return vector_results

        # Scanned pages or vector found nothing: try ONNX model
        if self._engine_available("default"):
            results = self._try_engine("default", options)
            if results:
                return results

        logger.warning("No checkbox engine available or produced results")
        return []

    def _detect_with_engine(
        self, engine_name: str, options: BaseCheckboxOptions
    ) -> List[Dict[str, Any]]:
        """Run a specific engine."""
        return self._try_engine(engine_name.lower(), options) or []

    def _try_engine(
        self, engine_name: str, options: BaseCheckboxOptions
    ) -> Optional[List[Dict[str, Any]]]:
        """Try running an engine, return None on failure."""
        try:
            detector = self._provider.get("checkbox", context=self._page, name=engine_name)
        except (LookupError, RuntimeError) as e:
            logger.debug("Engine '%s' not available: %s", engine_name, e)
            return None

        # Convert to engine-specific options before rendering so the
        # engine's defaults (e.g. resolution, sahi_enabled) take effect.
        # Only copy fields that differ from BaseCheckboxOptions defaults,
        # so that engine-specific defaults aren't overridden by base defaults.
        opts_class = get_options_class_for_engine(engine_name)
        if opts_class and not isinstance(options, opts_class):
            override_fields = {}
            for f in dataclasses.fields(BaseCheckboxOptions):
                val = getattr(options, f.name, None)
                default = f.default if f.default is not dataclasses.MISSING else None
                if val != default:
                    override_fields[f.name] = val
            try:
                options = opts_class(**override_fields)
            except TypeError:
                pass

        try:
            # Build detection context
            if engine_name == "vector":
                # Vector detector doesn't need image rendering
                context = DetectionContext(
                    page=self._page,
                    img_scale_x=1.0,
                    img_scale_y=1.0,
                )
                # Pass a dummy image (vector detector ignores it)
                from PIL import Image

                dummy = Image.new("RGB", (1, 1))
                return detector.detect(dummy, options, context)
            else:
                # Render page image
                resolution = getattr(options, "resolution", 150)
                image = self._page.render(resolution=resolution)
                if image is None:
                    logger.error("Page rendering returned None")
                    return None

                img_scale_x = self._page.width / image.width
                img_scale_y = self._page.height / image.height

                context = DetectionContext(
                    page=self._page,
                    img_scale_x=img_scale_x,
                    img_scale_y=img_scale_y,
                )
                detections = detector.detect(image, options, context)

                # Stamp actual scale factors so _convert_to_regions
                # doesn't need to re-render at a potentially different DPI.
                if detections:
                    for det in detections:
                        det["_img_scale_x"] = img_scale_x
                        det["_img_scale_y"] = img_scale_y

                return detections

        except Exception as e:
            logger.error("Engine '%s' failed: %s", engine_name, e, exc_info=True)
            return None

    def _engine_available(self, engine_name: str) -> bool:
        """Check if an engine is registered and available."""
        try:
            detector = self._provider.get("checkbox", context=self._page, name=engine_name)
            return detector.is_available()
        except (LookupError, RuntimeError):
            return False

    def _probe_page_type(self) -> str:
        """Cheap page type probe."""
        has_text = bool(self._page.find("text"))
        has_rects = bool(self._page.find("rect"))
        if has_text and has_rects:
            return "vector"
        elif not has_text and not has_rects:
            return "scanned"
        return "mixed"

    def _convert_to_regions(
        self, detections: List[Dict[str, Any]], options: BaseCheckboxOptions
    ) -> List[Region]:
        """Convert canonical detections to Region objects."""
        resolution = getattr(options, "resolution", 150)
        regions = []

        for det in detections:
            try:
                bbox = det["bbox"]
                coord_space = det.get("coord_space", "image")

                if coord_space == "pdf":
                    pdf_x0, pdf_y0, pdf_x1, pdf_y1 = bbox
                elif coord_space == "image":
                    # Convert image coords to PDF coords using the scale
                    # factors from the actual rendered image (stamped by
                    # _try_engine), falling back to re-rendering.
                    scale_x = det.get("_img_scale_x")
                    scale_y = det.get("_img_scale_y")
                    if scale_x is None or scale_y is None:
                        image = self._page.render(resolution=resolution)
                        if image is None:
                            continue
                        scale_x = self._page.width / image.width
                        scale_y = self._page.height / image.height
                    pdf_x0 = bbox[0] * scale_x
                    pdf_y0 = bbox[1] * scale_y
                    pdf_x1 = bbox[2] * scale_x
                    pdf_y1 = bbox[3] * scale_y
                else:
                    logger.warning("Unknown coord_space: %s", coord_space)
                    continue

                # Ensure valid bounds
                pdf_x0, pdf_x1 = min(pdf_x0, pdf_x1), max(pdf_x0, pdf_x1)
                pdf_y0, pdf_y1 = min(pdf_y0, pdf_y1), max(pdf_y0, pdf_y1)
                pdf_x0 = max(0, pdf_x0)
                pdf_y0 = max(0, pdf_y0)
                pdf_x1 = min(self._page.width, pdf_x1)
                pdf_y1 = min(self._page.height, pdf_y1)

                # Create region
                region = self._page.create_region(pdf_x0, pdf_y0, pdf_x1, pdf_y1)
                region.region_type = "checkbox"
                region.normalized_type = "checkbox"
                region.is_checked = det.get("is_checked")
                region.checkbox_state = det.get("checkbox_state", "unknown")
                region.confidence = float(det.get("confidence", 0.0))
                region.model = det.get("engine", "checkbox_detector")
                region.source = "checkbox"

                region.analyses["checkbox"] = {
                    "is_checked": region.is_checked,
                    "state": region.checkbox_state,
                    "confidence": region.confidence,
                    "model": region.model,
                    "engine": det.get("engine", "unknown"),
                }

                regions.append(region)

            except Exception as e:
                logger.warning("Could not process detection: %s. Error: %s", det, e)
                continue

        return regions

    def _filter_text_regions(self, regions: List[Region]) -> List[Region]:
        """Remove regions that contain significant text."""
        filtered = []
        for region in regions:
            try:
                text = region.extract_text().strip()
                if text:
                    # Allow single characters that might be check marks
                    if len(text) > 1 or text.isalnum():
                        logger.debug(
                            "Rejecting checkbox at %s - contains text: '%s'",
                            region.bbox,
                            text,
                        )
                        continue
            except Exception:
                pass
            filtered.append(region)
        return filtered

    def _store_results(self, regions: List[Region], existing: str) -> List[Region]:
        """Store regions on the page and return them.

        Note: for existing="replace", old regions are already removed
        at the start of detect() so alt_text doesn't interfere with
        the text-rejection filter.
        """
        for region in regions:
            self._page.add_region(region, source="checkbox")

        # Update analysis snapshot
        append_mode = existing.lower() == "append"
        checkbox_analysis = self._page.analyses.setdefault("checkbox", {})
        prior = list(checkbox_analysis.get("regions", [])) if append_mode else []
        checkbox_analysis["regions"] = prior + list(regions)

        logger.info("Checkbox detection complete. Found %d checkboxes.", len(regions))
        return regions

    def _detect_unicode_checkboxes(self) -> List[Region]:
        """Scan page words for Unicode ballot box characters."""
        regions = []
        for word in self._page.words:
            char = word.text.strip()
            if len(char) != 1 or char not in _UNICODE_CHECKBOXES:
                continue
            checked = char in _UNICODE_CHECKED
            region = self._page.create_region(word.x0, word.top, word.x1, word.bottom)
            region.region_type = "checkbox"
            region.normalized_type = "checkbox"
            region.is_checked = checked
            region.checkbox_state = "checked" if checked else "unchecked"
            region.confidence = 1.0
            region.model = "unicode"
            region.source = "checkbox"
            region.analyses["checkbox"] = {
                "is_checked": checked,
                "state": region.checkbox_state,
                "confidence": 1.0,
                "model": "unicode",
                "engine": "unicode",
            }
            regions.append(region)
        return regions

    def _dedup_unicode(
        self, unicode_regions: List[Region], visual_regions: List[Region]
    ) -> List[Region]:
        """Remove Unicode regions that overlap with existing visual detections."""
        if not visual_regions:
            return unicode_regions
        kept = []
        for ur in unicode_regions:
            ucx = (ur.x0 + ur.x1) / 2
            ucy = (ur.top + ur.bottom) / 2
            duplicate = False
            for vr in visual_regions:
                if vr.x0 <= ucx <= vr.x1 and vr.top <= ucy <= vr.bottom:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(ur)
        return kept

    def _build_options(
        self,
        engine: Optional[str],
        options: Optional[Union[BaseCheckboxOptions, Dict[str, Any]]],
        confidence: Optional[float],
        resolution: Optional[int],
        device: Optional[str],
        classify: Optional[bool],
        classify_with: Optional[Any],
        **kwargs,
    ) -> BaseCheckboxOptions:
        """Build final options from various inputs."""
        if isinstance(options, BaseCheckboxOptions):
            # Clone and override
            overrides = {}
            if confidence is not None:
                overrides["confidence"] = confidence
            if resolution is not None:
                overrides["resolution"] = resolution
            if device is not None:
                overrides["device"] = device
            if classify is not None:
                overrides["classify"] = classify
            if classify_with is not None:
                overrides["classify_with"] = classify_with
            if overrides:
                return dataclasses.replace(options, **overrides)
            return options

        # Build from scratch
        base_kwargs: Dict[str, Any] = {}
        if isinstance(options, dict):
            base_kwargs.update(options)
        base_kwargs.update(kwargs)

        # Map user-facing kwarg names to options fields
        if "model" in base_kwargs:
            base_kwargs["model_repo"] = base_kwargs.pop("model")

        if confidence is not None:
            base_kwargs["confidence"] = confidence
        if resolution is not None:
            base_kwargs["resolution"] = resolution
        if device is not None:
            base_kwargs["device"] = device
        if classify is not None:
            base_kwargs["classify"] = classify
        if classify_with is not None:
            base_kwargs["classify_with"] = classify_with

        # If engine specified, try to use its options class
        if engine and engine != "auto":
            opts_class = get_options_class_for_engine(engine)
            if opts_class:
                # Filter kwargs to only valid fields for this class
                valid_fields = {f.name for f in dataclasses.fields(opts_class)}
                filtered = {k: v for k, v in base_kwargs.items() if k in valid_fields}
                extra = {k: v for k, v in base_kwargs.items() if k not in valid_fields}
                if extra:
                    filtered["extra_args"] = extra
                try:
                    return opts_class(**filtered)
                except TypeError:
                    pass

        return BaseCheckboxOptions(
            **{
                k: v
                for k, v in base_kwargs.items()
                if k in {f.name for f in dataclasses.fields(BaseCheckboxOptions)}
            }
        )
