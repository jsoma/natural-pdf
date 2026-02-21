import dataclasses
import logging
from typing import Any, List, Optional, Set

from natural_pdf import options as npdf_options
from natural_pdf.analyzers.layout.layout_manager import (
    engine_name_for_options,
    get_options_class_for_engine,
)
from natural_pdf.analyzers.layout.layout_options import (
    BaseLayoutOptions,
    DetectionContext,
    LayoutOptions,
    VLMLayoutOptions,
)
from natural_pdf.elements.region import Region
from natural_pdf.engine_provider import get_provider

logger = logging.getLogger(__name__)

_DEFAULT_ENGINE_WARNING_SHOWN: Set[str] = set()


class LayoutAnalyzer:
    """
    Handles layout analysis for PDF pages, including image rendering,
    coordinate scaling, region creation, and result storage.
    """

    def __init__(self, page, layout_manager: Optional[Any] = None):
        """Initialize the layout analyzer for a page."""
        self._page = page
        if layout_manager is not None:
            logger.warning(
                "layout_manager argument is deprecated and ignored; engines are now provided via EngineProvider."
            )
        self._engine_provider = get_provider()

    def analyze_layout(
        self,
        engine: Optional[str] = None,
        options: Optional[LayoutOptions] = None,
        confidence: Optional[float] = None,
        classes: Optional[List[str]] = None,
        exclude_classes: Optional[List[str]] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> List[Region]:
        """
        Analyze the page layout using the registered layout engine.

        This method constructs the final options object, including internal context,
        and passes it to the requested engine via the EngineProvider.

        Args:
            engine: Name of the layout engine (e.g., 'yolo', 'tatr'). Uses manager's default if None and no options object given.
            options: Specific LayoutOptions object for advanced configuration. If provided, simple args (confidence, etc.) are ignored.
            confidence: Minimum confidence threshold (simple mode).
            classes: Specific classes to detect (simple mode).
            exclude_classes: Classes to exclude (simple mode).
            device: Device for inference (simple mode).
            **kwargs: Additional engine-specific arguments (added to options.extra_args or used by constructor if options=None).

        Returns:
            List of created Region objects.
        """
        logger.info(
            f"Page {self._page.number}: Analyzing layout (Engine: {engine or 'default'}, Options provided: {options is not None})..."
        )

        # --- Render Page Image (Standard Resolution) ---
        logger.debug(
            f"  Rendering page {self._page.number} to image for initial layout detection..."
        )
        try:
            layout_resolution = self._page.get_config("layout_image_resolution", 72, scope="pdf")
            # Use render() for clean image without highlights
            std_res_page_image = self._page.render(resolution=layout_resolution)
            if not std_res_page_image:
                raise ValueError("Initial page rendering returned None")
            logger.debug(
                f"  Initial rendered image size: {std_res_page_image.width}x{std_res_page_image.height}"
            )
        except Exception as e:
            logger.error(f"  Failed to render initial page image: {e}", exc_info=True)
            return []

        # --- Calculate Scaling Factors (Standard Res Image <-> PDF) ---
        if std_res_page_image.width == 0 or std_res_page_image.height == 0:
            logger.error(
                f"Page {self._page.number}: Invalid initial rendered image dimensions. Cannot scale results."
            )
            return []
        img_scale_x = self._page.width / std_res_page_image.width
        img_scale_y = self._page.height / std_res_page_image.height
        logger.debug(f"  StdRes Image -> PDF Scaling: x={img_scale_x:.4f}, y={img_scale_y:.4f}")

        # --- Construct Final Options Object ---
        final_options: BaseLayoutOptions

        if options is not None:
            logger.debug("Using user-provided options object.")
            # Safe copy: dataclasses.replace() avoids deepcopy (which breaks
            # non-copyable fields like OpenAI clients) while isolating mutable state.
            final_options = dataclasses.replace(
                options,
                extra_args=dict(options.extra_args),
                classes=list(options.classes) if options.classes else None,
                exclude_classes=list(options.exclude_classes) if options.exclude_classes else None,
            )
            if kwargs:
                logger.warning(
                    f"Ignoring simple mode keyword arguments {list(kwargs.keys())} because a full options object was provided."
                )
            # Infer engine from options type if engine arg wasn't provided
            if engine is None:
                inferred = engine_name_for_options(final_options)
                if inferred:
                    engine = inferred
                    logger.debug(f"Inferred engine '{engine}' from options type.")
                else:
                    logger.warning("Could not infer engine from provided options object.")
        else:
            # Construct options from simple args (engine, confidence, classes, etc.)
            logger.debug("Constructing options from simple arguments.")
            selected_engine = engine or self._default_engine_name()
            if engine is None:
                self._warn_on_default_engine(selected_engine)
            engine_lower = selected_engine.lower()

            options_class = get_options_class_for_engine(engine_lower)
            if options_class is None:
                available = list(self._engine_provider.list("layout").get("layout", ()))
                raise ValueError(
                    f"Unknown or unavailable engine: '{selected_engine}'. Available: {available}"
                )

            # Get base defaults
            base_defaults = BaseLayoutOptions()

            # Separate client from other kwargs
            client_instance = kwargs.pop("client", None)  # Get client, remove from kwargs

            # Separate model_name/model if provided for VLM
            model_name_kwarg = None
            languages_kwarg = None
            if issubclass(options_class, VLMLayoutOptions):
                model_name_kwarg = kwargs.pop("model_name", None) or kwargs.pop("model", None)
                languages_kwarg = kwargs.pop("languages", None)

            # Prepare args for constructor, prioritizing explicit args over defaults
            constructor_args = {
                "confidence": confidence if confidence is not None else base_defaults.confidence,
                "classes": classes,  # Pass None if not provided
                "exclude_classes": exclude_classes,  # Pass None if not provided
                "device": device if device is not None else base_defaults.device,
                # Pass client explicitly if constructing VLM options
                **(
                    {"client": client_instance}
                    if client_instance and issubclass(options_class, VLMLayoutOptions)
                    else {}
                ),
                # Pass model_name explicitly if constructing VLM options and it was provided
                **(
                    {"model_name": model_name_kwarg}
                    if model_name_kwarg and issubclass(options_class, VLMLayoutOptions)
                    else {}
                ),
                # Pass languages if constructing VLM options and it was provided
                **(
                    {"languages": languages_kwarg}
                    if languages_kwarg is not None and issubclass(options_class, VLMLayoutOptions)
                    else {}
                ),
                "extra_args": kwargs,  # Pass REMAINING kwargs here
            }

            try:
                final_options = options_class(**constructor_args)
                logger.debug(f"Constructed options: {final_options}")
            except TypeError as e:
                logger.error(
                    f"Failed to construct options object {options_class.__name__} with args {constructor_args}: {e}"
                )
                raise e

            engine = selected_engine

        # --- Build Detection Context ---
        detection_context = DetectionContext(
            layout_host=self._page,
            img_scale_x=img_scale_x,
            img_scale_y=img_scale_y,
        )

        # --- Call Layout Manager (ALWAYS with options object) ---
        detections, detector = self._run_layout_engine(
            std_res_page_image, final_options, engine=engine, context=detection_context
        )
        if detections is None:
            return []

        # --- Apply Canonical Type Mapping ---
        type_map = getattr(detector, "TYPE_MAP", {})
        for detection in detections:
            engine_normalized = detection.get("normalized_class", "unknown")
            detection["canonical_type"] = type_map.get(engine_normalized, engine_normalized)

        # --- Process Detections (Convert to Regions, Scale Coords from Image to PDF) ---
        layout_regions = []

        for detection in detections:
            try:
                # bbox is relative to std_res_page_image
                x_min, y_min, x_max, y_max = detection["bbox"]

                # Convert coordinates from image to PDF space
                pdf_x0 = x_min * img_scale_x
                pdf_y0 = y_min * img_scale_y
                pdf_x1 = x_max * img_scale_x
                pdf_y1 = y_max * img_scale_y

                # Ensure PDF coords are valid
                pdf_x0, pdf_x1 = min(pdf_x0, pdf_x1), max(pdf_x0, pdf_x1)
                pdf_y0, pdf_y1 = min(pdf_y0, pdf_y1), max(pdf_y0, pdf_y1)
                pdf_x0 = max(0, pdf_x0)
                pdf_y0 = max(0, pdf_y0)
                pdf_x1 = min(self._page.width, pdf_x1)
                pdf_y1 = min(self._page.height, pdf_y1)

                # Create a Region object with PDF coordinates
                region = Region(self._page, (pdf_x0, pdf_y0, pdf_x1, pdf_y1))
                region.region_type = detection.get("class", "unknown")
                region.normalized_type = detection.get("canonical_type", "unknown")
                region.confidence = detection.get("confidence", 0.0)
                region.model = detection.get("model", engine or "unknown")
                region.source = "detected"

                # Add extra info if available
                if "text" in detection:
                    region.text_content = detection["text"]

                layout_regions.append(region)

            except (KeyError, IndexError, TypeError, ValueError) as e:
                logger.warning(f"Could not process layout detection: {detection}. Error: {e}")
                continue

        # --- Store Results ---
        logger.debug(f"Storing {len(layout_regions)} processed layout regions (mode: replace).")
        self._page.clear_detected_layout_regions()

        for region in layout_regions:
            self._page.add_region(region, source="detected")

        logger.info(f"Layout analysis complete for page {self._page.number}.")

        # --- Engine-specific post-processing (e.g., TATR cell creation) ---
        if detector is not None and hasattr(detector, "post_process_regions"):
            detector.post_process_regions(layout_regions, final_options)

        return layout_regions

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_engine_name(
        self, engine: Optional[str], options: BaseLayoutOptions
    ) -> Optional[str]:
        if engine:
            return engine.lower()

        inferred = engine_name_for_options(options)
        return inferred.lower() if inferred else None

    def _warn_on_default_engine(self, engine_name: str) -> None:
        normalized = (engine_name or "").lower()
        if not normalized or normalized == "yolo":
            return

        if normalized in _DEFAULT_ENGINE_WARNING_SHOWN:
            return

        available_engines = self._engine_provider.list("layout").get("layout", ())
        available = ", ".join(sorted(available_engines))
        logger.warning(
            "analyze_layout() called without specifying an engine; defaulting to '%s'. "
            "Call page.analyze_layout('yolo') (or another engine name) or use natural_pdf.set_option('layout.engine', '<engine>') "
            "to choose explicitly. Available engines: %s",
            engine_name,
            available,
        )
        _DEFAULT_ENGINE_WARNING_SHOWN.add(normalized)

    def _run_layout_engine(self, image, options, engine: Optional[str], context=None):
        engine_name = self._resolve_engine_name(engine, options)
        if engine_name is None:
            logger.error("Unable to determine layout engine for provided options")
            return None, None

        try:
            detector = self._engine_provider.get("layout", context=self._page, name=engine_name)
            detections = detector.detect(image, options, context=context)
            logger.info(
                "  Layout engine '%s' returned %d detections.",
                engine_name,
                len(detections),
            )
            return detections, detector
        except LookupError as provider_err:
            raise RuntimeError(
                f"Layout engine '{engine_name}' is not registered: {provider_err}"
            ) from provider_err
        except Exception as exc:
            logger.error(
                "Layout engine '%s' failed via provider: %s", engine_name, exc, exc_info=True
            )
            raise

    def _default_engine_name(self) -> str:
        config_engine = self._page.get_config("layout_engine", None, scope="page")
        if isinstance(config_engine, str):
            config_engine = config_engine.strip().lower()

        global_default = getattr(npdf_options.layout, "engine", None)
        if isinstance(global_default, str):
            global_default = global_default.strip().lower()

        available = tuple(self._engine_provider.list("layout").get("layout", ()))
        if not available:
            raise RuntimeError("No layout engines are registered.")

        for candidate in (config_engine, global_default):
            if candidate and candidate in available:
                return candidate

        return available[0]
