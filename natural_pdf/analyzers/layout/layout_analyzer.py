import copy
import logging
from typing import Any, List, Optional

from natural_pdf import options as npdf_options
from natural_pdf.analyzers.layout.layout_manager import ENGINE_REGISTRY
from natural_pdf.analyzers.layout.layout_options import (
    BaseLayoutOptions,
    GeminiLayoutOptions,
    LayoutOptions,
    TATRLayoutOptions,
)
from natural_pdf.elements.region import Region
from natural_pdf.engine_provider import get_provider

logger = logging.getLogger(__name__)


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
        existing: str = "replace",
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
            existing: How to handle existing detected regions: 'replace' (default) or 'append'.
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
            final_options = copy.deepcopy(options)  # Copy to avoid modifying original user object
            if kwargs:
                logger.warning(
                    f"Ignoring simple mode keyword arguments {list(kwargs.keys())} because a full options object was provided."
                )
            # Infer engine from options type if engine arg wasn't provided
            if engine is None:
                for name, registry_entry in ENGINE_REGISTRY.items():
                    if isinstance(final_options, registry_entry["options_class"]):
                        engine = name
                        logger.debug(f"Inferred engine '{engine}' from options type.")
                        break
                if engine is None:
                    logger.warning("Could not infer engine from provided options object.")
        else:
            # Construct options from simple args (engine, confidence, classes, etc.)
            logger.debug("Constructing options from simple arguments.")
            selected_engine = engine or self._default_engine_name()
            engine_lower = selected_engine.lower()
            registry = ENGINE_REGISTRY

            if engine_lower not in registry:
                raise ValueError(
                    f"Unknown or unavailable engine: '{selected_engine}'. Available: {list(registry.keys())}"
                )

            options_class = registry[engine_lower]["options_class"]

            # Get base defaults
            base_defaults = BaseLayoutOptions()

            # Separate client from other kwargs
            client_instance = kwargs.pop("client", None)  # Get client, remove from kwargs

            # Separate model_name if provided for Gemini
            model_name_kwarg = None
            if issubclass(options_class, GeminiLayoutOptions):
                model_name_kwarg = kwargs.pop("model_name", None)

            # Prepare args for constructor, prioritizing explicit args over defaults
            constructor_args = {
                "confidence": confidence if confidence is not None else base_defaults.confidence,
                "classes": classes,  # Pass None if not provided
                "exclude_classes": exclude_classes,  # Pass None if not provided
                "device": device if device is not None else base_defaults.device,
                # Pass client explicitly if constructing Gemini options
                # Note: We check issubclass *before* calling constructor
                **(
                    {"client": client_instance}
                    if client_instance and issubclass(options_class, GeminiLayoutOptions)
                    else {}
                ),
                # Pass model_name explicitly if constructing Gemini options and it was provided
                **(
                    {"model_name": model_name_kwarg}
                    if model_name_kwarg and issubclass(options_class, GeminiLayoutOptions)
                    else {}
                ),
                "extra_args": kwargs,  # Pass REMAINING kwargs here
            }
            # Remove None values unless they are valid defaults (like classes=None)
            # We can pass all to the dataclass constructor; it handles defaults
            # **Filter constructor_args to remove None values that aren't defaults?**
            # For simplicity, let dataclass handle it for now.

            try:
                final_options = options_class(**constructor_args)
                logger.debug(f"Constructed options: {final_options}")
            except TypeError as e:
                logger.error(
                    f"Failed to construct options object {options_class.__name__} with args {constructor_args}: {e}"
                )
                # Filter kwargs to only include fields defined in the specific options class? Complex.
                # Re-raise for now, indicates programming error or invalid kwarg.
                raise e

            engine = selected_engine

        # --- Add Internal Context to extra_args (Applies to the final_options object) ---
        if not hasattr(final_options, "extra_args") or final_options.extra_args is None:
            # Ensure extra_args exists, potentially overwriting if needed
            final_options.extra_args = {}
        elif not isinstance(final_options.extra_args, dict):
            logger.warning(
                f"final_options.extra_args was not a dict ({type(final_options.extra_args)}), replacing with internal context."
            )
            final_options.extra_args = {}

        final_options.extra_args["_layout_host"] = self._page
        final_options.extra_args["_page_ref"] = self._page  # Backwards compatibility
        final_options.extra_args["_img_scale_x"] = img_scale_x
        final_options.extra_args["_img_scale_y"] = img_scale_y
        logger.debug(
            f"Added/updated internal context in final_options.extra_args: {final_options.extra_args}"
        )

        # --- Call Layout Manager (ALWAYS with options object) ---
        detections = self._run_layout_engine(std_res_page_image, final_options, engine=engine)
        if detections is None:
            return []

        # --- Process Detections (Convert to Regions, Scale Coords from Image to PDF) ---
        layout_regions = []
        docling_id_to_region = {}  # For hierarchy if using Docling

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
                region.normalized_type = detection.get("normalized_class", "unknown")
                region.confidence = detection.get("confidence", 0.0)
                region.model = detection.get("model", engine or "unknown")
                region.source = "detected"

                # Add extra info if available
                if "text" in detection:
                    region.text_content = detection["text"]

                docling_id = detection.get("docling_id")
                parent_id = detection.get("parent_id")
                if docling_id is not None or parent_id is not None:
                    docling_meta = region.metadata.setdefault("docling", {})
                    if docling_id is not None:
                        docling_meta["id"] = docling_id
                        docling_id_to_region[docling_id] = region
                    if parent_id is not None:
                        docling_meta["parent_id"] = parent_id

                layout_regions.append(region)

            except (KeyError, IndexError, TypeError, ValueError) as e:
                logger.warning(f"Could not process layout detection: {detection}. Error: {e}")
                continue

        # --- Build Hierarchy (if Docling results detected) ---
        if docling_id_to_region:
            logger.debug("Building Docling region hierarchy...")
            for region in layout_regions:
                docling_meta = region.metadata.get("docling", {})
                parent_id = docling_meta.get("parent_id")
                if parent_id:
                    parent_region = docling_id_to_region.get(parent_id)
                    if parent_region:
                        if hasattr(parent_region, "add_child"):
                            parent_region.add_child(region)
                        else:
                            logger.warning("Region object missing add_child method for hierarchy.")

        # --- Store Results ---
        logger.debug(f"Storing {len(layout_regions)} processed layout regions (mode: {existing}).")
        append_mode = existing.lower() == "append"
        if not append_mode:
            self._page.clear_detected_layout_regions()

        for region in layout_regions:
            self._page.add_region(region, source="detected")

        # Store layout regions in a dedicated attribute for easier access
        self._page.detected_layout_regions = self._page._regions.get("detected", [])
        logger.info(f"Layout analysis complete for page {self._page.number}.")

        # --- Auto-create cells if requested by TATR options ---
        if isinstance(final_options, TATRLayoutOptions) and final_options.create_cells:
            logger.info("  Option create_cells=True detected for TATR. Attempting cell creation...")
            created_cell_count = 0
            for region in layout_regions:
                # Only attempt on regions identified as tables by the TATR model
                if region.model == "tatr" and region.region_type == "table":
                    try:
                        # create_cells now modifies the page elements directly and returns self
                        region.create_cells()
                        # We could potentially count cells created here if needed,
                        # but the method logs its own count.
                    except Exception as cell_error:
                        logger.warning(
                            f"    Error calling create_cells for table region {region.bbox}: {cell_error}"
                        )
            logger.info("  Finished cell creation process triggered by options.")

        return layout_regions

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_engine_name(
        self, engine: Optional[str], options: BaseLayoutOptions
    ) -> Optional[str]:
        if engine:
            return engine.lower()

        for name, entry in ENGINE_REGISTRY.items():
            if isinstance(options, entry["options_class"]):
                return name.lower()
        return None

    def _run_layout_engine(self, image, options, engine: Optional[str]):
        engine_name = self._resolve_engine_name(engine, options)
        if engine_name is None:
            logger.error("Unable to determine layout engine for provided options")
            return None

        try:
            detector = self._engine_provider.get("layout", context=self._page, name=engine_name)
            detections = detector.detect(image, options)
            logger.info(
                "  Layout engine '%s' returned %d detections.",
                engine_name,
                len(detections),
            )
            return detections
        except LookupError as provider_err:
            raise RuntimeError(
                f"Layout engine '{engine_name}' is not registered: {provider_err}"
            ) from provider_err
        except Exception as exc:
            logger.error(
                "Layout engine '%s' failed via provider: %s", engine_name, exc, exc_info=True
            )
            return None

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
