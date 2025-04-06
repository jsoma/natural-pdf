import logging
from typing import List, Dict, Any, Optional, Union
from PIL import Image
import copy

from natural_pdf.elements.region import Region
from natural_pdf.analyzers.layout.layout_manager import LayoutManager
from natural_pdf.analyzers.layout.layout_options import LayoutOptions, TATRLayoutOptions, BaseLayoutOptions

logger = logging.getLogger(__name__)

class LayoutAnalyzer:
    """
    Handles layout analysis for PDF pages, including image rendering,
    coordinate scaling, region creation, and result storage.
    """
    
    def __init__(self, page, layout_manager: Optional[LayoutManager] = None):
        """
        Initialize the layout analyzer.
        
        Args:
            page: The Page object to analyze
            layout_manager: Optional LayoutManager instance. If None, will try to get from page's parent.
        """
        self._page = page
        self._layout_manager = layout_manager or getattr(page._parent, '_layout_manager', None)
        
        if not self._layout_manager:
            logger.warning(f"LayoutManager not available for page {page.number}. Layout analysis will fail.")
    
    def analyze_layout(
        self,
        engine: Optional[str] = None,
        options: Optional[LayoutOptions] = None,
        confidence: Optional[float] = None,
        classes: Optional[List[str]] = None,
        exclude_classes: Optional[List[str]] = None,
        device: Optional[str] = None,
        existing: str = "replace",
        **kwargs
    ) -> List[Region]:
        """
        Analyze the page layout using the configured LayoutManager.
        
        This method constructs the final options object, including internal context, 
        and passes it to the LayoutManager.
        
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
        if not self._layout_manager:
            logger.error(f"Page {self._page.number}: LayoutManager not available. Cannot analyze layout.")
            return []

        logger.info(f"Page {self._page.number}: Analyzing layout (Engine: {engine or 'default'}, Options provided: {options is not None})...")

        # --- Render Page Image (Standard Resolution) ---
        logger.debug(f"  Rendering page {self._page.number} to image for initial layout detection...")
        try:
            layout_scale = getattr(self._page._parent, '_config', {}).get('layout_image_scale', 1.5)
            layout_resolution = layout_scale * 72
            std_res_page_image = self._page.to_image(resolution=layout_resolution, include_highlights=False)
            if not std_res_page_image:
                raise ValueError("Initial page rendering returned None")
            logger.debug(f"  Initial rendered image size: {std_res_page_image.width}x{std_res_page_image.height}")
        except Exception as e:
            logger.error(f"  Failed to render initial page image: {e}", exc_info=True)
            return []
            
        # --- Calculate Scaling Factors (Standard Res Image <-> PDF) ---
        if std_res_page_image.width == 0 or std_res_page_image.height == 0:
            logger.error(f"Page {self._page.number}: Invalid initial rendered image dimensions. Cannot scale results.")
            return []
        img_scale_x = self._page.width / std_res_page_image.width
        img_scale_y = self._page.height / std_res_page_image.height
        logger.debug(f"  StdRes Image -> PDF Scaling: x={img_scale_x:.4f}, y={img_scale_y:.4f}")

        # --- Construct Final Options Object --- 
        final_options: BaseLayoutOptions
        
        if options is not None:
             # User provided a complete options object, use it directly
             logger.debug("Using user-provided options object.")
             final_options = copy.deepcopy(options) # Copy to avoid modifying original user object
             if kwargs:
                  logger.warning(f"Ignoring kwargs {list(kwargs.keys())} because a full options object was provided.")
             # Infer engine from options type if engine arg wasn't provided
             if engine is None:
                  for name, registry_entry in self._layout_manager.ENGINE_REGISTRY.items():
                       if isinstance(final_options, registry_entry['options_class']):
                            engine = name
                            logger.debug(f"Inferred engine '{engine}' from options type.")
                            break
                  if engine is None:
                       logger.warning("Could not infer engine from provided options object.")
        else:
             # Construct options from simple args (engine, confidence, classes, etc.)
             logger.debug("Constructing options from simple arguments.")
             selected_engine = engine or self._layout_manager.get_available_engines()[0] # Use provided or first available
             engine_lower = selected_engine.lower()
             registry = self._layout_manager.ENGINE_REGISTRY
             
             if engine_lower not in registry:
                  raise ValueError(f"Unknown or unavailable engine: '{selected_engine}'. Available: {list(registry.keys())}")
                  
             options_class = registry[engine_lower]['options_class']
             
             # Get base defaults
             base_defaults = BaseLayoutOptions()
             
             # Prepare args for constructor, prioritizing explicit args over defaults
             constructor_args = {
                 'confidence': confidence if confidence is not None else base_defaults.confidence,
                 'classes': classes, # Pass None if not provided
                 'exclude_classes': exclude_classes, # Pass None if not provided
                 'device': device if device is not None else base_defaults.device,
                 'extra_args': kwargs # Pass other kwargs here
             }
             # Remove None values unless they are valid defaults (like classes=None)
             # We can pass all to the dataclass constructor; it handles defaults
             
             try:
                  final_options = options_class(**constructor_args)
                  logger.debug(f"Constructed options: {final_options}")
             except TypeError as e:
                  logger.error(f"Failed to construct options object {options_class.__name__} with args {constructor_args}: {e}")
                  # Filter kwargs to only include fields defined in the specific options class? Complex.
                  # Re-raise for now, indicates programming error or invalid kwarg.
                  raise e

        # --- Add Internal Context to extra_args (ALWAYS) --- 
        if not hasattr(final_options, 'extra_args') or final_options.extra_args is None:
             final_options.extra_args = {}
        final_options.extra_args['_page_ref'] = self._page
        final_options.extra_args['_img_scale_x'] = img_scale_x
        final_options.extra_args['_img_scale_y'] = img_scale_y
        logger.debug(f"Added internal context to final_options.extra_args: {final_options.extra_args}")

        # --- Call Layout Manager with the Final Options --- 
        logger.debug(f"Calling Layout Manager with final options object.")
        try:
            # Pass only image and the constructed options object
            detections = self._layout_manager.analyze_layout(
                image=std_res_page_image, 
                options=final_options
                # No engine, confidence, classes etc. passed here directly
            )
            logger.info(f"  Layout Manager returned {len(detections)} detections.")
        except Exception as e:
            logger.error(f"  Layout analysis failed: {e}", exc_info=True)
            return []

        # --- Process Detections (Convert to Regions, Scale Coords from Image to PDF) ---
        layout_regions = []
        docling_id_to_region = {} # For hierarchy if using Docling

        for detection in detections:
            try:
                # bbox is relative to std_res_page_image
                x_min, y_min, x_max, y_max = detection['bbox']

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
                region.region_type = detection.get('class', 'unknown')
                region.normalized_type = detection.get('normalized_class', 'unknown')
                region.confidence = detection.get('confidence', 0.0)
                region.model = detection.get('model', engine or 'unknown')
                region.source = 'detected'
                
                # Add extra info if available
                if 'text' in detection: region.text_content = detection['text']
                if 'docling_id' in detection: region.docling_id = detection['docling_id']
                if 'parent_id' in detection: region.parent_id = detection['parent_id']

                layout_regions.append(region)

                # Track Docling IDs for hierarchy
                if hasattr(region, 'docling_id') and region.docling_id:
                    docling_id_to_region[region.docling_id] = region

            except (KeyError, IndexError, TypeError, ValueError) as e:
                logger.warning(f"Could not process layout detection: {detection}. Error: {e}")
                continue

        # --- Build Hierarchy (if Docling results detected) ---
        if docling_id_to_region:
            logger.debug("Building Docling region hierarchy...")
            for region in layout_regions:
                if hasattr(region, 'parent_id') and region.parent_id:
                    parent_region = docling_id_to_region.get(region.parent_id)
                    if parent_region:
                        if hasattr(parent_region, 'add_child'):
                            parent_region.add_child(region)
                        else:
                            logger.warning("Region object missing add_child method for hierarchy.")

        # --- Store Results ---
        logger.debug(f"Storing {len(layout_regions)} processed layout regions (mode: {existing}).")
        # Handle existing regions based on mode
        if existing.lower() == 'append':
            if 'detected' not in self._page._regions: self._page._regions['detected'] = []
            self._page._regions['detected'].extend(layout_regions)
        else: # Default is 'replace'
            self._page._regions['detected'] = layout_regions

        # Add regions to the element manager
        for region in layout_regions:
            self._page._element_mgr.add_region(region)

        # Store layout regions in a dedicated attribute for easier access
        self._page.detected_layout_regions = self._page._regions['detected']
        logger.info(f"Layout analysis complete for page {self._page.number}.")
        
        # --- Auto-create cells if requested by TATR options ---
        if isinstance(final_options, TATRLayoutOptions) and final_options.create_cells:
            logger.info(f"  Option create_cells=True detected for TATR. Attempting cell creation...")
            created_cell_count = 0
            for region in layout_regions:
                # Only attempt on regions identified as tables by the TATR model
                if region.model == 'tatr' and region.region_type == 'table':
                    try:
                        # create_cells now modifies the page elements directly and returns self
                        region.create_cells()
                        # We could potentially count cells created here if needed, 
                        # but the method logs its own count.
                    except Exception as cell_error:
                        logger.warning(f"    Error calling create_cells for table region {region.bbox}: {cell_error}")
            logger.info(f"  Finished cell creation process triggered by options.")
        
        return layout_regions 