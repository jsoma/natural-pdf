import logging
from typing import List, Dict, Any, Optional, Union
from PIL import Image

from natural_pdf.elements.region import Region
from natural_pdf.analyzers.layout.layout_manager import LayoutManager
from natural_pdf.analyzers.layout.layout_options import LayoutOptions

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
        existing: str = "replace"
    ) -> List[Region]:
        """
        Analyze the page layout using the configured LayoutManager.
        
        Args:
            engine: Name of the layout engine (e.g., 'yolo', 'tatr'). Uses manager's default if None.
            options: Specific LayoutOptions object for advanced configuration.
            confidence: Minimum confidence threshold (simple mode).
            classes: Specific classes to detect (simple mode).
            exclude_classes: Classes to exclude (simple mode).
            device: Device for inference (simple mode).
            existing: How to handle existing detected regions: 'replace' (default) or 'append'.
            
        Returns:
            List of created Region objects.
        """
        if not self._layout_manager:
            logger.error(f"Page {self._page.number}: LayoutManager not available. Cannot analyze layout.")
            return []

        logger.info(f"Page {self._page.number}: Analyzing layout (Engine: {engine or 'default'}, Options: {options is not None})...")

        # --- Render Page Image ---
        logger.debug(f"  Rendering page {self._page.number} to image for layout analysis...")
        try:
            # Use a resolution suitable for layout analysis, potentially configurable
            layout_scale = getattr(self._page._parent, '_config', {}).get('layout_image_scale', 1.5) # ~108 DPI default
            layout_resolution = layout_scale * 72
            # Render without existing highlights to avoid interference
            page_image = self._page.to_image(resolution=layout_resolution, include_highlights=False)
            logger.debug(f"  Rendered image size: {page_image.width}x{page_image.height}")
        except Exception as e:
            logger.error(f"  Failed to render page {self._page.number} to image: {e}", exc_info=True)
            return []

        # --- Prepare Arguments for Layout Manager ---
        manager_args = {'image': page_image, 'options': options, 'engine': engine}
        if confidence is not None: manager_args['confidence'] = confidence
        if classes is not None: manager_args['classes'] = classes
        if exclude_classes is not None: manager_args['exclude_classes'] = exclude_classes
        if device is not None: manager_args['device'] = device

        # --- Call Layout Manager ---
        logger.debug(f"  Calling Layout Manager...")
        try:
            detections = self._layout_manager.analyze_layout(**manager_args)
            logger.info(f"  Layout Manager returned {len(detections)} detections.")
        except Exception as e:
            logger.error(f"  Layout analysis failed: {e}", exc_info=True)
            return []

        # --- Process Detections (Convert to Regions, Scale Coords) ---
        # Calculate scale factor to convert from image back to PDF coordinates
        if page_image.width == 0 or page_image.height == 0:
            logger.error(f"Page {self._page.number}: Invalid rendered image dimensions ({page_image.width}x{page_image.height}). Cannot scale layout results.")
            return []
        scale_x = self._page.width / page_image.width
        scale_y = self._page.height / page_image.height
        logger.debug(f"  Scaling factors: x={scale_x:.4f}, y={scale_y:.4f}")

        layout_regions = []
        docling_id_to_region = {} # For hierarchy if using Docling

        for detection in detections:
            try:
                x_min, y_min, x_max, y_max = detection['bbox']

                # Convert coordinates from image to PDF space
                pdf_x0 = x_min * scale_x
                pdf_y0 = y_min * scale_y
                pdf_x1 = x_max * scale_x
                pdf_y1 = y_max * scale_y

                # Create a Region object
                region = Region(self._page, (pdf_x0, pdf_y0, pdf_x1, pdf_y1))
                region.region_type = detection.get('class', 'unknown') # Original class name
                region.normalized_type = detection.get('normalized_class', 'unknown') # Hyphenated name
                region.confidence = detection.get('confidence', 0.0)
                region.model = detection.get('model', engine or 'unknown') # Store model name
                region.source = 'detected'

                # Add extra info if available
                if 'text' in detection: region.text_content = detection['text']
                if 'docling_id' in detection: region.docling_id = detection['docling_id']
                if 'parent_id' in detection: region.parent_id = detection['parent_id']
                # Add other fields like polygon, position, row/col index if needed

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
        
        return layout_regions 