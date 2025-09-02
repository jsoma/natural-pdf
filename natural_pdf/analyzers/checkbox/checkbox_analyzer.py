"""Checkbox analyzer for PDF pages and regions."""

import logging
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from natural_pdf.elements.region import Region

from .checkbox_manager import CheckboxManager
from .checkbox_options import CheckboxOptions

logger = logging.getLogger(__name__)


class CheckboxAnalyzer:
    """
    Handles checkbox analysis for PDF pages and regions, including image rendering,
    coordinate scaling, region creation, and result storage.
    """

    def __init__(self, element, checkbox_manager: Optional[CheckboxManager] = None):
        """
        Initialize the checkbox analyzer.

        Args:
            element: The Page or Region object to analyze
            checkbox_manager: Optional CheckboxManager instance. If None, creates a new one.
        """
        self._element = element
        self._checkbox_manager = checkbox_manager or CheckboxManager()

        # Determine if element is a page or region
        self._is_page = hasattr(element, "number") and hasattr(element, "_parent")
        self._is_region = hasattr(element, "bbox") and hasattr(element, "_page")

        if self._is_region:
            self._page = element._page
        else:
            self._page = element

    def detect_checkboxes(
        self,
        engine: Optional[str] = None,
        options: Optional[Union[CheckboxOptions, Dict[str, Any]]] = None,
        confidence: Optional[float] = None,
        resolution: Optional[int] = None,
        device: Optional[str] = None,
        existing: str = "replace",
        limit: Optional[int] = None,
        **kwargs,
    ) -> List[Region]:
        """
        Detect checkboxes in the page or region.

        Args:
            engine: Name of the detection engine (default: 'rtdetr')
            options: CheckboxOptions instance or dict of options
            confidence: Minimum confidence threshold
            resolution: DPI for rendering (default: 150)
            device: Device for inference
            existing: How to handle existing checkbox regions: 'replace' (default) or 'append'
            limit: Maximum number of checkboxes to detect
            **kwargs: Additional engine-specific arguments

        Returns:
            List of created Region objects representing checkboxes
        """
        logger.info(
            f"Detecting checkboxes (Engine: {engine or 'default'}, "
            f"Element type: {'region' if self._is_region else 'page'})"
        )

        # Prepare options
        if options is None:
            # Build options from simple arguments
            option_kwargs = {}
            if confidence is not None:
                option_kwargs["confidence"] = confidence
            if resolution is not None:
                option_kwargs["resolution"] = resolution
            if device is not None:
                option_kwargs["device"] = device
            option_kwargs.update(kwargs)

            # Let manager create appropriate options
            final_options = None
            final_kwargs = option_kwargs
        else:
            # Use provided options
            final_options = options
            # Apply any overrides
            final_kwargs = {}
            if confidence is not None:
                final_kwargs["confidence"] = confidence
            if resolution is not None:
                final_kwargs["resolution"] = resolution
            if device is not None:
                final_kwargs["device"] = device
            final_kwargs.update(kwargs)

        # Render image
        try:
            resolution_val = (
                resolution
                or (
                    final_options.resolution
                    if final_options and hasattr(final_options, "resolution")
                    else None
                )
                or 150
            )

            if self._is_region:
                # For regions, crop the page image to just the region bounds
                page_image = self._page.render(resolution=resolution_val)
                if not page_image:
                    raise ValueError("Page rendering returned None")

                # Calculate region bounds in image coordinates
                img_scale_x = page_image.width / self._page.width
                img_scale_y = page_image.height / self._page.height

                x0, y0, x1, y1 = self._element.bbox
                img_x0 = int(x0 * img_scale_x)
                img_y0 = int(y0 * img_scale_y)
                img_x1 = int(x1 * img_scale_x)
                img_y1 = int(y1 * img_scale_y)

                # Crop to region
                image = page_image.crop((img_x0, img_y0, img_x1, img_y1))

                # Store crop offset for coordinate transformation
                crop_offset = (img_x0, img_y0)

            else:
                # For pages, use the full image
                image = self._page.render(resolution=resolution_val)
                if not image:
                    raise ValueError("Page rendering returned None")
                crop_offset = (0, 0)

            logger.debug(f"Rendered image size: {image.width}x{image.height}")

        except Exception as e:
            logger.error(f"Failed to render image: {e}", exc_info=True)
            return []

        # Calculate scaling factors
        if self._is_region:
            # For regions, scale is relative to the cropped image
            scale_x = (self._element.bbox[2] - self._element.bbox[0]) / image.width
            scale_y = (self._element.bbox[3] - self._element.bbox[1]) / image.height
            pdf_offset = (self._element.bbox[0], self._element.bbox[1])
        else:
            # For pages, scale is from image to PDF coordinates
            scale_x = self._page.width / image.width
            scale_y = self._page.height / image.height
            pdf_offset = (0, 0)

        # Run detection
        try:
            detections = self._checkbox_manager.detect_checkboxes(
                image=image, engine=engine, options=final_options, **final_kwargs
            )
            logger.info(f"Detected {len(detections)} checkboxes")
        except Exception as e:
            logger.error(f"Checkbox detection failed: {e}", exc_info=True)
            return []

        # Process detections into regions
        checkbox_regions = []

        for detection in detections:
            try:
                # Get image coordinates
                img_x0, img_y0, img_x1, img_y1 = detection["bbox"]

                if self._is_region:
                    # For regions, add crop offset and scale to page image coords
                    page_img_x0 = img_x0 + crop_offset[0]
                    page_img_y0 = img_y0 + crop_offset[1]
                    page_img_x1 = img_x1 + crop_offset[0]
                    page_img_y1 = img_y1 + crop_offset[1]

                    # Then scale to PDF coords
                    pdf_x0 = page_img_x0 * (
                        self._page.width / (self._page.render(resolution=resolution_val).width)
                    )
                    pdf_y0 = page_img_y0 * (
                        self._page.height / (self._page.render(resolution=resolution_val).height)
                    )
                    pdf_x1 = page_img_x1 * (
                        self._page.width / (self._page.render(resolution=resolution_val).width)
                    )
                    pdf_y1 = page_img_y1 * (
                        self._page.height / (self._page.render(resolution=resolution_val).height)
                    )
                else:
                    # For pages, directly scale to PDF coordinates
                    pdf_x0 = img_x0 * scale_x + pdf_offset[0]
                    pdf_y0 = img_y0 * scale_y + pdf_offset[1]
                    pdf_x1 = img_x1 * scale_x + pdf_offset[0]
                    pdf_y1 = img_y1 * scale_y + pdf_offset[1]

                # Ensure valid bounds
                pdf_x0, pdf_x1 = min(pdf_x0, pdf_x1), max(pdf_x0, pdf_x1)
                pdf_y0, pdf_y1 = min(pdf_y0, pdf_y1), max(pdf_y0, pdf_y1)
                pdf_x0 = max(0, pdf_x0)
                pdf_y0 = max(0, pdf_y0)
                pdf_x1 = min(self._page.width, pdf_x1)
                pdf_y1 = min(self._page.height, pdf_y1)

                # For region detection, skip checkboxes outside the region bounds
                if self._is_region:
                    region_x0, region_y0, region_x1, region_y1 = self._element.bbox
                    # Check if checkbox center is within region
                    cb_center_x = (pdf_x0 + pdf_x1) / 2
                    cb_center_y = (pdf_y0 + pdf_y1) / 2
                    if not (
                        region_x0 <= cb_center_x <= region_x1
                        and region_y0 <= cb_center_y <= region_y1
                    ):
                        continue  # Skip this checkbox

                # Create region
                region = Region(self._page, (pdf_x0, pdf_y0, pdf_x1, pdf_y1))
                region.region_type = "checkbox"
                region.normalized_type = "checkbox"
                region.is_checked = detection.get("is_checked", False)
                region.checkbox_state = detection.get("checkbox_state", "unchecked")
                region.confidence = detection.get("confidence", 0.0)
                region.model = detection.get("model", "checkbox_detector")
                region.source = "checkbox"

                # Store original class for debugging
                region.original_class = detection.get("class", "unknown")

                # Check if region contains text - if so, it's probably not a checkbox
                # Get reject_with_text setting from options or kwargs, default to True
                reject_with_text = True
                if final_options:
                    reject_with_text = getattr(final_options, "reject_with_text", True)
                else:
                    reject_with_text = kwargs.get("reject_with_text", True)

                if reject_with_text:
                    text_in_region = region.extract_text().strip()
                    if text_in_region:
                        # Allow only single characters that might be check marks
                        if len(text_in_region) > 1 or text_in_region.isalnum():
                            logger.debug(
                                f"Rejecting checkbox at {region.bbox} - contains text: '{text_in_region}'"
                            )
                            continue

                checkbox_regions.append(region)

            except Exception as e:
                logger.warning(f"Could not process checkbox detection: {detection}. Error: {e}")
                continue

        # Apply limit if specified
        if limit is not None and len(checkbox_regions) > limit:
            # Sort by confidence (highest first) and take top N
            checkbox_regions = sorted(checkbox_regions, key=lambda r: r.confidence, reverse=True)[
                :limit
            ]

        # Final cleanup - ensure no overlapping boxes (this shouldn't be needed if NMS worked)
        cleaned_regions = []
        for region in checkbox_regions:
            overlaps = False
            for kept_region in cleaned_regions:
                # Check if bboxes overlap
                r1 = region.bbox
                r2 = kept_region.bbox
                if not (r1[2] <= r2[0] or r2[2] <= r1[0] or r1[3] <= r2[1] or r2[3] <= r1[1]):
                    overlaps = True
                    logger.warning(
                        f"Found overlapping checkbox regions after NMS: {r1} overlaps {r2}"
                    )
                    break
            if not overlaps:
                cleaned_regions.append(region)

        if len(cleaned_regions) < len(checkbox_regions):
            logger.warning(
                f"Removed {len(checkbox_regions) - len(cleaned_regions)} overlapping checkboxes in final cleanup"
            )
            checkbox_regions = cleaned_regions

        # Store results
        logger.debug(f"Storing {len(checkbox_regions)} checkbox regions (mode: {existing})")

        # Initialize storage if needed
        if not hasattr(self._page, "_regions"):
            self._page._regions = {}

        # Handle existing regions
        if existing.lower() == "append":
            if "checkbox" not in self._page._regions:
                self._page._regions["checkbox"] = []
            self._page._regions["checkbox"].extend(checkbox_regions)
        else:  # replace
            # Remove old checkbox regions from element manager
            if "checkbox" in self._page._regions:
                old_checkboxes = self._page._regions["checkbox"]
                if (
                    hasattr(self._page._element_mgr, "_elements")
                    and self._page._element_mgr._elements
                ):
                    current_regions = self._page._element_mgr._elements.get("regions", [])
                    # Remove old checkbox regions
                    self._page._element_mgr._elements["regions"] = [
                        r for r in current_regions if r not in old_checkboxes
                    ]
            self._page._regions["checkbox"] = checkbox_regions

        # Add to element manager
        for region in checkbox_regions:
            self._page._element_mgr.add_region(region)

        # Store for easy access
        self._page.detected_checkbox_regions = self._page._regions.get("checkbox", [])

        logger.info(f"Checkbox detection complete. Found {len(checkbox_regions)} checkboxes.")

        return checkbox_regions
