"""Checkbox analyzer for PDF pages and regions."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from natural_pdf.elements.region import Region

from .checkbox_options import CheckboxOptions
from .registry import get_detector, prepare_checkbox_options

logger = logging.getLogger(__name__)


class CheckboxAnalyzer:
    """
    Handles checkbox analysis for PDF pages and regions, including image rendering,
    coordinate scaling, region creation, and result storage.
    """

    def __init__(self, element):
        """
        Initialize the checkbox analyzer.

        Args:
            element: The Page or Region object to analyze
        """
        self._element = element

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
        # Prepare options
        option_kwargs: Dict[str, Any] = dict(kwargs)
        if confidence is not None:
            option_kwargs["confidence"] = confidence
        if resolution is not None:
            option_kwargs["resolution"] = resolution
        if device is not None:
            option_kwargs["device"] = device

        engine_name, final_options = prepare_checkbox_options(
            engine, options, overrides=option_kwargs
        )

        logger.info(
            "Detecting checkboxes (Engine: %s, Element type: %s)",
            engine_name,
            "region" if self._is_region else "page",
        )

        # Render image
        try:
            resolution_val = resolution or getattr(final_options, "resolution", None) or 150

            pdf_scale_from_page_image: Optional[Tuple[float, float]] = None

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

                pdf_scale_from_page_image = (
                    self._page.width / page_image.width,
                    self._page.height / page_image.height,
                )

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
            detector = get_detector(engine_name)
            detections = detector.detect(image, final_options)
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
                    if pdf_scale_from_page_image is None:
                        raise RuntimeError("Missing page image scale for checkbox detection")
                    pdf_scale_x, pdf_scale_y = pdf_scale_from_page_image
                    pdf_x0 = page_img_x0 * pdf_scale_x
                    pdf_y0 = page_img_y0 * pdf_scale_y
                    pdf_x1 = page_img_x1 * pdf_scale_x
                    pdf_y1 = page_img_y1 * pdf_scale_y
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
                region = self._page.create_region(pdf_x0, pdf_y0, pdf_x1, pdf_y1)
                region.region_type = "checkbox"
                region.normalized_type = "checkbox"
                region.is_checked = bool(detection.get("is_checked", False))
                region.checkbox_state = str(detection.get("checkbox_state", "unchecked"))
                region.confidence = float(detection.get("confidence", 0.0))
                region.model = detection.get("model", "checkbox_detector")
                region.source = "checkbox"

                # Store original class for debugging
                region.original_class = detection.get("class", "unknown")

                region.analyses["checkbox"] = {
                    "is_checked": region.is_checked,
                    "state": region.checkbox_state,
                    "confidence": region.confidence,
                    "model": region.model,
                    "class": region.original_class,
                }

                # Check if region contains text - if so, it's probably not a checkbox
                # Get reject_with_text setting from options or kwargs, default to True
                reject_with_text = getattr(final_options, "reject_with_text", True)

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

        append_mode = existing.lower() == "append"
        if not append_mode:
            self._page.remove_regions(source="checkbox", region_type="checkbox")

        # Register regions with the page (ensures element manager + provenance)
        for region in checkbox_regions:
            self._page.add_region(region, source="checkbox")

        # Persist analysis snapshot for downstream consumers
        checkbox_analysis = self._page.analyses.setdefault("checkbox", {})
        prior_regions = list(checkbox_analysis.get("regions", [])) if append_mode else []
        checkbox_analysis["regions"] = prior_regions + list(checkbox_regions)
        checkbox_analysis["engine"] = engine_name

        logger.info(f"Checkbox detection complete. Found {len(checkbox_regions)} checkboxes.")

        return checkbox_regions
