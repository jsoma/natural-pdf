"""Vector checkbox detector — zero dependencies beyond pdfplumber.

Finds small square-ish rectangles in the parsed PDF element tree.
Returns bboxes in PDF coordinate space (no image rendering needed).
"""

import logging
from typing import Any, Dict, List, Optional

from PIL import Image

from .base import CheckboxDetector, DetectionContext
from .checkbox_options import BaseCheckboxOptions, VectorCheckboxOptions

logger = logging.getLogger(__name__)


class VectorCheckboxDetector(CheckboxDetector):
    """Detect checkboxes by finding small square rects in native PDF elements."""

    def is_available(self) -> bool:
        return True  # No extra dependencies

    def detect(
        self,
        image: Image.Image,
        options: BaseCheckboxOptions,
        context: Optional[DetectionContext] = None,
    ) -> List[Dict[str, Any]]:
        if context is None or context.page is None:
            self.logger.warning("VectorCheckboxDetector requires context.page")
            return []

        page = context.page

        # Coerce options
        if isinstance(options, VectorCheckboxOptions):
            opts = options
        else:
            opts = VectorCheckboxOptions(
                confidence=options.confidence,
                resolution=options.resolution,
                device=options.device,
            )

        # Query all rect elements from the page
        rects = page.find_all("rect")
        if not rects:
            return []

        candidates: List[Dict[str, Any]] = []

        for rect in rects:
            bbox = rect.bbox  # (x0, y0, x1, y1) in PDF coords
            x0, y0, x1, y1 = bbox
            w = x1 - x0
            h = y1 - y0

            if w <= 0 or h <= 0:
                continue

            # Size filter
            if w < opts.min_size or h < opts.min_size:
                continue
            if w > opts.max_size or h > opts.max_size:
                continue

            # Aspect ratio filter (reject non-square)
            aspect = max(w, h) / min(w, h)
            if aspect > opts.max_aspect_ratio:
                continue

            # Stroke filter (optional)
            if opts.require_stroke:
                stroke = getattr(rect, "stroke", None)
                if not stroke:
                    continue

            # Fill filter: reject solid-filled dark rects (likely bullets)
            fill = getattr(rect, "fill", None)
            non_stroking_color = getattr(rect, "non_stroking_color", None)
            if fill and non_stroking_color:
                # Check if it's a dark solid fill
                color = non_stroking_color
                if isinstance(color, (list, tuple)):
                    # Grayscale or RGB - check if dark
                    if len(color) == 1 and color[0] < 0.3:
                        continue
                    if len(color) == 3 and all(c < 0.3 for c in color):
                        continue

            candidates.append(
                {
                    "bbox": bbox,
                    "coord_space": "pdf",
                    "confidence": 0.9,  # High confidence for geometric match
                    "label": "checkbox",
                    "engine": "vector",
                    "is_checked": None,
                    "checkbox_state": "unknown",
                }
            )

        # Density filter: if too many candidates, likely a table grid
        if len(candidates) > 50:
            self.logger.info(
                "Vector detector found %d candidates (>50), likely table grid — returning empty",
                len(candidates),
            )
            return []

        self.logger.info("Vector detector found %d checkbox candidates", len(candidates))
        return candidates
