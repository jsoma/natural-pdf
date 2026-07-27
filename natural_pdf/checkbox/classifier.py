"""Checkbox state classification (checked vs unchecked).

Provides pixel-metric heuristics and Judge adapter for classifying
detected checkbox regions. Used by the orchestrator after detection.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class CheckboxClassifier:
    """Classify checkbox regions as checked or unchecked."""

    @staticmethod
    def classify_by_pixels(
        region: Any,
        page: Any,
        min_render_size: int = 24,
    ) -> Tuple[bool, str, float]:
        """Classify a region by pixel metrics with dynamic DPI.

        Small checkboxes (6-10 PDF points) at 150 DPI become ~12-20px,
        too small for reliable metrics. We compute DPI dynamically to
        ensure the rendered region is at least min_render_size px on
        its shortest side.

        Args:
            region: Region object with bbox and page reference.
            page: Page object for rendering.
            min_render_size: Minimum pixels on shortest side.

        Returns:
            (is_checked, state_str, confidence)
        """
        try:
            # Calculate dynamic DPI
            x0, y0, x1, y1 = region.bbox
            w_pts = x1 - x0
            h_pts = y1 - y0
            min_pts = min(w_pts, h_pts)

            if min_pts <= 0:
                return False, "unknown", 0.0

            # Points to pixels: px = pts * dpi / 72
            # We want px >= min_render_size
            # dpi >= min_render_size * 72 / min_pts
            needed_dpi = max(150, int(min_render_size * 72 / min_pts) + 1)
            needed_dpi = min(needed_dpi, 600)  # Cap at 600 DPI

            # Render the region
            img = region.render(resolution=needed_dpi, crop=True)
            if img is None:
                return False, "unknown", 0.0

            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.asarray(img))

            metrics = CheckboxClassifier._extract_metrics(img)
            return CheckboxClassifier._classify_from_metrics(metrics)

        except Exception as e:
            logger.warning("Pixel classification failed: %s", e)
            return False, "unknown", 0.0

    @staticmethod
    def classify_by_judge(
        region: Any,
        judge: Any,
    ) -> Tuple[bool, str, float]:
        """Classify using a Judge instance.

        Args:
            region: Region object (must implement SupportsRender).
            judge: Judge instance with decide() method.

        Returns:
            (is_checked, state_str, confidence)
        """
        try:
            decision = judge.decide(region)
            label = decision.label
            score = decision.score
            is_checked = label.lower() in ("checked", "1", "tick", "filled")
            state = "checked" if is_checked else "unchecked"
            return is_checked, state, score
        except Exception as e:
            logger.warning("Judge classification failed: %s", e)
            return False, "unknown", 0.0

    @staticmethod
    def classify_regions(
        regions: List[Any],
        page: Any,
        judge: Optional[Any] = None,
        classify: bool = True,
    ) -> None:
        """Classify a list of regions in-place.

        Precedence: Judge (if provided) > Model output (if not None) > Pixel heuristics.

        Args:
            regions: List of Region objects with is_checked/checkbox_state attributes.
            page: Page object for rendering.
            judge: Optional Judge instance.
            classify: Whether to run classification at all.
        """
        for region in regions:
            current_checked = getattr(region, "is_checked", None)
            current_state = getattr(region, "checkbox_state", "unknown")

            # If model already gave an answer and no Judge, skip (unless state is unknown)
            if not classify and current_checked is not None:
                continue

            # Precedence: Judge > model output > pixel heuristics
            if judge is not None:
                is_checked, state, conf = CheckboxClassifier.classify_by_judge(region, judge)
                region.is_checked = is_checked
                region.checkbox_state = state
                if hasattr(region, "analyses") and "checkbox" in region.analyses:
                    region.analyses["checkbox"]["classified_by"] = "judge"
                continue

            if current_checked is not None and current_state != "unknown":
                # Model already provided classification
                continue

            if classify:
                # Fall back to pixel heuristics
                is_checked, state, conf = CheckboxClassifier.classify_by_pixels(region, page)
                region.is_checked = is_checked
                region.checkbox_state = state
                if hasattr(region, "analyses") and "checkbox" in region.analyses:
                    region.analyses["checkbox"]["classified_by"] = "pixels"

    @staticmethod
    def _extract_metrics(img: Image.Image) -> Dict[str, float]:
        """Extract image metrics for classification.

        Uses the same metrics as Judge._extract_metrics for consistency.
        """
        gray = np.array(img.convert("L"))
        h, w = gray.shape

        metrics: Dict[str, float] = {}

        # Center darkness
        cy, cx = h // 2, w // 2
        center_size = min(5, h // 4, w // 4)
        center = gray[
            max(0, cy - center_size) : min(h, cy + center_size + 1),
            max(0, cx - center_size) : min(w, cx + center_size + 1),
        ]
        metrics["center_darkness"] = float(255 - np.mean(center))

        # Overall ink density
        metrics["ink_density"] = float(255 - np.mean(gray))

        # Dark pixel ratio
        metrics["dark_pixel_ratio"] = float(np.sum(gray < 200) / gray.size)

        # Standard deviation
        metrics["std_dev"] = float(np.std(gray))

        return metrics

    @staticmethod
    def _classify_from_metrics(metrics: Dict[str, float]) -> Tuple[bool, str, float]:
        """Classify as checked/unchecked from pixel metrics.

        Thresholds derived from typical checkbox patterns:
        - Checked checkboxes have higher ink density and center darkness
        - Unchecked checkboxes are mostly white with border pixels only
        """
        ink = metrics.get("ink_density", 0)
        center = metrics.get("center_darkness", 0)
        dark_ratio = metrics.get("dark_pixel_ratio", 0)

        # Voting: multiple signals must agree
        votes_checked = 0
        total_votes = 0

        if ink > 40:
            votes_checked += 1
        total_votes += 1

        if center > 50:
            votes_checked += 1
        total_votes += 1

        if dark_ratio > 0.15:
            votes_checked += 1
        total_votes += 1

        is_checked = votes_checked >= 2
        confidence = votes_checked / total_votes if total_votes > 0 else 0.5
        state = "checked" if is_checked else "unchecked"

        return is_checked, state, confidence
