"""Built-in provider engine for line-based guide detection."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from natural_pdf.analyzers.guides.helpers import (
    Bounds,
    GuidesContext,
    _bounds_from_object,
    _collect_line_elements,
)
from natural_pdf.guides.guides_provider import Axis, GuidesDetectionResult, GuidesEngine

logger = logging.getLogger(__name__)


class LinesGuidesEngine(GuidesEngine):
    """Detect guides from vector or pixel line information."""

    def detect(
        self,
        *,
        axis: Axis,
        method: str,
        context: GuidesContext,
        options: Dict[str, Any],
    ) -> GuidesDetectionResult:
        threshold = options.get("threshold", "auto")
        source_label = options.get("source_label")
        max_lines_h = options.get("max_lines_h")
        max_lines_v = options.get("max_lines_v")
        outer = options.get("outer", False)
        detection_method = options.get("detection_method", "auto")
        resolution = options.get("resolution", 192)
        detect_kwargs = {
            k: v
            for k, v in options.items()
            if k
            not in {
                "threshold",
                "source_label",
                "max_lines_h",
                "max_lines_v",
                "outer",
                "detection_method",
                "resolution",
            }
        }

        bounds = _bounds_from_object(context)
        if bounds is None:
            raise ValueError(
                f"Could not determine bounds for object {context!r} when detecting lines."
            )

        verticals: List[float] = []
        horizontals: List[float] = []

        lines: List[Any] = []
        method = detection_method

        if method in ("vector", "auto"):
            lines = _collect_line_elements(context)
            if source_label:
                lines = [line for line in lines if getattr(line, "source", None) == source_label]
            if method == "auto":
                if lines:
                    method = "vector"
                else:
                    method = "pixels"

        if method == "pixels":
            if not hasattr(context, "detect_lines"):
                raise ValueError(f"Object {context} does not support pixel-based line detection")

            default_label = source_label or "guides_detection"
            detect_params: Dict[str, Any] = {
                "resolution": resolution,
                "source_label": default_label,
                "horizontal": True,
                "vertical": True,
                "replace": True,
                "method": detect_kwargs.get("method", "projection"),
            }

            if threshold == "auto":
                detect_params["peak_threshold_h"] = 0.5
                detect_params["peak_threshold_v"] = 0.5
            else:
                detect_params["peak_threshold_h"] = float(threshold)
                detect_params["peak_threshold_v"] = float(threshold)

            detect_params["max_lines_h"] = max_lines_h
            detect_params["max_lines_v"] = max_lines_v

            for key in [
                "min_gap_h",
                "min_gap_v",
                "binarization_method",
                "adaptive_thresh_block_size",
                "adaptive_thresh_C_val",
                "morph_op_h",
                "morph_kernel_h",
                "morph_op_v",
                "morph_kernel_v",
                "smoothing_sigma_h",
                "smoothing_sigma_v",
                "peak_width_rel_height",
            ]:
                if key in detect_kwargs:
                    detect_params[key] = detect_kwargs[key]

            context.detect_lines(**detect_params)
            lines = [
                line
                for line in _collect_line_elements(context)
                if getattr(line, "source", None) == detect_params["source_label"]
            ]
        elif method != "vector":
            raise ValueError(
                f"Unsupported detection method '{detection_method}'. Use 'pixels', 'vector', or 'auto'."
            )

        if not lines and not hasattr(context, "lines") and not hasattr(context, "find_all"):
            logger.warning(f"Object {context} has no lines or find_all method")

        h_line_data: List[tuple[float, float, Any]] = []
        v_line_data: List[tuple[float, float, Any]] = []

        for line in lines:
            if hasattr(line, "is_horizontal") and getattr(line, "is_horizontal"):
                y = (line.top + line.bottom) / 2
                length = getattr(
                    line, "width", abs(getattr(line, "x1", 0) - getattr(line, "x0", 0))
                )
                h_line_data.append((y, float(length), line))
            if hasattr(line, "is_vertical") and getattr(line, "is_vertical"):
                x = (line.x0 + line.x1) / 2
                length = getattr(
                    line, "height", abs(getattr(line, "bottom", 0) - getattr(line, "top", 0))
                )
                v_line_data.append((x, float(length), line))

        horizontals = self._select_lines(h_line_data, max_lines_h)
        verticals = self._select_lines(v_line_data, max_lines_v)

        if outer:
            if axis == "vertical":
                if not verticals or verticals[0] > bounds[0]:
                    verticals.insert(0, bounds[0])
                if not verticals or verticals[-1] < bounds[2]:
                    verticals.append(bounds[2])
            if axis == "horizontal":
                if not horizontals or horizontals[0] > bounds[1]:
                    horizontals.insert(0, bounds[1])
                if not horizontals or horizontals[-1] < bounds[3]:
                    horizontals.append(bounds[3])

        if axis == "vertical":
            coords = sorted({float(v) for v in verticals})
        else:
            coords = sorted({float(h) for h in horizontals})

        return GuidesDetectionResult(coordinates=coords)

    @staticmethod
    def _select_lines(
        line_data: Sequence[tuple[float, float, Any]],
        max_lines: Optional[int],
    ) -> List[float]:
        if not line_data:
            return []
        if max_lines:
            ordered = sorted(line_data, key=lambda entry: entry[1], reverse=True)
            coords = [coord for coord, _, _ in ordered[: max_lines or len(ordered)]]
        else:
            coords = [coord for coord, _, _ in line_data]
        return sorted({float(coord) for coord in coords})
