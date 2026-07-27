"""Built-in provider engine for line-based guide detection."""

from __future__ import annotations

import logging
import warnings
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

from natural_pdf.guides.guides_provider import (
    Axis,
    GuidesBothDetectionResult,
    GuidesDetectionResult,
    GuidesEngine,
)
from natural_pdf.guides.helpers import (
    GuidesContext,
    _bounds_from_object,
    _collect_line_elements,
)

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
        verticals, horizontals = self._detect_coordinates(
            context=context,
            options=options,
            axis_label=axis,
        )
        coords = verticals if axis == "vertical" else horizontals
        return GuidesDetectionResult(coordinates=coords)

    def detect_both(
        self,
        *,
        method: str,
        context: GuidesContext,
        options: Dict[str, Any],
    ) -> GuidesBothDetectionResult:
        verticals, horizontals = self._detect_coordinates(
            context=context,
            options=options,
            axis_label="both",
        )
        return GuidesBothDetectionResult(vertical=verticals, horizontal=horizontals)

    def _detect_coordinates(
        self,
        *,
        context: GuidesContext,
        options: Dict[str, Any],
        axis_label: str,
    ) -> tuple[List[float], List[float]]:
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
                    reason = f"{len(lines)} vector line element(s) exist"
                else:
                    method = "pixels"
                    if source_label:
                        reason = f"no vector line elements matched source_label={source_label!r}"
                    else:
                        reason = "no vector line elements exist"
                warnings.warn(
                    "Guides line detection used detection_method='auto' for "
                    f"{axis_label} guides and selected detection_method='{method}' because {reason}. "
                    f"Specify detection_method='{method}' to make this choice explicit and "
                    "silence this warning.",
                    UserWarning,
                    stacklevel=6,
                )

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

            line_element_data = self._detect_pixel_line_element_data(context, detect_params)
            if line_element_data is not None:
                lines = [self._line_like_from_element_data(data) for data in line_element_data]
            else:
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
            if axis_label in ("vertical", "both"):
                if not verticals or verticals[0] > bounds[0]:
                    verticals.insert(0, bounds[0])
                if not verticals or verticals[-1] < bounds[2]:
                    verticals.append(bounds[2])
            if axis_label in ("horizontal", "both"):
                if not horizontals or horizontals[0] > bounds[1]:
                    horizontals.insert(0, bounds[1])
                if not horizontals or horizontals[-1] < bounds[3]:
                    horizontals.append(bounds[3])

        return (
            sorted({float(v) for v in verticals}),
            sorted({float(h) for h in horizontals}),
        )

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

    @staticmethod
    def _detect_pixel_line_element_data(
        context: GuidesContext,
        detect_params: Dict[str, Any],
    ) -> Optional[List[Dict[str, Any]]]:
        services = getattr(context, "services", None)
        shapes = getattr(services, "shapes", None)
        detector = getattr(shapes, "detect_line_element_data", None)
        if not callable(detector):
            return None
        return detector(context, **detect_params)

    @staticmethod
    def _line_like_from_element_data(data: Dict[str, Any]) -> Any:
        x0 = float(data.get("x0", 0.0))
        x1 = float(data.get("x1", x0))
        top = float(data.get("top", 0.0))
        bottom = float(data.get("bottom", top))
        width = abs(x1 - x0)
        height = abs(bottom - top)
        return SimpleNamespace(
            x0=x0,
            x1=x1,
            top=top,
            bottom=bottom,
            width=width,
            height=height,
            source=data.get("source"),
            is_horizontal=width >= height,
            is_vertical=height > width,
        )
