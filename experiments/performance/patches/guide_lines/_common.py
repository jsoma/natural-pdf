"""Shared helpers for guide-line detection experiments."""

from __future__ import annotations

from typing import Any, Iterable, Optional

from natural_pdf.analyzers.guides.helpers import _bounds_from_object, _collect_line_elements

PIXEL_DETECT_KWARGS = {
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
}


def metric_count(label: str, amount: int | float = 1) -> None:
    try:
        from experiments.performance import vector_metrics

        vector_metrics.count(label, amount)
    except Exception:
        return


def _filtered_lines(context: Any, source_label: Optional[str]) -> list[Any]:
    lines = _collect_line_elements(context)
    if source_label:
        lines = [line for line in lines if getattr(line, "source", None) == source_label]
    return lines


def _line_length(line: Any, orientation: str) -> float:
    if orientation == "horizontal":
        return float(
            getattr(
                line,
                "width",
                abs(getattr(line, "x1", 0.0) - getattr(line, "x0", 0.0)),
            )
        )
    return float(
        getattr(
            line,
            "height",
            abs(getattr(line, "bottom", 0.0) - getattr(line, "top", 0.0)),
        )
    )


def _line_midpoint(line: Any, orientation: str) -> float:
    if orientation == "horizontal":
        return float((getattr(line, "top", 0.0) + getattr(line, "bottom", 0.0)) / 2)
    return float((getattr(line, "x0", 0.0) + getattr(line, "x1", 0.0)) / 2)


def select_line_coordinates(
    lines: Iterable[Any],
    *,
    orientation: str,
    max_lines: Optional[int],
) -> list[float]:
    line_data: list[tuple[float, float]] = []
    flag = "is_horizontal" if orientation == "horizontal" else "is_vertical"
    for line in lines:
        if not getattr(line, flag, False):
            continue
        line_data.append((_line_midpoint(line, orientation), _line_length(line, orientation)))

    if max_lines:
        line_data = sorted(line_data, key=lambda item: item[1], reverse=True)[:max_lines]
    return sorted({float(coord) for coord, _ in line_data})


def add_outer_boundaries(
    *,
    verticals: list[float],
    horizontals: list[float],
    context: Any,
    outer: bool,
) -> tuple[list[float], list[float]]:
    if not outer:
        return verticals, horizontals

    bounds = _bounds_from_object(context)
    if bounds is None:
        return verticals, horizontals

    x0, top, x1, bottom = bounds
    if not verticals or verticals[0] > x0:
        verticals.insert(0, float(x0))
    if not verticals or verticals[-1] < x1:
        verticals.append(float(x1))
    if not horizontals or horizontals[0] > top:
        horizontals.insert(0, float(top))
    if not horizontals or horizontals[-1] < bottom:
        horizontals.append(float(bottom))
    return sorted({float(v) for v in verticals}), sorted({float(h) for h in horizontals})


def pixel_detect_params(
    *,
    threshold: float | str,
    source_label: Optional[str],
    max_lines_h: Optional[int],
    max_lines_v: Optional[int],
    resolution: int,
    detect_kwargs: dict[str, Any],
    horizontal: bool = True,
    vertical: bool = True,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "resolution": resolution,
        "source_label": source_label or "guides_detection",
        "horizontal": horizontal,
        "vertical": vertical,
        "replace": True,
        "method": detect_kwargs.get("method", "projection"),
    }
    if threshold == "auto":
        params["peak_threshold_h"] = 0.5
        params["peak_threshold_v"] = 0.5
    else:
        params["peak_threshold_h"] = float(threshold)
        params["peak_threshold_v"] = float(threshold)

    params["max_lines_h"] = max_lines_h
    params["max_lines_v"] = max_lines_v
    for key in PIXEL_DETECT_KWARGS:
        if key in detect_kwargs:
            params[key] = detect_kwargs[key]
    return params


def detect_both_line_coordinates(
    context: Any,
    *,
    threshold: float | str,
    source_label: Optional[str],
    max_lines_h: Optional[int],
    max_lines_v: Optional[int],
    outer: bool,
    detection_method: str,
    resolution: int,
    detect_kwargs: dict[str, Any],
) -> tuple[list[float], list[float], str]:
    method = detection_method
    lines: list[Any] = []

    if method in ("vector", "auto"):
        metric_count("guide_lines.vector_collect")
        lines = _filtered_lines(context, source_label)
        if method == "auto":
            method = "vector" if lines else "pixels"

    if method == "pixels":
        detector = getattr(context, "detect_lines", None)
        if detector is None:
            raise ValueError(f"Object {context!r} does not support pixel-based line detection")
        params = pixel_detect_params(
            threshold=threshold,
            source_label=source_label,
            max_lines_h=max_lines_h,
            max_lines_v=max_lines_v,
            resolution=resolution,
            detect_kwargs=detect_kwargs,
        )
        metric_count("guide_lines.pixel_detect")
        detector(**params)
        lines = _filtered_lines(context, params["source_label"])
    elif method != "vector":
        raise ValueError(
            f"Unsupported detection method {detection_method!r}. Use 'pixels', 'vector', or 'auto'."
        )

    verticals = select_line_coordinates(lines, orientation="vertical", max_lines=max_lines_v)
    horizontals = select_line_coordinates(lines, orientation="horizontal", max_lines=max_lines_h)
    verticals, horizontals = add_outer_boundaries(
        verticals=verticals,
        horizontals=horizontals,
        context=context,
        outer=outer,
    )
    return verticals, horizontals, method
