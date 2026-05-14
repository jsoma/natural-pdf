"""Classify vector line guide candidates with NumPy arrays."""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np

from experiments.performance.patches.guide_lines._common import metric_count

METADATA = {
    "track": "guide_lines",
    "candidate": "vector_numpy_classify",
    "cache_only": False,
    "hypothesis": (
        "Vector-line guide detection can turn line geometry into arrays once and use "
        "boolean masks/top-k selection instead of per-line Python classification."
    ),
}


def _record_array_bytes(*arrays) -> None:
    try:
        from experiments.performance import vector_metrics

        vector_metrics.array_bytes("guide_lines.vector_arrays", *arrays)
    except Exception:
        return


@contextmanager
def install():
    from natural_pdf.analyzers.guides.helpers import _bounds_from_object, _collect_line_elements
    from natural_pdf.guides.engines.lines import LinesGuidesEngine
    from natural_pdf.guides.guides_provider import GuidesDetectionResult

    original_detect = LinesGuidesEngine.detect

    def patched_detect(self, *, axis, method, context, options):
        detection_method = options.get("detection_method", "auto")
        source_label = options.get("source_label")
        if detection_method not in ("vector", "auto"):
            return original_detect(self, axis=axis, method=method, context=context, options=options)

        lines = _collect_line_elements(context)
        if source_label:
            lines = [line for line in lines if getattr(line, "source", None) == source_label]
        if detection_method == "auto" and not lines:
            return original_detect(self, axis=axis, method=method, context=context, options=options)

        metric_count("guide_lines.vector_numpy_runs")
        if not lines:
            return GuidesDetectionResult(coordinates=[])

        x0 = np.asarray([float(getattr(line, "x0", 0.0)) for line in lines], dtype=np.float64)
        x1 = np.asarray([float(getattr(line, "x1", 0.0)) for line in lines], dtype=np.float64)
        top = np.asarray([float(getattr(line, "top", 0.0)) for line in lines], dtype=np.float64)
        bottom = np.asarray(
            [float(getattr(line, "bottom", 0.0)) for line in lines],
            dtype=np.float64,
        )
        _record_array_bytes(x0, x1, top, bottom)

        dx = np.abs(x1 - x0)
        dy = np.abs(bottom - top)
        if axis == "horizontal":
            mask = (dy <= 1.0) & (dx > 1.0)
            coords = (top[mask] + bottom[mask]) / 2.0
            lengths = dx[mask]
            max_lines = options.get("max_lines_h")
        else:
            mask = (dx <= 1.0) & (dy > 1.0)
            coords = (x0[mask] + x1[mask]) / 2.0
            lengths = dy[mask]
            max_lines = options.get("max_lines_v")

        if max_lines and len(coords) > max_lines:
            keep = np.argpartition(lengths, -int(max_lines))[-int(max_lines) :]
            coords = coords[keep]

        values = sorted({float(value) for value in coords.tolist()})

        if options.get("outer", False):
            bounds = _bounds_from_object(context)
            if bounds is not None:
                if axis == "vertical":
                    if not values or values[0] > bounds[0]:
                        values.insert(0, float(bounds[0]))
                    if not values or values[-1] < bounds[2]:
                        values.append(float(bounds[2]))
                else:
                    if not values or values[0] > bounds[1]:
                        values.insert(0, float(bounds[1]))
                    if not values or values[-1] < bounds[3]:
                        values.append(float(bounds[3]))
                values = sorted({float(value) for value in values})

        return GuidesDetectionResult(coordinates=values)

    LinesGuidesEngine.detect = patched_detect
    try:
        yield
    finally:
        LinesGuidesEngine.detect = original_detect
