"""Built-in engine for whitespace-based guide detection."""

from __future__ import annotations

from typing import Any, Dict

from natural_pdf.guides.guides_provider import Axis, GuidesDetectionResult, GuidesEngine


class WhitespaceGuidesEngine(GuidesEngine):
    """Use divide + snap-to-whitespace heuristic."""

    def detect(
        self,
        *,
        axis: Axis,
        method: str,
        context: Any,
        options: Dict[str, Any],
    ) -> GuidesDetectionResult:
        from natural_pdf.analyzers.guides.base import Guides

        min_gap = float(options.get("min_gap", 10))
        guides = Guides.divide(context, n=3, axis=axis)

        guides.snap_to_whitespace(
            axis=axis,
            min_gap=min_gap,
            detection_method="text",
            on_no_snap="ignore",
        )

        coords = guides.vertical if axis == "vertical" else guides.horizontal
        return GuidesDetectionResult(coordinates=list(float(v) for v in coords))
