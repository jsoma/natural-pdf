"""Compute both guide-line axes in one pass for axis='both'."""

from __future__ import annotations

from contextlib import contextmanager

from experiments.performance.patches.guide_lines._common import detect_both_line_coordinates

METADATA = {
    "track": "guide_lines",
    "candidate": "direct_both_axis",
    "cache_only": False,
    "hypothesis": (
        "When both guide axes are requested, collect vector lines or run pixel detection once "
        "and derive both coordinate lists from the same line set."
    ),
}


@contextmanager
def install():
    from natural_pdf.analyzers.guides.base import Guides
    from natural_pdf.analyzers.guides.helpers import _bounds_from_object, _is_flow_region

    original_from_lines = Guides.from_lines
    original_add_lines = Guides.add_lines

    def patched_from_lines(
        cls,
        obj,
        axis="both",
        threshold="auto",
        source_label=None,
        max_lines_h=None,
        max_lines_v=None,
        outer=False,
        detection_method="auto",
        resolution=192,
        **detect_kwargs,
    ):
        if axis != "both" or _is_flow_region(obj):
            return original_from_lines.__func__(
                cls,
                obj,
                axis=axis,
                threshold=threshold,
                source_label=source_label,
                max_lines_h=max_lines_h,
                max_lines_v=max_lines_v,
                outer=outer,
                detection_method=detection_method,
                resolution=resolution,
                **detect_kwargs,
            )

        verticals, horizontals, _ = detect_both_line_coordinates(
            obj,
            threshold=threshold,
            source_label=source_label,
            max_lines_h=max_lines_h,
            max_lines_v=max_lines_v,
            outer=outer,
            detection_method=detection_method,
            resolution=resolution,
            detect_kwargs=detect_kwargs,
        )
        return cls(
            verticals=verticals,
            horizontals=horizontals,
            context=obj,
            bounds=_bounds_from_object(obj),
        )

    def patched_add_lines(
        self,
        axis="both",
        obj=None,
        threshold="auto",
        source_label=None,
        max_lines_h=None,
        max_lines_v=None,
        outer=False,
        detection_method="auto",
        resolution=192,
        **detect_kwargs,
    ):
        target_obj = obj or self.context
        if axis != "both" or target_obj is None or _is_flow_region(target_obj):
            return original_add_lines(
                self,
                axis=axis,
                obj=obj,
                threshold=threshold,
                source_label=source_label,
                max_lines_h=max_lines_h,
                max_lines_v=max_lines_v,
                outer=outer,
                detection_method=detection_method,
                resolution=resolution,
                **detect_kwargs,
            )

        verticals, horizontals, _ = detect_both_line_coordinates(
            target_obj,
            threshold=threshold,
            source_label=source_label,
            max_lines_h=max_lines_h,
            max_lines_v=max_lines_v,
            outer=outer,
            detection_method=detection_method,
            resolution=resolution,
            detect_kwargs=detect_kwargs,
        )
        self.vertical.extend(verticals)
        self.horizontal.extend(horizontals)
        return self

    Guides.from_lines = classmethod(patched_from_lines)
    Guides.add_lines = patched_add_lines
    try:
        yield
    finally:
        Guides.from_lines = original_from_lines
        Guides.add_lines = original_add_lines
