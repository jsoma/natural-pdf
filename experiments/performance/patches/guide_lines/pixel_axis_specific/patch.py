"""Run pixel line detection only for the requested single guide axis."""

from __future__ import annotations

import inspect
from contextlib import contextmanager

from experiments.performance.patches.guide_lines._common import metric_count

METADATA = {
    "track": "guide_lines",
    "candidate": "pixel_axis_specific",
    "cache_only": False,
    "hypothesis": (
        "Single-axis pixel guide detection should not spend morphology/peak work on the "
        "opposite axis. Both-axis calls are left unchanged to avoid changing final page state."
    ),
}

_BOTH_AXIS_FLAG = "_npdf_perf_guides_both_axis"


@contextmanager
def install():
    from natural_pdf.analyzers.guides.base import Guides
    from natural_pdf.guides.engines.lines import LinesGuidesEngine

    original_detect = LinesGuidesEngine.detect
    original_from_lines = Guides.from_lines
    original_add_lines = Guides.add_lines

    def patched_detect(self, *, axis, method, context, options):
        detection_method = options.get("detection_method", "auto")
        detector = getattr(context, "detect_lines", None)
        if (
            getattr(context, _BOTH_AXIS_FLAG, False)
            or detector is None
            or detection_method not in ("pixels", "auto")
        ):
            return original_detect(self, axis=axis, method=method, context=context, options=options)

        def wrapped_detect_lines(**kwargs):
            kwargs = dict(kwargs)
            if axis == "horizontal":
                kwargs["horizontal"] = True
                kwargs["vertical"] = False
                metric_count("guide_lines.pixel_axis_specific_horizontal")
            else:
                kwargs["horizontal"] = False
                kwargs["vertical"] = True
                metric_count("guide_lines.pixel_axis_specific_vertical")
            return detector(**kwargs)

        try:
            setattr(context, "detect_lines", wrapped_detect_lines)
        except Exception:
            return original_detect(self, axis=axis, method=method, context=context, options=options)

        try:
            return original_detect(self, axis=axis, method=method, context=context, options=options)
        finally:
            try:
                setattr(context, "detect_lines", detector)
            except Exception:
                pass

    def patched_from_lines(cls, obj, *args, **kwargs):
        axis = kwargs.get("axis", args[0] if args else "both")
        if axis != "both":
            return original_from_lines.__func__(cls, obj, *args, **kwargs)
        previous = getattr(obj, _BOTH_AXIS_FLAG, None)
        setattr(obj, _BOTH_AXIS_FLAG, True)
        try:
            return original_from_lines.__func__(cls, obj, *args, **kwargs)
        finally:
            if previous is None:
                try:
                    delattr(obj, _BOTH_AXIS_FLAG)
                except Exception:
                    pass
            else:
                setattr(obj, _BOTH_AXIS_FLAG, previous)

    def patched_add_lines(self, *args, **kwargs):
        axis = kwargs.get("axis", args[0] if args else "both")
        target_obj = kwargs.get("obj") or self.context
        if axis != "both" or target_obj is None:
            return original_add_lines(self, *args, **kwargs)
        previous = getattr(target_obj, _BOTH_AXIS_FLAG, None)
        setattr(target_obj, _BOTH_AXIS_FLAG, True)
        try:
            return original_add_lines(self, *args, **kwargs)
        finally:
            if previous is None:
                try:
                    delattr(target_obj, _BOTH_AXIS_FLAG)
                except Exception:
                    pass
            else:
                setattr(target_obj, _BOTH_AXIS_FLAG, previous)

    patched_from_lines.__signature__ = inspect.signature(original_from_lines.__func__)  # type: ignore[attr-defined]
    patched_add_lines.__signature__ = inspect.signature(original_add_lines)  # type: ignore[attr-defined]

    LinesGuidesEngine.detect = patched_detect
    Guides.from_lines = classmethod(patched_from_lines)
    Guides.add_lines = patched_add_lines
    try:
        yield
    finally:
        LinesGuidesEngine.detect = original_detect
        Guides.from_lines = original_from_lines
        Guides.add_lines = original_add_lines
