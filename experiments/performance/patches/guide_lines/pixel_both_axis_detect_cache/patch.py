"""Reuse one pixel line-detection call inside a both-axis guide operation."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from experiments.performance.patches.guide_lines._common import metric_count

METADATA = {
    "track": "guide_lines",
    "candidate": "pixel_both_axis_detect_cache",
    "cache_only": True,
    "hypothesis": (
        "The current both-axis guide path can render/detect pixels twice. A short-lived "
        "operation cache estimates the value of removing that duplicate work."
    ),
}


def _hashable(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((key, _hashable(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(item) for item in value)
    try:
        hash(value)
    except Exception:
        return repr(value)
    return value


@contextmanager
def _pixel_detect_cache(target: Any):
    original = getattr(target, "detect_lines", None)
    if original is None:
        yield
        return

    cache: dict[Any, Any] = {}

    def wrapped_detect_lines(**kwargs):
        key = tuple(
            sorted(
                (name, _hashable(value))
                for name, value in kwargs.items()
                if name
                not in {
                    "horizontal",
                    "vertical",
                    "max_lines_h",
                    "max_lines_v",
                    "replace",
                }
            )
        )
        if key in cache:
            metric_count("guide_lines.pixel_detect_cache_hit")
            return cache[key]
        metric_count("guide_lines.pixel_detect_cache_miss")
        result = original(**kwargs)
        cache[key] = result
        return result

    try:
        setattr(target, "detect_lines", wrapped_detect_lines)
    except Exception:
        yield
        return

    try:
        yield
    finally:
        try:
            setattr(target, "detect_lines", original)
        except Exception:
            pass


@contextmanager
def install():
    from natural_pdf.analyzers.guides.base import Guides

    original_from_lines = Guides.from_lines
    original_add_lines = Guides.add_lines

    def patched_from_lines(cls, obj, *args, **kwargs):
        axis = kwargs.get("axis", args[0] if args else "both")
        if axis != "both":
            return original_from_lines.__func__(cls, obj, *args, **kwargs)
        with _pixel_detect_cache(obj):
            return original_from_lines.__func__(cls, obj, *args, **kwargs)

    def patched_add_lines(self, *args, **kwargs):
        axis = kwargs.get("axis", args[0] if args else "both")
        target_obj = kwargs.get("obj") or self.context
        if axis != "both" or target_obj is None:
            return original_add_lines(self, *args, **kwargs)
        with _pixel_detect_cache(target_obj):
            return original_add_lines(self, *args, **kwargs)

    Guides.from_lines = classmethod(patched_from_lines)
    Guides.add_lines = patched_add_lines
    try:
        yield
    finally:
        Guides.from_lines = original_from_lines
        Guides.add_lines = original_add_lines
