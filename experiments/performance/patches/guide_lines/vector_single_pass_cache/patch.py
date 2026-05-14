"""Cache vector line collection within a both-axis guide operation."""

from __future__ import annotations

from contextlib import contextmanager

from experiments.performance.patches.guide_lines._common import metric_count

METADATA = {
    "track": "guide_lines",
    "candidate": "vector_single_pass_cache",
    "cache_only": True,
    "hypothesis": (
        "If vector guide overhead is mostly repeated line collection for axis='both', "
        "a short-lived collection cache should show it without rewriting coordinate logic."
    ),
}


@contextmanager
def install():
    from natural_pdf.analyzers.guides.base import Guides
    from natural_pdf.guides.engines import lines as lines_engine

    original_collect = lines_engine._collect_line_elements
    original_from_lines = Guides.from_lines
    original_add_lines = Guides.add_lines
    active_stack: list[dict[int, list[object]]] = []

    def patched_collect(obj):
        if not active_stack:
            return original_collect(obj)
        cache = active_stack[-1]
        key = id(obj)
        if key in cache:
            metric_count("guide_lines.vector_collect_cache_hit")
            return cache[key]
        metric_count("guide_lines.vector_collect_cache_miss")
        lines = original_collect(obj)
        cache[key] = lines
        return lines

    @contextmanager
    def collect_cache():
        active_stack.append({})
        try:
            yield
        finally:
            active_stack.pop()

    def patched_from_lines(cls, obj, *args, **kwargs):
        axis = kwargs.get("axis", args[0] if args else "both")
        if axis != "both":
            return original_from_lines.__func__(cls, obj, *args, **kwargs)
        with collect_cache():
            return original_from_lines.__func__(cls, obj, *args, **kwargs)

    def patched_add_lines(self, *args, **kwargs):
        axis = kwargs.get("axis", args[0] if args else "both")
        if axis != "both":
            return original_add_lines(self, *args, **kwargs)
        with collect_cache():
            return original_add_lines(self, *args, **kwargs)

    lines_engine._collect_line_elements = patched_collect
    Guides.from_lines = classmethod(patched_from_lines)
    Guides.add_lines = patched_add_lines
    try:
        yield
    finally:
        lines_engine._collect_line_elements = original_collect
        Guides.from_lines = original_from_lines
        Guides.add_lines = original_add_lines
