from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Iterable, List, Set

from natural_pdf.core.interfaces import HasHighlighter, HasPages, HasSinglePage


def resolve_highlighter(*sources: Any):
    """
    Resolve a HighlightingService from any combination of visualizable sources.
    """
    if not sources:
        raise RuntimeError("No sources provided for highlighter resolution")

    visited: Set[int] = set()
    queue: List[Any] = list(sources)

    while queue:
        obj = queue.pop(0)
        if obj is None:
            continue

        marker = id(obj)
        if marker in visited:
            continue
        visited.add(marker)

        if isinstance(obj, HasHighlighter):
            return obj.get_highlighter()

        if hasattr(obj, "get_highlighter"):
            getter = getattr(obj, "get_highlighter")
            if callable(getter):
                try:
                    return getter()  # type: ignore[call-arg]
                except Exception:
                    pass

        highlighter = getattr(obj, "_highlighter", None)
        if highlighter is not None:
            return highlighter

        if isinstance(obj, HasSinglePage):
            queue.append(obj.page)
            continue

        if isinstance(obj, HasPages):
            queue.extend(obj.pages)
            continue

        pages_attr = getattr(obj, "pages", None)
        if isinstance(pages_attr, Iterable):
            queue.extend(pages_attr)
            continue

        page_attr = getattr(obj, "page", None)
        if page_attr is not None:
            queue.append(page_attr)
            continue

        if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
            queue.extend(obj)

    raise RuntimeError("Cannot locate HighlightingService for the provided sources")
