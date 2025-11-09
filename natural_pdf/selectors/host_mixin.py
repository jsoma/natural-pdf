"""Shared protocol/mixin for selector-capable hosts."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SupportsSelectorHost(Protocol):
    """Protocol describing the minimal surface selector engines rely on."""

    def selector_page(self) -> Any: ...

    def selector_region(self) -> Any: ...

    def selector_flow(self) -> Any: ...


class SelectorHostMixin:
    """Mixin that provides default protocol implementations for host objects."""

    def selector_page(self) -> Any:  # pragma: no cover - trivial accessors
        if hasattr(self, "page_number"):
            # Pages (and Page-like shims) expose page_number; treat as the page itself.
            return self
        page = getattr(self, "_page", None)
        if page is not None:
            return page
        return getattr(self, "page", None)

    def selector_region(self) -> Any:
        if hasattr(self, "bbox"):
            return self
        return getattr(self, "region", None)

    def selector_flow(self) -> Any:
        if hasattr(self, "segments"):
            return self
        return getattr(self, "flow", None)
