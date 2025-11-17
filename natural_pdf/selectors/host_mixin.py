"""Shared protocol/mixin for selector-capable hosts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Protocol, Sequence, Union, runtime_checkable

from natural_pdf.services.base import resolve_service

if TYPE_CHECKING:  # pragma: no cover
    from natural_pdf.elements.base import Element
    from natural_pdf.elements.element_collection import ElementCollection


class SelectorFindMethod(Protocol):
    """Shared signature for ``find`` across selector-capable hosts."""

    def __call__(
        self,
        selector: Optional[str] = None,
        *,
        text: Optional[Union[str, Sequence[str]]] = None,
        overlap: Optional[str] = None,
        apply_exclusions: bool = True,
        regex: bool = False,
        case: bool = True,
        text_tolerance: Optional[Dict[str, Any]] = None,
        auto_text_tolerance: Optional[Union[bool, Dict[str, Any]]] = None,
        reading_order: bool = True,
        near_threshold: Optional[float] = None,
        engine: Optional[str] = None,
    ) -> Optional["Element"]: ...


class SelectorFindAllMethod(Protocol):
    """Shared signature for ``find_all`` across selector-capable hosts."""

    def __call__(
        self,
        selector: Optional[str] = None,
        *,
        text: Optional[Union[str, Sequence[str]]] = None,
        overlap: Optional[str] = None,
        apply_exclusions: bool = True,
        regex: bool = False,
        case: bool = True,
        text_tolerance: Optional[Dict[str, Any]] = None,
        auto_text_tolerance: Optional[Union[bool, Dict[str, Any]]] = None,
        reading_order: bool = True,
        near_threshold: Optional[float] = None,
        engine: Optional[str] = None,
    ) -> "ElementCollection": ...


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

    def find(
        self,
        selector: Optional[str] = None,
        *,
        text: Optional[Union[str, Sequence[str]]] = None,
        overlap: Optional[str] = None,
        apply_exclusions: bool = True,
        regex: bool = False,
        case: bool = True,
        text_tolerance: Optional[Dict[str, Any]] = None,
        auto_text_tolerance: Optional[Union[bool, Dict[str, Any]]] = None,
        reading_order: bool = True,
        near_threshold: Optional[float] = None,
        engine: Optional[str] = None,
    ) -> Optional["Element"]:
        """Resolve a selector/text query against the host using the selector service."""

        return resolve_service(self, "selector").find(
            self,
            selector=selector,
            text=text,
            overlap=overlap,
            apply_exclusions=apply_exclusions,
            regex=regex,
            case=case,
            text_tolerance=text_tolerance,
            auto_text_tolerance=auto_text_tolerance,
            reading_order=reading_order,
            near_threshold=near_threshold,
            engine=engine,
        )

    def find_all(
        self,
        selector: Optional[str] = None,
        *,
        text: Optional[Union[str, Sequence[str]]] = None,
        overlap: Optional[str] = None,
        apply_exclusions: bool = True,
        regex: bool = False,
        case: bool = True,
        text_tolerance: Optional[Dict[str, Any]] = None,
        auto_text_tolerance: Optional[Union[bool, Dict[str, Any]]] = None,
        reading_order: bool = True,
        near_threshold: Optional[float] = None,
        engine: Optional[str] = None,
    ) -> "ElementCollection":
        """Return every element that matches the selector/text query."""

        return resolve_service(self, "selector").find_all(
            self,
            selector=selector,
            text=text,
            overlap=overlap,
            apply_exclusions=apply_exclusions,
            regex=regex,
            case=case,
            text_tolerance=text_tolerance,
            auto_text_tolerance=auto_text_tolerance,
            reading_order=reading_order,
            near_threshold=near_threshold,
            engine=engine,
        )
