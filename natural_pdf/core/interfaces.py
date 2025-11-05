from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable

if TYPE_CHECKING:
    from natural_pdf.core.highlighting_service import HighlightingService
    from natural_pdf.core.page import Page
    from natural_pdf.elements.base import Element
    from natural_pdf.elements.element_collection import ElementCollection
    from natural_pdf.elements.region import Region

Bounds = Tuple[float, float, float, float]


class SupportsSections(ABC):
    """Minimal contract for objects that participate in section extraction pipelines."""

    @abstractmethod
    def find_all(
        self,
        selector: Optional[str] = None,
        *,
        text: Optional[Union[str, Sequence[str]]] = None,
        **kwargs,
    ) -> "ElementCollection":
        """Locate elements relative to the object."""

    @abstractmethod
    def get_sections(
        self,
        start_elements: Union[str, Sequence["Element"], "ElementCollection", None] = None,
        end_elements: Union[str, Sequence["Element"], "ElementCollection", None] = None,
        **kwargs,
    ) -> "ElementCollection":
        """Extract logical sections bounded by the supplied markers."""

    @abstractmethod
    def to_region(self) -> "Region":
        """Return a Region representation suitable for flow-based operations."""


@runtime_checkable
class SupportsBBox(Protocol):
    """Objects exposing a bounding box."""

    @property
    def bbox(self) -> Bounds: ...


@runtime_checkable
class SupportsGeometry(SupportsBBox, Protocol):
    """Objects exposing geometric coordinates and a parent page."""

    @property
    def page(self) -> "Page": ...

    @property
    def x0(self) -> float: ...

    @property
    def x1(self) -> float: ...

    @property
    def top(self) -> float: ...

    @property
    def bottom(self) -> float: ...


@runtime_checkable
class SupportsElement(SupportsGeometry, Protocol):
    """Objects that behave like natural-pdf elements or regions."""

    def extract_text(self, *args: Any, **kwargs: Any) -> str: ...


@runtime_checkable
class HasPolygon(Protocol):
    """Objects that can expose polygonal geometry."""

    @property
    def has_polygon(self) -> bool: ...

    @property
    def polygon(self) -> Sequence[Tuple[float, float]]: ...


@runtime_checkable
class HasSinglePage(Protocol):
    """Objects guaranteed to reside on a single page."""

    @property
    def page(self) -> "Page": ...


@runtime_checkable
class HasPages(Protocol):
    """Objects that may span multiple pages."""

    @property
    def pages(self) -> Sequence["Page"]: ...


@runtime_checkable
class HasHighlighter(Protocol):
    """Objects that can provide a HighlightingService."""

    def get_highlighter(self) -> "HighlightingService": ...


@runtime_checkable
class HasConfig(Protocol):
    """Objects that can resolve configuration values."""

    def get_config(self, key: str, default: Any = None, *, scope: str = "region") -> Any: ...


@runtime_checkable
class HasManager(Protocol):
    """Objects that can provide service managers by name."""

    def get_manager(self, name: str) -> Any: ...
