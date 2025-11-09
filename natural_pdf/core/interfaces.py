from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

if TYPE_CHECKING:
    from PIL import Image as PILImage

    from natural_pdf.core.element_manager import ElementManager
    from natural_pdf.core.highlighting_service import HighlightingService
    from natural_pdf.core.page import Page
    from natural_pdf.elements.base import Element
    from natural_pdf.elements.element_collection import ElementCollection
    from natural_pdf.elements.region import Region
    from natural_pdf.flows.region import FlowRegion

Bounds = Tuple[float, float, float, float]


class SupportsSections(ABC):
    """Minimal contract for objects that participate in section extraction pipelines."""

    @abstractmethod
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
    def to_region(self) -> Union["Region", "FlowRegion"]:
        """Return a region-like object suitable for flow-based operations."""


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

    def find(
        self,
        selector: Optional[str] = None,
        *,
        text: Optional[Union[str, Sequence[str]]] = None,
        overlap: Optional[str] = "full",
        apply_exclusions: bool = True,
        regex: bool = False,
        case: bool = True,
        text_tolerance: Optional[Dict[str, Any]] = None,
        auto_text_tolerance: Optional[Union[bool, Dict[str, Any]]] = None,
        reading_order: bool = True,
        near_threshold: Optional[float] = None,
        engine: Optional[str] = None,
    ) -> Optional["Element"]: ...

    def find_all(
        self,
        selector: Optional[str] = None,
        *,
        text: Optional[Union[str, Sequence[str]]] = None,
        overlap: Optional[str] = "full",
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


@runtime_checkable
class SupportsTextElements(Protocol):
    """Objects exposing section utilities plus word/char collections."""

    def find_all(
        self,
        selector: Optional[str] = None,
        *,
        text: Optional[Union[str, Sequence[str]]] = None,
        **kwargs: Any,
    ) -> "ElementCollection": ...

    def get_sections(
        self,
        start_elements: Union[str, Sequence["Element"], "ElementCollection", None] = None,
        end_elements: Union[str, Sequence["Element"], "ElementCollection", None] = None,
        **kwargs: Any,
    ) -> "ElementCollection": ...

    def to_region(self) -> "Region": ...

    @property
    def words(self) -> Sequence["Element"]: ...

    @property
    def chars(self) -> Sequence["Element"]: ...

    def add_exclusion(
        self, exclusion: Callable[["Page"], Optional["ElementCollection"]]
    ) -> "SupportsTextElements": ...

    def get_all_elements_raw(self) -> Sequence["Element"]: ...

    def get_elements_by_type(self, element_type: str) -> Sequence[Any]: ...

    def add_element(self, element: Any, element_type: str = "words") -> bool: ...

    def remove_element(self, element: Any, element_type: Optional[str] = None) -> bool: ...

    def remove_elements_by_source(self, element_type: str, source: str) -> int: ...

    def remove_ocr_elements(self) -> int: ...

    def clear_text_layer(self) -> Tuple[int, int]: ...

    def create_text_elements_from_ocr(
        self, ocr_results: Any, scale_x: Optional[float] = None, scale_y: Optional[float] = None
    ) -> Sequence[Any]: ...

    def iter_regions(self) -> Sequence["Region"]: ...

    def remove_regions_by_source(self, source: str) -> int: ...


@runtime_checkable
class Renderable(Protocol):
    """Objects that can be rendered to an image and expose dimensions."""

    @property
    def width(self) -> float: ...

    @property
    def height(self) -> float: ...

    def render(self, resolution: int = 72, **kwargs: Any) -> "PILImage.Image": ...


@runtime_checkable
class RenderablePage(Renderable, Protocol):
    """Renderable pages that expose a stable page number."""

    @property
    def number(self) -> int: ...


@runtime_checkable
class RenderableRegion(SupportsGeometry, Renderable, Protocol):
    """Renderable regions tied to a parent page."""


@runtime_checkable
class HasRenderablePages(Protocol):
    """Objects exposing a collection of renderable pages."""

    @property
    def pages(self) -> Sequence["RenderablePage"]: ...


@runtime_checkable
class SupportsPDFCollection(Protocol):
    """Collections that yield objects exposing renderable pages."""

    def __iter__(self) -> Iterator["HasRenderablePages"]: ...
