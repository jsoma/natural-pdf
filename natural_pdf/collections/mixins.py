from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

T = TypeVar("T")

if TYPE_CHECKING:  # pragma: no cover
    from natural_pdf import PDF, Page
    from natural_pdf.core.interfaces import SupportsElement, SupportsSections
    from natural_pdf.core.page_collection import PageCollection
    from natural_pdf.core.pdf_collection import PDFCollection
    from natural_pdf.elements.base import Element
    from natural_pdf.elements.element_collection import ElementCollection
    from natural_pdf.elements.region import Region


class _SupportsApply(Protocol):
    def apply(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any: ...


class _SectionHost(Protocol):
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
    ) -> Optional["Element"]: ...

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
    ) -> "ElementCollection": ...

    def extract_text(
        self,
        separator: str = "\n",
        apply_exclusions: bool = True,
        **kwargs: Any,
    ) -> str: ...


class DirectionalCollectionMixin:
    """Directional helpers for collections that expose ``apply``."""

    def below(self: _SupportsApply, **kwargs: Any) -> "ElementCollection":
        return self.apply(lambda element: element.below(**kwargs))

    def above(self: _SupportsApply, **kwargs: Any) -> "ElementCollection":
        return self.apply(lambda element: element.above(**kwargs))

    def left(self: _SupportsApply, **kwargs: Any) -> "ElementCollection":
        return self.apply(lambda element: element.left(**kwargs))

    def right(self: _SupportsApply, **kwargs: Any) -> "ElementCollection":
        return self.apply(lambda element: element.right(**kwargs))

    def expand(self: _SupportsApply, *args: Any, **kwargs: Any) -> "ElementCollection":
        return self.apply(lambda element: element.expand(*args, **kwargs))


class SectionsCollectionMixin:
    """Shared helpers for collections of objects implementing SupportsSections."""

    def _iter_sections(self) -> Iterable["_SectionHost"]:
        raise NotImplementedError

    def find(
        self,
        selector: Optional[str] = None,
        *,
        text: Optional[Union[str, Sequence[str]]] = None,
        apply_exclusions: bool = True,
        regex: bool = False,
        case: bool = True,
        text_tolerance: Optional[Dict[str, Any]] = None,
        auto_text_tolerance: Optional[Union[bool, Dict[str, Any]]] = None,
        reading_order: bool = True,
    ):
        if selector is not None and text is not None:
            raise ValueError("Provide either 'selector' or 'text', not both.")
        if selector is None and text is None:
            raise ValueError("Provide either 'selector' or 'text'.")

        for section in self._iter_sections():
            found = section.find(
                selector=selector,
                text=text,
                apply_exclusions=apply_exclusions,
                regex=regex,
                case=case,
                text_tolerance=text_tolerance,
                auto_text_tolerance=auto_text_tolerance,
                reading_order=reading_order,
            )
            if found:
                return found
        return None

    def find_all(
        self,
        selector: Optional[str] = None,
        *,
        text: Optional[Union[str, Sequence[str]]] = None,
        apply_exclusions: bool = True,
        regex: bool = False,
        case: bool = True,
        text_tolerance: Optional[Dict[str, Any]] = None,
        auto_text_tolerance: Optional[Union[bool, Dict[str, Any]]] = None,
        reading_order: bool = True,
    ):
        if selector is not None and text is not None:
            raise ValueError("Provide either 'selector' or 'text', not both.")
        if selector is None and text is None:
            raise ValueError("Provide either 'selector' or 'text'.")

        from natural_pdf.elements.element_collection import ElementCollection

        elements: List["Element"] = []
        for section in self._iter_sections():
            collection = section.find_all(
                selector=selector,
                text=text,
                apply_exclusions=apply_exclusions,
                regex=regex,
                case=case,
                text_tolerance=text_tolerance,
                auto_text_tolerance=auto_text_tolerance,
                reading_order=reading_order,
            )
            if collection:
                elements.extend(collection.elements)

        deduped: List["Element"] = []
        seen: set[int] = set()
        for element in elements:
            marker = id(element)
            if marker in seen:
                continue
            seen.add(marker)
            deduped.append(element)
        return ElementCollection(deduped)

    def extract_text(  # type: ignore[override]
        self,
        separator: str = "\n",
        apply_exclusions: bool = True,
        **kwargs: Any,
    ) -> str:
        texts = [
            section.extract_text(apply_exclusions=apply_exclusions, **kwargs)
            for section in self._iter_sections()
        ]
        return separator.join(t for t in texts if t)

    def extract_each_text(self, apply_exclusions: bool = True, **kwargs: Any) -> List[str]:
        return [
            section.extract_text(apply_exclusions=apply_exclusions, **kwargs)
            for section in self._iter_sections()
        ]


class ApplyMixin:
    """Provide ``apply``/``map``/``attr`` utilities for element collections."""

    def _get_items_for_apply(self) -> Iterable[Any]:
        return cast(Iterable[Any], self)

    def apply(self: Any, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        show_progress = kwargs.pop("show_progress", False)
        items_iterable = self._get_items_for_apply()

        total_items = 0
        try:
            total_items = len(self)  # type: ignore[arg-type]
        except TypeError:
            logger.debug("Collection does not expose __len__; skipping progress bar length.")

        if show_progress and total_items > 0:
            items_iterable = tqdm(
                items_iterable,
                total=total_items,
                desc=f"Applying {func.__name__}",
                unit=self.__class__.__name__.lower(),
            )
        elif show_progress:
            logger.info(
                "Applying %s (progress bar disabled for zero/unknown length).",
                func.__name__,
            )

        results: List[Any] = [func(item, *args, **kwargs) for item in items_iterable]

        # Lazy imports to avoid circular dependencies
        from natural_pdf import PDF, Page
        from natural_pdf.core.interfaces import SupportsElement
        from natural_pdf.core.page_collection import PageCollection
        from natural_pdf.core.pdf_collection import PDFCollection
        from natural_pdf.elements.base import Element
        from natural_pdf.elements.element_collection import ElementCollection
        from natural_pdf.elements.region import Region

        if isinstance(self, ElementCollection):
            non_none = [r for r in results if r is not None]
            if non_none and all(isinstance(r, (Element, Region)) for r in non_none):
                typed_results = cast(List[SupportsElement], results)
                return ElementCollection(typed_results)
            return results

        if isinstance(self, PageCollection):
            non_none = [r for r in results if r is not None]
            if non_none and all(isinstance(r, Page) for r in non_none):
                return PageCollection(results)
            return results

        if isinstance(self, PDFCollection):
            non_none = [r for r in results if r is not None]
            if non_none and all(isinstance(r, PDF) for r in non_none):
                return PDFCollection(results)
            return results

        if not results:
            return []

        first_non_none = next((r for r in results if r is not None), None)
        if isinstance(first_non_none, (Element, Region)):
            typed_results = cast(List[SupportsElement], results)
            return ElementCollection(typed_results)
        if isinstance(first_non_none, Page):
            return PageCollection(results)
        if isinstance(first_non_none, PDF):
            return PDFCollection(results)

        return results

    def map(
        self: Any,
        func: Callable[..., Any],
        *args: Any,
        skip_empty: bool = False,
        **kwargs: Any,
    ) -> Any:
        results = self.apply(func, *args, **kwargs)

        # Imports kept local to avoid circular references when map is unused.
        from natural_pdf.core.page_collection import PageCollection
        from natural_pdf.core.pdf_collection import PDFCollection
        from natural_pdf.elements.element_collection import ElementCollection

        if skip_empty:
            filtered = [r for r in cast(Sequence[Any], results) if r]
            if isinstance(results, ElementCollection):
                return ElementCollection(filtered)
            if isinstance(results, PageCollection):
                return PageCollection(filtered)
            if isinstance(results, PDFCollection):
                return PDFCollection(filtered)
            return filtered

        return results

    def attr(self: Any, name: str, skip_empty: bool = True) -> List[Any]:
        return self.map(lambda item: getattr(item, name, None), skip_empty=skip_empty)

    def unique(self: Any, key: Optional[Callable[[Any], Any]] = None) -> Any:
        items_iterable = list(self._get_items_for_apply())
        seen: set[Any] = set()
        unique_items: List[Any] = []

        for item in items_iterable:
            comparison_key = key(item) if key else item
            try:
                hashable_key = comparison_key
                if hashable_key not in seen:
                    seen.add(hashable_key)
                    unique_items.append(item)
            except TypeError:
                str_key = str(comparison_key)
                if str_key not in seen:
                    seen.add(str_key)
                    unique_items.append(item)

        return type(self)(unique_items)

    def filter(self: Any, predicate: Callable[[Any], bool]) -> Any:
        items_iterable = self._get_items_for_apply()
        filtered_items = [item for item in items_iterable if predicate(item)]
        return type(self)(filtered_items)
