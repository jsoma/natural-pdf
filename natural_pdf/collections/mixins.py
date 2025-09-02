import logging
from typing import Any, Callable, Iterable, TypeVar

from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

T = TypeVar("T")  # Generic type for items in the collection


class DirectionalCollectionMixin:
    """
    Mixin providing directional methods for collections of elements/regions.
    """

    def below(self, **kwargs) -> "ElementCollection":
        """Find regions below all elements in this collection."""
        return self.apply(lambda element: element.below(**kwargs))

    def above(self, **kwargs) -> "ElementCollection":
        """Find regions above all elements in this collection."""
        return self.apply(lambda element: element.above(**kwargs))

    def left(self, **kwargs) -> "ElementCollection":
        """Find regions to the left of all elements in this collection."""
        return self.apply(lambda element: element.left(**kwargs))

    def right(self, **kwargs) -> "ElementCollection":
        """Find regions to the right of all elements in this collection."""
        return self.apply(lambda element: element.right(**kwargs))

    def expand(self, *args, **kwargs) -> "ElementCollection":
        """Expand all elements in this collection.

        Args:
            *args: If a single positional argument is provided, expands all elements
                   by that amount in all directions.
            **kwargs: Keyword arguments for directional expansion (left, right, top, bottom, etc.)

        Examples:
            # Expand all elements by 5 pixels in all directions
            collection.expand(5)

            # Expand with different amounts in each direction
            collection.expand(left=10, right=5, top=3, bottom=7)
        """
        return self.apply(lambda element: element.expand(*args, **kwargs))


class ApplyMixin:
    """
    Mixin class providing an `.apply()` method for collections.

    Assumes the inheriting class implements `__iter__` and `__len__` appropriately
    for the items to be processed by `apply`.
    """

    def _get_items_for_apply(self) -> Iterable[Any]:
        """
        Returns the iterable of items to apply the function to.
        Defaults to iterating over `self`. Subclasses should override this
        if the default iteration is not suitable for the apply operation.
        """
        # Default to standard iteration over the collection itself
        return iter(self)

    def apply(self: Any, func: Callable[[Any, ...], Any], *args, **kwargs) -> Iterable[Any]:
        """
        Applies a function to each item in the collection.

        Args:
            func: The function to apply to each item. The item itself
                  will be passed as the first argument to the function.
            *args: Additional positional arguments to pass to func.
            **kwargs: Additional keyword arguments to pass to func.
                      A special keyword argument 'show_progress' (bool, default=False)
                      can be used to display a progress bar.
        """
        show_progress = kwargs.pop("show_progress", False)
        # Derive unit name from class name
        unit_name = self.__class__.__name__.lower()
        items_iterable = self._get_items_for_apply()

        # Need total count for tqdm, assumes __len__ is implemented by the inheriting class
        total_items = 0
        try:
            total_items = len(self)
        except TypeError:  # Handle cases where __len__ might not be defined on self
            logger.warning(f"Could not determine collection length for progress bar.")

        if show_progress and total_items > 0:
            items_iterable = tqdm(
                items_iterable, total=total_items, desc=f"Applying {func.__name__}", unit=unit_name
            )
        elif show_progress:
            logger.info(
                f"Applying {func.__name__} (progress bar disabled for zero/unknown length)."
            )

        results = [func(item, *args, **kwargs) for item in items_iterable]

        # Import here to avoid circular imports
        from natural_pdf import PDF, Page
        from natural_pdf.core.page_collection import PageCollection
        from natural_pdf.core.pdf_collection import PDFCollection
        from natural_pdf.elements.base import Element
        from natural_pdf.elements.element_collection import ElementCollection
        from natural_pdf.elements.region import Region

        # Determine the return type based on the input collection type
        # This handles empty results correctly
        if self.__class__.__name__ == "ElementCollection":
            return ElementCollection(results)
        elif self.__class__.__name__ == "PageCollection":
            return PageCollection(results)
        elif self.__class__.__name__ == "PDFCollection":
            return PDFCollection(results)

        # If not a known collection type, try to infer from results
        if not results:
            return []

        first_non_none = next((r for r in results if r is not None), None)
        first_type = type(first_non_none) if first_non_none is not None else None

        # Return the appropriate collection based on result type (...generally)
        if first_type and (issubclass(first_type, Element) or issubclass(first_type, Region)):
            return ElementCollection(results)
        elif first_type == PDF:
            return PDFCollection(results)
        elif first_type == Page:
            return PageCollection(results)

        return results

    def filter(self: Any, predicate: Callable[[Any], bool]) -> Any:
        """
        Filters the collection based on a predicate function.

        Args:
            predicate: A function that takes an item and returns True if the item
                       should be included in the result, False otherwise.

        Returns:
            A new collection of the same type containing only the items
            for which the predicate returned True.
        """
        items_iterable = self._get_items_for_apply()
        filtered_items = [item for item in items_iterable if predicate(item)]

        return type(self)(filtered_items)
