import logging
from typing import Any, Callable, Iterable, Optional, TypeVar

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

        # Determine the return type based on BOTH input collection type AND result type
        # For ElementCollection, check if results are actually elements/regions
        if self.__class__.__name__ == "ElementCollection":
            # Check if all non-None results are elements/regions
            non_none = [r for r in results if r is not None]
            if non_none and all(isinstance(r, (Element, Region)) for r in non_none):
                return ElementCollection(results)
            else:
                # Results are not elements (e.g., strings from extract_text), return plain list
                return results

        elif self.__class__.__name__ == "PageCollection":
            # Check if results are actually pages
            non_none = [r for r in results if r is not None]
            if non_none and all(isinstance(r, Page) for r in non_none):
                return PageCollection(results)
            else:
                return results

        elif self.__class__.__name__ == "PDFCollection":
            # Check if results are actually PDFs
            non_none = [r for r in results if r is not None]
            if non_none and all(isinstance(r, PDF) for r in non_none):
                return PDFCollection(results)
            else:
                return results

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

    def map(self: Any, func: Callable, *args, skip_empty: bool = False, **kwargs) -> Any:
        """
        Transform each item in the collection using the provided function.

        Args:
            func: Transformation function to apply to each item
            *args: Additional positional arguments to pass to func
            skip_empty: If True, removes None and empty values from results (default: False)
            **kwargs: Additional keyword arguments to pass to func

        Returns:
            The transformed results. Type depends on what the function returns:
            - If func returns elements/regions, returns appropriate collection type
            - If func returns other values (e.g., strings), returns a list

        Examples:
            # Extract text from all elements (including None for elements without text)
            texts = elements.map(lambda e: e.extract_text())

            # Extract text, skipping None and empty strings
            texts = elements.map(lambda e: e.extract_text(), skip_empty=True)

            # Transform elements (returns ElementCollection)
            expanded = elements.map(lambda e: e.expand(10))

            # With additional arguments
            results = elements.map(process_element, normalize=True, skip_empty=True)
        """
        # Use apply to get results with proper type handling
        results = self.apply(func, *args, **kwargs)

        if skip_empty:
            # Import collection types
            from natural_pdf.core.page_collection import PageCollection
            from natural_pdf.core.pdf_collection import PDFCollection
            from natural_pdf.elements.element_collection import ElementCollection

            # Filter out empty values
            filtered = [r for r in results if r]

            # Preserve the collection type if it's a collection
            if isinstance(results, ElementCollection):
                return ElementCollection(filtered)
            elif isinstance(results, PageCollection):
                return PageCollection(filtered)
            elif isinstance(results, PDFCollection):
                return PDFCollection(filtered)
            else:
                # It's a raw list
                return filtered

        return results

    def unique(self: Any, key: Optional[Callable] = None) -> Any:
        """
        Remove duplicate items from the collection.

        Args:
            key: Optional function to compute a comparison key from each element.
                 If not provided, elements are compared directly.

        Returns:
            A new collection of the same type with duplicates removed.
            Order is preserved (first occurrence of each unique item is kept).

        Examples:
            # Remove duplicate elements
            unique_elements = elements.unique()

            # Remove duplicates based on text content
            unique_by_text = elements.unique(key=lambda e: e.extract_text())

            # Remove duplicates based on position
            unique_by_pos = elements.unique(key=lambda e: (e.x, e.y))
        """
        items_iterable = self._get_items_for_apply()
        seen = set()
        unique_items = []

        for item in items_iterable:
            # Compute the comparison key
            if key is not None:
                comparison_key = key(item)
            else:
                comparison_key = item

            # For unhashable types, convert to a hashable representation
            try:
                hashable_key = comparison_key
                if hashable_key not in seen:
                    seen.add(hashable_key)
                    unique_items.append(item)
            except TypeError:
                # Handle unhashable types (like lists, dicts, or custom objects)
                # For elements/regions, use their string representation
                str_key = str(comparison_key)
                if str_key not in seen:
                    seen.add(str_key)
                    unique_items.append(item)

        # Return the appropriate collection type
        return type(self)(unique_items)

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
