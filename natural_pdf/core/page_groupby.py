"""
PageGroupBy class for grouping pages by selector text or callable results.
"""

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

from tqdm.auto import tqdm

from natural_pdf.utils.color_utils import format_color_value

if TYPE_CHECKING:
    from natural_pdf.core.page import Page
    from natural_pdf.core.page_collection import PageCollection


class PageGroupBy:
    """
    A groupby object for PageCollections that supports both iteration and dict-like access.

    This class provides pandas-like groupby functionality for natural-pdf PageCollections.
    Pages are grouped by the result of applying a selector string or callable function.

    Supports:
    - Direct iteration: for key, pages in grouped:
    - Dict-like access: grouped.get(key), grouped.get_group(key)
    - Batch operations: grouped.apply(func)
    """

    def __init__(
        self,
        page_collection: "PageCollection",
        by: Union[str, Callable],
        *,
        show_progress: bool = True,
    ):
        """
        Initialize the PageGroupBy object.

        Args:
            page_collection: The PageCollection to group
            by: CSS selector string or callable function for grouping
            show_progress: Whether to show progress bar during computation (default: True)
        """
        self.page_collection = page_collection
        self.by = by
        self.show_progress = show_progress
        self._groups: Optional[Dict[Any, "PageCollection"]] = None

    def _compute_groups(self) -> Dict[Any, "PageCollection"]:
        """
        Compute the groups by applying the selector/callable to each page.

        Returns:
            Dictionary mapping group keys to PageCollection objects
        """
        if self._groups is not None:
            return self._groups

        groups = defaultdict(list)

        # Setup progress bar if enabled and collection is large enough
        pages_iterator = self.page_collection.pages
        total_pages = len(self.page_collection)

        if self.show_progress and total_pages > 1:  # Show progress for more than 1 page
            desc = f"Grouping by {'selector' if isinstance(self.by, str) else 'function'}"
            pages_iterator = tqdm(pages_iterator, desc=desc, unit="pages", total=total_pages)

        for page in pages_iterator:
            if callable(self.by):
                # Apply callable function
                key = self.by(page)
            else:
                # Apply selector string
                element = page.find(self.by)
                if element:
                    key = element.extract_text()
                else:
                    key = None

            groups[key].append(page)

        # Convert lists to PageCollections
        from natural_pdf.core.page_collection import PageCollection

        self._groups = {key: PageCollection(pages) for key, pages in groups.items()}

        return self._groups

    def __iter__(self) -> Iterator[Tuple[Any, "PageCollection"]]:
        """
        Support direct iteration: for key, pages in grouped:

        Yields:
            Tuples of (group_key, PageCollection)
        """
        groups = self._compute_groups()
        return iter(groups.items())

    def get(
        self, key: Any, default: Optional["PageCollection"] = None
    ) -> Optional["PageCollection"]:
        """
        Dict-like access to get a specific group.

        Args:
            key: The group key to look up
            default: Value to return if key is not found

        Returns:
            PageCollection for the group, or default if not found
        """
        groups = self._compute_groups()
        return groups.get(key, default)

    def get_group(self, key: Any) -> "PageCollection":
        """
        Pandas-style access to get a specific group.

        Args:
            key: The group key to look up

        Returns:
            PageCollection for the group

        Raises:
            KeyError: If the group key is not found
        """
        groups = self._compute_groups()
        if key not in groups:
            raise KeyError(f"Group key '{key}' not found")
        return groups[key]

    def keys(self) -> List[Any]:
        """
        Get all group keys.

        Returns:
            List of all group keys
        """
        groups = self._compute_groups()
        return list(groups.keys())

    def __getitem__(self, index: Union[int, Any]) -> "PageCollection":
        """
        Access groups by index or key.

        Args:
            index: Integer index (0-based) or group key

        Returns:
            PageCollection for the specified group

        Examples:
            grouped = pages.groupby('text[size=16]')

            # Access by index (useful for quick exploration)
            first_group = grouped[0]        # First group by order
            second_group = grouped[1]       # Second group
            last_group = grouped[-1]        # Last group

            # Access by key (same as .get_group())
            madison = grouped['CITY OF MADISON']
        """
        groups = self._compute_groups()

        if isinstance(index, int):
            # Access by integer index
            keys_list = list(groups.keys())
            original_index = index  # Keep original for error message
            if index < 0:
                index = len(keys_list) + index  # Support negative indexing
            if not (0 <= index < len(keys_list)):
                raise IndexError(f"Group index {original_index} out of range")
            key = keys_list[index]
            return groups[key]
        else:
            # Access by key (same as get_group)
            if index not in groups:
                raise KeyError(f"Group key '{index}' not found")
            return groups[index]

    def apply(self, func: Callable[["PageCollection"], Any]) -> Dict[Any, Any]:
        """
        Apply a function to each group.

        Args:
            func: Function to apply to each PageCollection group

        Returns:
            Dictionary mapping group keys to function results
        """
        groups = self._compute_groups()
        return {key: func(pages) for key, pages in groups.items()}

    def show(self, **kwargs):
        """
        Show each group separately with headers.

        Args:
            **kwargs: Arguments passed to each group's show() method
        """
        groups = self._compute_groups()
        for key, pages in groups.items():
            # Format the key for display, converting colors to hex if needed
            if isinstance(self.by, str):
                # If grouped by a string selector, check if it's a color attribute
                formatted_key = format_color_value(key, attr_name=self.by)
            else:
                # For callable grouping, try to format as color
                formatted_key = format_color_value(key)

            print(f"\n--- Group: {formatted_key} ({len(pages)} pages) ---")
            pages.show(**kwargs)

    def __len__(self) -> int:
        """Return the number of groups."""
        groups = self._compute_groups()
        return len(groups)

    def info(self) -> None:
        """
        Print information about all groups.

        Useful for quick exploration of group structure.
        """
        groups = self._compute_groups()
        print(f"PageGroupBy with {len(groups)} groups:")
        print("-" * 40)

        for i, (key, pages) in enumerate(groups.items()):
            if key is None:
                key_display = "None"
            else:
                # Format the key for display, converting colors to hex if needed
                if isinstance(self.by, str):
                    formatted_key = format_color_value(key, attr_name=self.by)
                else:
                    formatted_key = format_color_value(key)
                key_display = f"'{formatted_key}'"
            print(f"[{i}] {key_display}: {len(pages)} pages")

    def __repr__(self) -> str:
        """String representation showing group count."""
        groups = self._compute_groups()
        return f"<PageGroupBy(groups={len(groups)})>"
