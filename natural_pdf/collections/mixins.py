import logging
from typing import Callable, Iterable, Any, TypeVar
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

T = TypeVar("T") # Generic type for items in the collection

class ApplyMixin:
    \"\"\"
    Mixin class providing an `.apply()` method for collections.

    Assumes the inheriting class implements `__iter__` and `__len__` appropriately
    for the items to be processed by `apply`.
    \"\"\"
    def _get_items_for_apply(self) -> Iterable[Any]:
        \"\"\"
        Returns the iterable of items to apply the function to.
        Defaults to iterating over `self`. Subclasses should override this
        if the default iteration is not suitable for the apply operation.
        \"\"\"
        # Default to standard iteration over the collection itself
        return iter(self)

    def apply(self: Any, func: Callable[[Any, ...], Any], *args, **kwargs) -> None:
        \"\"\"
        Applies a function to each item in the collection.

        Args:
            func: The function to apply to each item. The item itself
                  will be passed as the first argument to the function.
            *args: Additional positional arguments to pass to func.
            **kwargs: Additional keyword arguments to pass to func.
                      A special keyword argument 'show_progress' (bool, default=False)
                      can be used to display a progress bar.
        \"\"\"
        show_progress = kwargs.pop('show_progress', False)
        # Derive unit name from class name
        unit_name = self.__class__.__name__.lower()
        items_iterable = self._get_items_for_apply()

        # Need total count for tqdm, assumes __len__ is implemented by the inheriting class
        total_items = 0
        try:
             total_items = len(self)
        except TypeError: # Handle cases where __len__ might not be defined on self
             logger.warning(f"Could not determine collection length for progress bar.")

        if show_progress and total_items > 0:
            items_iterable = tqdm(items_iterable, total=total_items, desc=f"Applying {func.__name__}", unit=unit_name)
        elif show_progress:
             logger.info(f"Applying {func.__name__} (progress bar disabled for zero/unknown length).")

        for item in items_iterable:
            try:
                # Apply the function with the item and any extra args/kwargs
                func(item, *args, **kwargs)
            except Exception as e:
                # Log and continue for batch operations
                logger.error(f"Error applying {func.__name__} to {item}: {e}", exc_info=True)
                # Optionally add a mechanism to collect errors

        # Returns None, primarily used for side effects.