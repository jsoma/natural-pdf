from __future__ import annotations

"""Internal registry that maps public capability names to service methods."""

from collections import defaultdict
from typing import Any, Callable, Dict, ItemsView, Iterator, Tuple

DelegateFunc = Callable[..., Any]


class DelegateRegistry:
    def __init__(self) -> None:
        self._entries: Dict[str, Dict[str, DelegateFunc]] = defaultdict(dict)

    def register(self, capability: str, method_name: str, func: DelegateFunc) -> None:
        """Store a delegate and guard against accidental duplicates."""
        methods = self._entries[capability]
        if method_name in methods:
            raise ValueError(
                f"Delegate '{method_name}' already registered for capability '{capability}'"
            )
        methods[method_name] = func

    def iter_entries(self, capability: str) -> ItemsView[str, DelegateFunc]:
        """Return an iterable of (method_name, func) pairs for the capability."""
        return self._entries.get(capability, {}).items()


_REGISTRY = DelegateRegistry()


def register_delegate(capability: str, method_name: str):
    """Decorator that records a service method for later attachment to hosts."""

    def decorator(func: DelegateFunc) -> DelegateFunc:
        _REGISTRY.register(capability, method_name, func)
        return func

    return decorator


def iter_delegates(capability: str) -> Iterator[Tuple[str, DelegateFunc]]:
    """Yield registered delegates for the requested capability."""
    return iter(_REGISTRY.iter_entries(capability))
