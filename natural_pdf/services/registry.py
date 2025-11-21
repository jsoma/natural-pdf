from __future__ import annotations

"""Internal registry that maps public capability names to service methods."""

from collections import defaultdict
from importlib import import_module
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

_CAPABILITY_MODULES = {
    "checkbox": "natural_pdf.services.checkbox_service",
    "classification": "natural_pdf.services.classification_service",
    "describe": "natural_pdf.services.describe_service",
    "exclusion": "natural_pdf.services.exclusion_service",
    "extraction": "natural_pdf.services.extraction_service",
    "guides": "natural_pdf.services.guides_service",
    "layout": "natural_pdf.services.layout_service",
    "navigation": "natural_pdf.services.navigation_service",
    "ocr": "natural_pdf.services.ocr_service",
    "qa": "natural_pdf.services.qa_service",
    "rendering": "natural_pdf.services.rendering_service",
    "selector": "natural_pdf.services.selector_service",
    "shapes": "natural_pdf.services.shape_detection_service",
    "table": "natural_pdf.services.table_service",
    "text": "natural_pdf.services.text_service",
    "vision": "natural_pdf.services.vision_service",
}

_IMPORTED_CAPABILITIES: Dict[str, bool] = {}


def _ensure_capability_loaded(capability: str) -> None:
    if _IMPORTED_CAPABILITIES.get(capability):
        return
    module_path = _CAPABILITY_MODULES.get(capability)
    if module_path is None:
        _IMPORTED_CAPABILITIES[capability] = True
        return
    import_module(module_path)
    _IMPORTED_CAPABILITIES[capability] = True


def register_delegate(capability: str, method_name: str):
    """Decorator that records a service method for later attachment to hosts."""

    def decorator(func: DelegateFunc) -> DelegateFunc:
        _REGISTRY.register(capability, method_name, func)
        return func

    return decorator


def iter_delegates(capability: str) -> Iterator[Tuple[str, DelegateFunc]]:
    """Yield registered delegates for the requested capability."""
    _ensure_capability_loaded(capability)
    return iter(_REGISTRY.iter_entries(capability))
