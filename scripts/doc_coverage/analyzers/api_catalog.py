"""Build API catalog by introspecting the natural_pdf module."""

import importlib
import inspect
from typing import Callable

# Additional submodules to include for natural_pdf
# Maps class names to their import paths
ADDITIONAL_CLASSES: dict[str, str] = {
    "TableResult": "natural_pdf.tables",
    "ElementCollection": "natural_pdf.elements.element_collection",
    "TextElement": "natural_pdf.elements.text",
    "LineElement": "natural_pdf.elements.line",
    "RectElement": "natural_pdf.elements.rect",
    "ImageElement": "natural_pdf.elements.image",
}


def build_api_catalog(
    module_name: str = "natural_pdf",
    include_inherited: bool = False,
    include_submodules: bool = True,
    exclude_internal: bool = False,
) -> dict[str, list[str]]:
    """Build a catalog of public API methods from a module.

    Args:
        module_name: The module to introspect.
        include_inherited: Whether to include inherited methods.
        include_submodules: Whether to include classes from submodules.
        exclude_internal: Whether to exclude internal-looking methods.

    Returns:
        Dictionary mapping class names to lists of public method names.
    """
    module = importlib.import_module(module_name)
    catalog = {}

    # Get all public classes from the module
    for name, obj in inspect.getmembers(module):
        if name.startswith("_"):
            continue

        if inspect.isclass(obj):
            methods = _get_public_methods(obj, include_inherited, exclude_internal)
            if methods:
                catalog[name] = methods

    # Include additional classes from submodules
    if include_submodules and module_name == "natural_pdf":
        for class_name, submodule_path in ADDITIONAL_CLASSES.items():
            if class_name in catalog:
                continue  # Already included
            try:
                submodule = importlib.import_module(submodule_path)
                cls = getattr(submodule, class_name, None)
                if cls and inspect.isclass(cls):
                    methods = _get_public_methods(cls, include_inherited, exclude_internal)
                    if methods:
                        catalog[class_name] = methods
            except (ImportError, AttributeError):
                pass  # Submodule or class not found

    return catalog


# Prefixes that indicate internal/utility methods
INTERNAL_PREFIXES = (
    "get_",
    "set_",
    "is_",
    "has_",
    "ensure_",
    "invalidate_",
)

# Methods to always exclude (internal implementations)
EXCLUDED_METHODS = {
    "append",
    "extend",
    "insert",
    "remove",
    "pop",
    "clear",
    "count",
    "index",  # list methods
    "copy",
    "sort",
    "reverse",  # more list methods
    "keys",
    "values",
    "items",
    "update",
    "get",  # dict methods
}


def _get_public_methods(
    cls: type,
    include_inherited: bool,
    exclude_internal: bool = False,
) -> list[str]:
    """Get public methods of a class.

    Args:
        cls: The class to inspect.
        include_inherited: Whether to include inherited methods.
        exclude_internal: Whether to exclude internal-looking methods.

    Returns:
        List of public method names.
    """
    methods = []

    if include_inherited:
        # Get all methods including inherited
        members = inspect.getmembers(cls, predicate=_is_method_or_property)
    else:
        # Only get methods defined directly on this class
        members = []
        for name in dir(cls):
            if name.startswith("_"):
                continue
            # Check if it's defined in this class (not inherited)
            for klass in cls.__mro__:
                if name in klass.__dict__:
                    if klass is cls or klass.__module__.startswith("natural_pdf"):
                        obj = getattr(cls, name, None)
                        if _is_method_or_property(obj):
                            members.append((name, obj))
                    break

    for name, _ in members:
        if name.startswith("_"):
            continue

        if exclude_internal:
            # Skip internal-looking methods
            if name.startswith(INTERNAL_PREFIXES):
                continue
            if name in EXCLUDED_METHODS:
                continue

        methods.append(name)

    return sorted(set(methods))


def _is_method_or_property(obj) -> bool:
    """Check if object is a method, function, or property."""
    return (
        inspect.isfunction(obj)
        or inspect.ismethod(obj)
        or isinstance(obj, property)
        or isinstance(obj, classmethod)
        or isinstance(obj, staticmethod)
    )
