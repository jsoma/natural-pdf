from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from natural_pdf.services.base import resolve_service
from natural_pdf.services.registry import iter_delegates

FallbackMap = Optional[Dict[str, Callable[..., Any]]]


def attach_capability(cls, capability: str, fallback_map: FallbackMap = None):
    """Attach registered service delegates to a class."""

    def _make_method(method_name: str, delegate_func):
        def method(self, *args, **kwargs):
            attrs = getattr(self, "__dict__", {})
            if "_context" in attrs and "_service_cache" in attrs:
                service = resolve_service(self, capability)
                bound = delegate_func.__get__(service, type(service))
                return bound(self, *args, **kwargs)

            if fallback_map:
                fallback = fallback_map.get(method_name)
                if fallback is not None:
                    return fallback(self, *args, **kwargs)

            service = resolve_service(self, capability)
            bound = delegate_func.__get__(service, type(service))
            return bound(self, *args, **kwargs)

        method.__name__ = method_name
        return method

    for method_name, func in iter_delegates(capability):
        if method_name in cls.__dict__:
            continue
        setattr(cls, method_name, _make_method(method_name, func))
    return cls
