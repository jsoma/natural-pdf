from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from natural_pdf.services.base import resolve_service
from natural_pdf.services.registry import iter_delegates

FallbackMap = Optional[Dict[str, Callable[..., Any]]]

_CustomCapability = Tuple[str, str, Callable[..., Any], bool]
_REGISTERED_CUSTOM_CAPABILITIES: List[_CustomCapability] = []


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


def register_capability(
    capability: str,
    helper: Callable[..., Any],
    *,
    hosts: Optional[Iterable[type]] = None,
    overwrite: bool = False,
):
    """Attach a helper to every service host and remember it for future subclasses."""

    method_name = helper.__name__
    entry: _CustomCapability = (capability, method_name, helper, overwrite)
    _REGISTERED_CUSTOM_CAPABILITIES.append(entry)

    target_hosts = list(hosts) if hosts is not None else list(_iter_service_hosts())
    for host in target_hosts:
        _apply_custom_helper(host, entry)
    return helper


def _register_host_for_custom_capabilities(host: type) -> None:
    """Called from ServiceHostMixin to apply helpers to new subclasses."""

    for entry in _REGISTERED_CUSTOM_CAPABILITIES:
        _apply_custom_helper(host, entry)


def _iter_service_hosts():
    """Yield all currently defined ServiceHostMixin subclasses."""

    from natural_pdf.services.base import ServiceHostMixin

    seen: set[type] = set()
    stack: list[type] = list(ServiceHostMixin.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        yield cls
        stack.extend(cls.__subclasses__())


def _apply_custom_helper(host: type, entry: _CustomCapability) -> None:
    capability, method_name, helper, overwrite = entry
    existing = getattr(host, method_name, None)
    if existing is not None and not overwrite:
        return
    setattr(host, method_name, helper)
    attach_capability(host, capability)
