from __future__ import annotations

from typing import Any, Dict

from natural_pdf.core.context import PDFContext


class ServiceHostMixin:
    """Provides helpers for objects that access services via PDFContext."""

    _context: PDFContext
    _service_cache: Dict[str, Any]

    def _init_service_host(self, context: PDFContext) -> None:
        self._context = context
        self._service_cache = {}

    def _get_service(self, capability: str) -> Any:
        if capability in self._service_cache:
            return self._service_cache[capability]
        service = self._context.get_service(capability)
        self._service_cache[capability] = service
        return service


def resolve_service(host: Any, capability: str) -> Any:
    """Return a service for hosts that may or may not inherit ServiceHostMixin."""

    attrs = getattr(host, "__dict__", {})
    if "_context" in attrs and "_service_cache" in attrs:
        getter = getattr(host, "_get_service", None)
        if callable(getter):
            return getter(capability)

    context = getattr(resolve_service, "_fallback_context", None)
    if context is None:
        context = PDFContext.with_defaults()
        setattr(resolve_service, "_fallback_context", context)
    return context.get_service(capability)
