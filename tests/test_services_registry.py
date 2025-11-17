from __future__ import annotations

import pytest

from natural_pdf.core.context import PDFContext
from natural_pdf.services.base import ServiceHostMixin
from natural_pdf.services.delegates import attach_capability
from natural_pdf.services.registry import DelegateRegistry, iter_delegates, register_delegate


def test_delegate_registry_rejects_duplicates():
    registry = DelegateRegistry()
    registry.register("navigation", "below", lambda self, host: None)

    with pytest.raises(ValueError):
        registry.register("navigation", "below", lambda self, host: None)


def test_register_delegate_decorator_wires_function():
    cap = "example"

    @register_delegate(cap, "say_hi")
    def say_hi(service, host):
        return f"hi from {host}"

    entries = dict(iter_delegates(cap))
    assert entries["say_hi"] is say_hi


def test_attach_capability_uses_registered_service():
    cap = "test-delegate-service"

    class DemoService:
        def __init__(self, _context):
            self.calls = []

        @register_delegate(cap, "ping")
        def ping(self, host, marker: str = "svc"):
            self.calls.append(marker)
            return f"ping:{marker}"

    class Host(ServiceHostMixin):
        def __init__(self):
            ctx = PDFContext(service_factories={cap: lambda ctx: DemoService(ctx)})
            self._init_service_host(ctx)

    attach_capability(Host, cap)
    host = Host()
    assert host.ping(marker="alpha") == "ping:alpha"


def test_attach_capability_uses_fallback_when_no_context():
    cap = "test-delegate-fallback"

    @register_delegate(cap, "hello")
    def hello(service, host):
        return "service-bound"

    class LegacyHost:
        pass

    attach_capability(LegacyHost, cap, fallback_map={"hello": lambda self: "fallback"})
    assert LegacyHost().hello() == "fallback"
