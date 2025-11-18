"""Describe/inspect helper functions that delegate to the describe service."""

from __future__ import annotations

from typing import Any

from natural_pdf.services.base import resolve_service


def describe(self, **kwargs: Any) -> Any:
    """Delegate to DescribeService.describe and return its result."""

    service = resolve_service(self, "describe")
    return service.describe(self, **kwargs)


def inspect(self, limit: int = 30, **kwargs: Any) -> Any:
    """Delegate to DescribeService.inspect for consistent signatures/docs."""

    service = resolve_service(self, "describe")
    return service.inspect(self, limit=limit, **kwargs)


__all__ = ["describe", "inspect"]
