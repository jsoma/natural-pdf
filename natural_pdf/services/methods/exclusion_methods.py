"""Shared exclusion helpers."""

from __future__ import annotations

from typing import Any, Optional

from natural_pdf.services.base import resolve_service


def add_exclusion(
    self,
    exclusion: Any,
    label: Optional[str] = None,
    method: str = "region",
):
    """Register an exclusion on the host via the exclusion service."""

    service = resolve_service(self, "exclusion")
    service.add_exclusion(self, exclusion, label=label, method=method)
    return self


__all__ = ["add_exclusion"]
