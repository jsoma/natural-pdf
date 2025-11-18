"""Shape-detection helper functions."""

from __future__ import annotations

from typing import Any

from natural_pdf.services.base import resolve_service


def detect_lines(self, **kwargs: Any):
    """Delegate line detection to the shape detection service."""

    service = resolve_service(self, "shapes")
    service.detect_lines(self, **kwargs)
    return self


__all__ = ["detect_lines"]
