"""Structured extraction helper functions that call the extraction service."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Type, Union

from pydantic import BaseModel

from natural_pdf.services.base import resolve_service


def extract(
    self,
    schema: Union[Type[BaseModel], Sequence[str]],
    client: Any = None,
    analysis_key: str = "structured",
    prompt: Optional[str] = None,
    using: str = "text",
    model: Optional[str] = None,
    engine: Optional[str] = None,
    overwrite: bool = True,
    **kwargs: Any,
) -> Any:
    """Run structured extraction through the extraction service."""

    service = resolve_service(self, "extraction")
    service.extract(
        self,
        schema=schema,
        client=client,
        analysis_key=analysis_key,
        prompt=prompt,
        using=using,
        model=model,
        engine=engine,
        overwrite=overwrite,
        **kwargs,
    )
    return self


def extracted(
    self,
    field_name: Optional[str] = None,
    analysis_key: Optional[str] = None,
) -> Any:
    """Fetch a previously stored extraction result via the extraction service."""

    service = resolve_service(self, "extraction")
    return service.extracted(
        self,
        field_name=field_name,
        analysis_key=analysis_key,
    )


__all__ = ["extract", "extracted"]
