"""Core helpers shared by all engine registry adapters."""

from __future__ import annotations

from typing import Any, Callable, Optional

from natural_pdf.engine_provider import get_provider

__all__ = ["register_engine", "register_builtin", "list_engines"]


def register_engine(
    capability: str,
    name: str,
    factory: Callable[..., Any],
    *,
    replace: bool = True,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Register an engine factory for the given capability/name pair."""

    provider = get_provider()
    provider.register(
        capability,
        name.strip().lower(),
        factory,
        metadata=metadata,
        replace=replace,
    )


def register_builtin(
    provider,
    capability: str,
    name: str,
    factory: Callable[..., Any],
    *,
    replace: bool = True,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """
    Helper used by built-in registrations.

    If a provider is supplied, the factory is registered against it; otherwise the global
    registry is used. This keeps testing hooks simple without duplicating branching logic
    in every capability module.
    """

    if provider is not None:
        provider.register(
            capability,
            name,
            factory,
            replace=replace,
            metadata=metadata,
        )
    else:
        register_engine(
            capability,
            name,
            factory,
            replace=replace,
            metadata=metadata,
        )


def list_engines(capability: Optional[str] = None) -> dict[str, tuple[str, ...]]:
    """Return registered engines for a specific capability (or all if None)."""

    provider = get_provider()
    return provider.list(capability)
