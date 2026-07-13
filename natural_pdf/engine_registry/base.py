"""Core helpers shared by all engine registry adapters."""

from __future__ import annotations

from typing import Any, Callable, Optional

from natural_pdf.engine_provider import EngineCacheKey, EngineLifetime, get_provider

__all__ = ["register_engine", "register_builtin", "list_engines"]


def register_engine(
    capability: str,
    name: str,
    factory: Callable[..., Any],
    *,
    replace: bool = True,
    metadata: Optional[dict[str, Any]] = None,
    lifetime: EngineLifetime = "context",
    cache_key: Optional[EngineCacheKey] = None,
) -> None:
    """Register an engine factory for the given capability/name pair.

    Extension engines default to context-and-options caching. Pass
    ``lifetime="singleton"`` only for factories independent of retrieval
    inputs, or ``lifetime="transient"`` when callers own every instance.
    ``cache_key`` can define intentional sharing for context-lifetime engines.
    """

    provider = get_provider()
    provider.register(
        capability,
        name.strip().lower(),
        factory,
        metadata=metadata,
        replace=replace,
        lifetime=lifetime,
        cache_key=cache_key,
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
            lifetime="singleton",
        )
    else:
        get_provider().register(
            capability,
            name,
            factory,
            replace=replace,
            metadata=metadata,
            lifetime="singleton",
        )


def list_engines(capability: Optional[str] = None) -> dict[str, tuple[str, ...]]:
    """Return registered engines for a specific capability (or all if None)."""

    provider = get_provider()
    raw = provider.list(capability)
    return {cap: tuple(names) for cap, names in raw.items()}
