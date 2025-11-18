"""Provider helpers and built-in engines for table extraction."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Protocol

from natural_pdf.engine_provider import get_provider
from natural_pdf.engine_registry import register_builtin, register_table_engine
from natural_pdf.tables.engines.pdfplumber import PdfPlumberTablesEngine
from natural_pdf.tables.engines.tatr import TATRTableEngine
from natural_pdf.tables.engines.text import TextTablesEngine

logger = logging.getLogger(__name__)


class TableExtractionEngine(Protocol):
    """Protocol for pluggable table extraction engines."""

    def extract_tables(
        self,
        *,
        context: Any,
        region: Any,
        table_settings: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[List[List[Optional[str]]]]:
        """Extract tables for the provided region."""
        ...


def register_table_engines(provider=None) -> None:
    """Register built-in table engines with the provider or global registry."""

    def _register(name: str, factory: Callable[..., Any]) -> None:
        register_builtin(provider, "tables", name, factory)

    _register("pdfplumber_auto", lambda **_: PdfPlumberTablesEngine("auto"))
    _register("pdfplumber", lambda **_: PdfPlumberTablesEngine("direct"))
    _register("stream", lambda **_: PdfPlumberTablesEngine("stream"))
    _register("lattice", lambda **_: PdfPlumberTablesEngine("lattice"))
    _register("tatr", lambda **_: TATRTableEngine())
    _register("text", lambda **_: TextTablesEngine())


def run_table_engine(
    *,
    context: Any,
    region: Any,
    engine_name: str,
    table_settings: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> List[List[List[Optional[str]]]]:
    """Execute a registered table engine and return the extracted tables.

    Args:
        context: Object initiating the extraction (Page, Region, Flow, etc.).
        region: Region-like object for which the engine should compute tables.
        engine_name: Registered engine identifier.
        table_settings: Mutable dictionary of pdfplumber settings passed to the engine.
        **kwargs: Additional capability-specific arguments.
    """

    provider = get_provider()
    engine = provider.get("tables", context=context, name=engine_name)
    return engine.extract_tables(
        context=context,
        region=region,
        table_settings=table_settings,
        **kwargs,
    )


def normalize_table_settings(table_settings: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return a deep copy of table settings so engines can mutate safely."""

    if table_settings is None:
        return {}
    if isinstance(table_settings, dict):
        return deepcopy(table_settings)
    return deepcopy(dict(table_settings))


def resolve_table_engine_name(
    *,
    context: Any,
    requested: Optional[str] = None,
    scope: str = "region",
) -> str:
    """Resolve the desired table engine using explicit args, config, and defaults."""

    provider = get_provider()
    available = tuple(provider.list("tables").get("tables", ()))
    if not available:
        raise RuntimeError("No table engines are registered.")

    candidates = (
        _normalize_engine_name(requested),
        _normalize_engine_name(_context_option(context, "tables", "table_engine", scope)),
        _normalize_engine_name(_global_table_option("engine")),
        "pdfplumber_auto",
    )

    for candidate in candidates:
        mapped = _alias_engine_name(candidate)
        if mapped and mapped in available:
            return mapped

    raise LookupError(f"No suitable table engine found. Available engines: {available}.")


def _normalize_engine_name(name: Optional[Any]) -> Optional[str]:
    if isinstance(name, str):
        stripped = name.strip().lower()
        return stripped or None
    return None


def _alias_engine_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    if name in {"stream", "lattice", "pdfplumber"}:
        return name
    if name in {"auto", "default", "pdfplumber_auto"}:
        return "pdfplumber_auto"
    return name


def _context_option(host: Any, capability: str, key: str, scope: str) -> Any:
    context = getattr(host, "_context", None)
    if context is not None and hasattr(context, "get_option"):
        value = context.get_option(capability, key, host=host, scope=scope)
        if value is not None:
            return value

    getter = getattr(host, "get_config", None)
    if callable(getter):
        sentinel = object()
        try:
            value = getter(key, sentinel, scope=scope)
        except TypeError:
            try:
                value = getter(key, sentinel)
            except TypeError:
                value = sentinel
        if value is not sentinel:
            return value

    cfg = getattr(host, "_config", None)
    if isinstance(cfg, dict):
        return cfg.get(key)
    return None


def _global_table_option(attr: str) -> Any:
    try:
        import natural_pdf
    except Exception:  # pragma: no cover
        return None

    options = getattr(natural_pdf, "options", None)
    if options is None:
        return None
    section = getattr(options, "tables", None)
    if section is None:
        return None
    return getattr(section, attr, None)


try:  # Register engines on import so capability is available immediately.
    register_table_engines()
except Exception:  # pragma: no cover
    logger.exception("Failed to register built-in table engines")


__all__ = [
    "normalize_table_settings",
    "register_table_engines",
    "resolve_table_engine_name",
    "run_table_engine",
    "PdfPlumberTablesEngine",
    "TATRTableEngine",
    "TextTablesEngine",
]
