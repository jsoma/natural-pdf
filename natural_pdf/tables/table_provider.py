"""Provider helpers and built-in engines for table extraction."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Dict, List, Optional, Protocol

from natural_pdf.engine_provider import get_provider

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
    ) -> List[List[List[str]]]:
        """Extract tables for the provided region."""


class PdfPlumberTablesEngine:
    """Wraps pdfplumber-based extraction strategies for provider dispatch."""

    def __init__(self, mode: str):
        self._mode = mode

    def extract_tables(
        self,
        *,
        context: Any,
        region: Any,
        table_settings: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> List[List[List[str]]]:
        settings = dict(table_settings or {})
        if self._mode == "direct":
            return self._extract_direct(region, settings)
        if self._mode == "stream":
            return self._extract_stream(region, settings)
        if self._mode == "lattice":
            return self._extract_lattice(region, settings)
        if self._mode == "auto":
            return self._extract_auto(region, settings)
        raise ValueError(f"Unsupported pdfplumber table mode: {self._mode}")

    # -- strategy helpers -------------------------------------------------
    def _extract_direct(self, region: Any, settings: Dict[str, Any]) -> List[List[List[str]]]:
        return region._extract_tables_plumber(settings)

    def _extract_stream(self, region: Any, settings: Dict[str, Any]) -> List[List[List[str]]]:
        settings.setdefault("vertical_strategy", "text")
        settings.setdefault("horizontal_strategy", "text")
        return region._extract_tables_plumber(settings)

    def _extract_lattice(self, region: Any, settings: Dict[str, Any]) -> List[List[List[str]]]:
        settings.setdefault("vertical_strategy", "lines")
        settings.setdefault("horizontal_strategy", "lines")
        return region._extract_tables_plumber(settings)

    def _extract_auto(self, region: Any, settings: Dict[str, Any]) -> List[List[List[str]]]:
        logger.debug(
            "Region %s: Auto-detecting tables extraction method...", getattr(region, "bbox", None)
        )
        try:
            lattice_tables = self._extract_lattice(region, settings.copy())
            if self._has_meaningful_content(lattice_tables):
                logger.debug(
                    "Region %s: 'lattice' method found %d tables",
                    getattr(region, "bbox", None),
                    len(lattice_tables),
                )
                return lattice_tables
            logger.debug(
                "Region %s: 'lattice' method found no meaningful tables",
                getattr(region, "bbox", None),
            )
        except Exception as exc:
            logger.debug(
                "Region %s: 'lattice' method failed: %s", getattr(region, "bbox", None), exc
            )

        logger.debug(
            "Region %s: Falling back to 'stream' method for tables", getattr(region, "bbox", None)
        )
        return self._extract_stream(region, settings.copy())

    @staticmethod
    def _has_meaningful_content(tables: Optional[List[List[List[str]]]]) -> bool:
        if not tables:
            return False
        for table in tables:
            if not table:
                continue
            for row in table:
                if row and any((cell or "").strip() for cell in row):
                    return True
        return False


def register_table_engines(provider=None) -> None:
    """Register built-in table engines with the global provider."""

    provider = provider or get_provider()
    provider.register("tables", "pdfplumber_auto", lambda **_: PdfPlumberTablesEngine("auto"))
    provider.register("tables", "pdfplumber", lambda **_: PdfPlumberTablesEngine("direct"))
    provider.register("tables", "stream", lambda **_: PdfPlumberTablesEngine("stream"))
    provider.register("tables", "lattice", lambda **_: PdfPlumberTablesEngine("lattice"))


def run_table_engine(
    *,
    context: Any,
    region: Any,
    engine_name: str,
    table_settings: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> List[List[List[str]]]:
    """Execute a registered table engine and return the extracted tables.

    Args:
        context: Object initiating the extraction (Page, Region, Flow, etc.).
        region: Region-like object providing ``_extract_tables_plumber``.
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
        _normalize_engine_name(_context_config_value(context, "table_engine", scope)),
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


def _context_config_value(context: Any, key: str, scope: str) -> Any:
    sentinel = object()
    getter = getattr(context, "get_config", None)
    if callable(getter):
        try:
            value = getter(key, sentinel, scope=scope)
        except TypeError:
            try:
                value = getter(key, sentinel)
            except TypeError:
                value = sentinel
        if value is not sentinel:
            return value
    cfg = getattr(context, "_config", None)
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
]
