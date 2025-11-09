"""Provider utilities for table structure detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, Sequence, Set

import natural_pdf
from natural_pdf.engine_provider import get_provider
from natural_pdf.engine_registry import register_builtin, register_structure_engine

logger = logging.getLogger(__name__)


@dataclass
class StructureDetectionResult:
    capabilities: Set[str] = field(default_factory=set)
    rows: Sequence[Any] = ()
    columns: Sequence[Any] = ()
    headers: Sequence[Any] = ()
    cells: Sequence[Any] = ()
    metadata: Optional[Dict[str, Any]] = None


class StructureEngine(Protocol):
    def detect(
        self,
        *,
        context: Any,
        region: Any,
        options: Optional[Dict[str, Any]] = None,
    ) -> StructureDetectionResult: ...


DEFAULT_STRUCTURE_ENGINE = "tatr"


def register_structure_engines(provider=None) -> None:
    from natural_pdf.tables.structure_engines.tatr import TATRStructureEngine

    factory = lambda **_: TATRStructureEngine()
    register_builtin(provider, "tables.detect_structure", "tatr", factory)


def run_table_structure_engine(
    *,
    context: Any,
    region: Any,
    engine_name: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Optional[StructureDetectionResult]:
    engine_id = engine_name or resolve_structure_engine_name(context)
    if not engine_id:
        return None
    provider = get_provider()
    engine = provider.get("tables.detect_structure", context=context, name=engine_id)
    return engine.detect(context=context, region=region, options=options or {})


def resolve_structure_engine_name(
    context: Any,
    requested: Optional[str] = None,
    *,
    scope: str = "region",
) -> Optional[str]:
    if requested:
        return requested

    model = getattr(context, "model", None)
    if isinstance(model, str) and model.lower() == "tatr":
        return "tatr"

    options = getattr(natural_pdf, "options", None)
    if options is not None:
        tables_opts = getattr(options, "tables", None)
        if tables_opts is not None:
            engine = getattr(tables_opts, "structure_engine", None)
            if engine:
                return engine

    if scope == "region":
        return None
    return None


try:
    register_structure_engines()
except Exception:  # pragma: no cover
    logger.exception("Failed to register structure engines")


__all__ = [
    "StructureDetectionResult",
    "run_table_structure_engine",
    "resolve_structure_engine_name",
    "register_structure_engines",
]
