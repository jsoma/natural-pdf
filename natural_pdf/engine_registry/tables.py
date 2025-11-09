"""Table-related registry helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Optional, Type

from .base import register_engine

__all__ = [
    "register_table_engine",
    "register_table_function",
    "register_structure_engine",
]

if TYPE_CHECKING:  # pragma: no cover
    from natural_pdf.tables.result import TableResult

_TABLE_RESULT_CLASS: Optional[Type[Any]] = None


def register_table_engine(
    name: str,
    factory: Callable[..., Any],
    *,
    replace: bool = True,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Register a full-featured table extraction engine."""

    register_engine(
        "tables",
        name,
        factory,
        replace=replace,
        metadata=metadata,
    )


def register_table_function(
    name: str,
    func: Callable[..., Any],
    *,
    replace: bool = True,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """
    Register a lightweight function-based table engine.

    The callable may return a ``TableResult``, a list of rows, or a list of tables.
    """

    def _factory(**_: Any) -> _FunctionalTableEngine:
        return _FunctionalTableEngine(func)

    register_table_engine(name=name, factory=_factory, replace=replace, metadata=metadata)


def register_structure_engine(
    name: str,
    factory: Callable[..., Any],
    *,
    replace: bool = True,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Register a table-structure detection engine (e.g., TATR cell consumer)."""

    register_engine(
        "tables.detect_structure",
        name,
        factory,
        replace=replace,
        metadata=metadata,
    )


class _FunctionalTableEngine:
    """Adapter that lets contributors register simple callables as engines."""

    def __init__(self, func: Callable[..., Any]) -> None:
        self._func = func

    def extract_tables(
        self,
        *,
        context: Any,
        region: Any,
        table_settings: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> list[list[list[str]]]:
        output = self._func(
            region=region,
            context=context,
            table_settings=table_settings,
            **kwargs,
        )
        return _normalize_table_output(output)


def _normalize_table_output(output: Any) -> list[list[list[str]]]:
    if output is None:
        return []

    if _is_table_result(output):
        return [_coerce_table(output)]

    if isinstance(output, Sequence) and not isinstance(output, (str, bytes)):
        if not output:
            return []

        if _looks_like_single_table(output):
            return [_coerce_table(output)]

        tables: list[list[list[str]]] = []
        for table in output:  # type: ignore[assignment]
            if table is None:
                continue
            if _is_table_result(table):
                tables.append(_coerce_table(table))
            else:
                tables.append(_coerce_table(table))
        return tables

    raise TypeError(
        "Functional table engines must return either TableResult, a list of rows, or a list of tables."
    )


def _coerce_table(table: Any) -> list[list[str]]:
    if _is_table_result(table):
        rows_iter = table  # type: ignore[assignment]
    elif isinstance(table, Sequence) and not isinstance(table, (str, bytes)):
        rows_iter = table  # type: ignore[assignment]
    else:
        raise TypeError("Table data must be a sequence of rows.")

    rows: list[list[str]] = []
    for row in rows_iter:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise TypeError("Each row must be a sequence of cell values.")
        rows.append([str(cell) if cell is not None else "" for cell in row])
    return rows


def _looks_like_single_table(candidate: Sequence[Any]) -> bool:
    first = candidate[0]
    return _looks_like_row(first)


def _looks_like_row(candidate: Any) -> bool:
    if not isinstance(candidate, Sequence) or isinstance(candidate, (str, bytes)):
        return False
    if not candidate:
        return True
    first = candidate[0]
    return not (isinstance(first, Sequence) and not isinstance(first, (str, bytes)))


def _is_table_result(obj: Any) -> bool:
    cls = _get_table_result_class()
    return cls is not None and isinstance(obj, cls)


def _get_table_result_class() -> Optional[Type[Any]]:
    global _TABLE_RESULT_CLASS
    if _TABLE_RESULT_CLASS is None:
        try:
            from natural_pdf.tables.result import TableResult
        except Exception:  # pragma: no cover - defensive
            return None
        _TABLE_RESULT_CLASS = TableResult
    return _TABLE_RESULT_CLASS
