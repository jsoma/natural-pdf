"""Common helper functions for table extraction."""

from __future__ import annotations

from typing import List, Optional, Sequence


def select_primary_table(
    tables: Sequence[Sequence[Sequence[Optional[str]]]],
) -> List[List[Optional[str]]]:
    """Choose the largest table (rows * cols) from a provider response."""

    best_table: List[List[Optional[str]]] = []
    best_score = -1
    for table in tables or []:
        if not table:
            continue
        row_count = len(table)
        col_count = max((len(row) for row in table), default=0)
        score = row_count * col_count
        if score > best_score:
            best_table = [list(row) for row in table]
            best_score = score
    return best_table


def tables_have_content(
    tables: Sequence[Sequence[Sequence[Optional[str]]]],
) -> bool:
    """Return True if any table contains non-empty cell text."""

    return any(
        any(any((cell or "").strip() for cell in row) for row in table if table) for table in tables
    )
