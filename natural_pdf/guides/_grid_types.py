"""Typed grid-build result containers for internal guides workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

Bounds = Tuple[float, float, float, float]


@dataclass
class GridBuildCounts:
    table: int = 0
    rows: int = 0
    columns: int = 0
    cells: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "table": self.table,
            "rows": self.rows,
            "columns": self.columns,
            "cells": self.cells,
        }

    def add(self, other: "GridBuildCounts") -> None:
        self.table += other.table
        self.rows += other.rows
        self.columns += other.columns
        self.cells += other.cells


@dataclass
class GridBuildRegions:
    table: Any = None
    rows: List[Any] = field(default_factory=list)
    columns: List[Any] = field(default_factory=list)
    cells: List[Any] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "table": self.table,
            "rows": list(self.rows),
            "columns": list(self.columns),
            "cells": list(self.cells),
        }


@dataclass
class GridBuildResult:
    counts: GridBuildCounts = field(default_factory=GridBuildCounts)
    regions: GridBuildRegions = field(default_factory=GridBuildRegions)
    effective_bbox: Optional[Bounds] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "counts": self.counts.as_dict(),
            "regions": self.regions.as_dict(),
        }
