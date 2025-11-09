"""PdfPlumber-backed table extraction engines."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from natural_pdf.tables.utils.common import tables_have_content
from natural_pdf.tables.utils.plumber import extract_tables_plumber

logger = logging.getLogger(__name__)


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
        **kwargs: Any,
    ) -> List[List[List[Optional[str]]]]:
        settings = dict(table_settings or {})
        apply_exclusions = kwargs.get("apply_exclusions", True)

        if self._mode == "direct":
            return extract_tables_plumber(
                region,
                table_settings=settings,
                apply_exclusions=apply_exclusions,
            )
        if self._mode == "stream":
            settings.setdefault("vertical_strategy", "text")
            settings.setdefault("horizontal_strategy", "text")
            return extract_tables_plumber(
                region,
                table_settings=settings,
                apply_exclusions=apply_exclusions,
            )
        if self._mode == "lattice":
            settings.setdefault("vertical_strategy", "lines")
            settings.setdefault("horizontal_strategy", "lines")
            return extract_tables_plumber(
                region,
                table_settings=settings,
                apply_exclusions=apply_exclusions,
            )
        if self._mode == "auto":
            return self._extract_auto(region, settings, apply_exclusions=apply_exclusions)
        raise ValueError(f"Unsupported pdfplumber table mode: {self._mode}")

    def _extract_auto(
        self,
        region,
        settings: Dict[str, Any],
        *,
        apply_exclusions: bool,
    ) -> List[List[List[Optional[str]]]]:
        logger.debug(
            "Region %s: Auto-detecting tables extraction method...",
            getattr(region, "bbox", None),
        )

        lattice_tables = extract_tables_plumber(
            region,
            table_settings={
                **settings,
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
            },
            apply_exclusions=apply_exclusions,
        )
        if tables_have_content(lattice_tables):
            logger.debug(
                "Region %s: 'lattice' method found %d tables",
                getattr(region, "bbox", None),
                len(lattice_tables),
            )
            return lattice_tables

        logger.debug(
            "Region %s: Falling back to 'stream' method for tables",
            getattr(region, "bbox", None),
        )
        return extract_tables_plumber(
            region,
            table_settings={**settings, "vertical_strategy": "text", "horizontal_strategy": "text"},
            apply_exclusions=apply_exclusions,
        )
