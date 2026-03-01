"""Shared pdfplumber helper functions for table extraction."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from natural_pdf.tables.utils.guides import adjust_explicit_vertical_guides

logger = logging.getLogger(__name__)


def inject_text_tolerances(region, table_settings: Dict[str, Any]) -> None:
    """Inject default tolerances for text strategies unless already provided."""

    page_cfg = region.page._config
    pdf_cfg = page_cfg if page_cfg else region.page._parent._config

    uses_text = "text" in (
        table_settings.get("vertical_strategy"),
        table_settings.get("horizontal_strategy"),
    )
    if not uses_text:
        return

    if "text_x_tolerance" not in table_settings and "x_tolerance" not in table_settings:
        x_tol = pdf_cfg.get("x_tolerance")
        if x_tol is not None:
            table_settings.setdefault("text_x_tolerance", x_tol)

    if "text_y_tolerance" not in table_settings and "y_tolerance" not in table_settings:
        y_tol = pdf_cfg.get("y_tolerance")
        if y_tol is not None:
            table_settings.setdefault("text_y_tolerance", y_tol)

    if "snap_tolerance" not in table_settings and "snap_x_tolerance" not in table_settings:
        snap = max(1, round((pdf_cfg.get("y_tolerance", 1)) * 0.9))
        table_settings.setdefault("snap_tolerance", snap)

    if "join_tolerance" not in table_settings and "join_x_tolerance" not in table_settings:
        join = table_settings.get("snap_tolerance", 1)
        table_settings.setdefault("join_tolerance", join)
        table_settings.setdefault("join_x_tolerance", join)
        table_settings.setdefault("join_y_tolerance", join)


def filter_page_for_exclusions(region, base_page, *, apply_exclusions: bool):
    """Return a pdfplumber page filtered to remove chars in exclusion regions."""

    if not apply_exclusions or not getattr(region.page, "_exclusions", None):
        return base_page

    exclusion_regions = region._get_exclusion_regions(include_callable=True)

    def _keep_char(obj):
        if obj.get("object_type") != "char":
            return True
        cx = (obj["x0"] + obj["x1"]) / 2.0
        cy = (obj["top"] + obj["bottom"]) / 2.0
        for reg in exclusion_regions:
            if reg.x0 <= cx <= reg.x1 and reg.top <= cy <= reg.bottom:
                return False
        return True

    try:
        return base_page.filter(_keep_char)
    except Exception as exc:
        logger.warning(
            "Region %s: Failed to filter pdfplumber chars for exclusions: %s",
            getattr(region, "bbox", None),
            exc,
        )
        return base_page


def crop_page_to_region(page, bbox):
    """Crop a pdfplumber page to the requested bbox, returning None if invalid."""

    page_bbox = page.bbox
    clipped_bbox = (
        max(bbox[0], page_bbox[0]),
        max(bbox[1], page_bbox[1]),
        min(bbox[2], page_bbox[2]),
        min(bbox[3], page_bbox[3]),
    )

    if clipped_bbox[2] <= clipped_bbox[0] or clipped_bbox[3] <= clipped_bbox[1]:
        return None
    return page.crop(clipped_bbox)


def process_tables_with_rtl(
    region, tables: Sequence[Sequence[Sequence[Optional[str]]]]
) -> List[List[List[Optional[str]]]]:
    """Apply RTL processing to extracted tables."""

    processed: List[List[List[Optional[str]]]] = []
    for table in tables or []:
        table_rows: List[List[Optional[str]]] = []
        for row in table:
            processed_row: List[Optional[str]] = []
            for cell in row:
                if cell is not None:
                    processed_row.append(region._apply_rtl_processing_to_text(cell))
                else:
                    processed_row.append(cell)
            table_rows.append(processed_row)
        processed.append(table_rows)
    return processed


def _has_alt_text_regions(region) -> bool:
    """Return True if *region* contains any child regions with alt_text."""
    for r in region.page.iter_regions():
        alt = getattr(r, "alt_text", None)
        if alt is None or r is region:
            continue
        cx = (r.x0 + r.x1) / 2
        cy = (r.top + r.bottom) / 2
        if region.x0 <= cx <= region.x1 and region.top <= cy <= region.bottom:
            return True
    return False


def _extract_tables_via_regions(
    found_tables,
    page,
    *,
    apply_exclusions: bool,
) -> List[List[List[Optional[str]]]]:
    """Re-extract each cell using Region.extract_text() so alt_text is included."""
    from natural_pdf.elements.region import Region

    result: List[List[List[Optional[str]]]] = []
    for ft in found_tables:
        processed_table: List[List[Optional[str]]] = []
        for row in ft.rows:
            processed_row: List[Optional[str]] = []
            for cell_bbox in row.cells:
                if cell_bbox is None:
                    processed_row.append(None)
                    continue
                cell_region = Region(page, cell_bbox)
                text = cell_region.extract_text(apply_exclusions=apply_exclusions).strip()
                processed_row.append(text or None)
            processed_table.append(processed_row)
        result.append(processed_table)
    return result


def extract_tables_plumber(
    region,
    table_settings: Dict[str, Any],
    *,
    apply_exclusions: bool,
) -> List[List[List[Optional[str]]]]:
    """Extract tables from a pdfplumber crop using shared helper logic."""

    inject_text_tolerances(region, table_settings)
    adjust_explicit_vertical_guides(region, table_settings, apply_exclusions=apply_exclusions)

    base_plumber_page = region.page._page
    filtered_page = filter_page_for_exclusions(
        region, base_plumber_page, apply_exclusions=apply_exclusions
    )
    cropped = crop_page_to_region(filtered_page, region.bbox)
    if cropped is None:
        return []

    if _has_alt_text_regions(region):
        try:
            found_tables = cropped.find_tables(table_settings)
        except Exception:
            logger.debug(
                "Region %s: find_tables failed; falling back without alt_text",
                getattr(region, "bbox", None),
            )
            found_tables = None

        if found_tables is not None:
            tables = _extract_tables_via_regions(
                found_tables, region.page, apply_exclusions=apply_exclusions
            )
            return process_tables_with_rtl(region, tables)

    tables = cropped.extract_tables(table_settings)
    return process_tables_with_rtl(region, tables)
