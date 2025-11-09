"""Utility helpers shared across table engines and mixins."""

from .cells import build_table_from_cells, extract_cell_text, merge_ocr_config
from .common import select_primary_table, tables_have_content
from .guides import adjust_explicit_vertical_guides
from .plumber import (
    crop_page_to_region,
    extract_tables_plumber,
    filter_page_for_exclusions,
    inject_text_tolerances,
    process_tables_with_rtl,
)

__all__ = [
    "select_primary_table",
    "tables_have_content",
    "adjust_explicit_vertical_guides",
    "build_table_from_cells",
    "extract_cell_text",
    "merge_ocr_config",
    "inject_text_tolerances",
    "filter_page_for_exclusions",
    "crop_page_to_region",
    "process_tables_with_rtl",
    "extract_tables_plumber",
]
