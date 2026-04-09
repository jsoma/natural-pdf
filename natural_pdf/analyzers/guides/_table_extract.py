"""Table extraction orchestration for Guides."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Union

from natural_pdf.elements.element_collection import ElementCollection

from ._grid_builder import remove_temporary_grid_regions
from ._targets import resolve_cleanup_pages

if TYPE_CHECKING:
    from natural_pdf.core.page import Page
    from natural_pdf.core.page_collection import PageCollection
    from natural_pdf.elements.region import Region
    from natural_pdf.tables.result import TableResult


def extract_table_from_guides(
    guides: Any,
    *,
    target: Optional[
        Union[
            "Page",
            "Region",
            "PageCollection",
            "ElementCollection",
            List[Union["Page", "Region"]],
        ]
    ] = None,
    source: str = "guides_temp",
    cell_padding: float = 0.5,
    include_outer_boundaries: bool = False,
    method: Optional[str] = None,
    table_settings: Optional[dict] = None,
    use_ocr: bool = False,
    ocr_config: Optional[dict] = None,
    text_options: Optional[Dict] = None,
    cell_extraction_func: Optional[Callable[["Region"], Optional[str]]] = None,
    show_progress: bool = False,
    content_filter: Optional[Union[str, Callable[[str], bool], List[str]]] = None,
    apply_exclusions: bool = True,
    multi_page: Literal["auto", True, False] = "auto",
    header: Union[str, List[str], None] = "first",
    skip_repeating_headers: Optional[bool] = None,
    structure_engine: Optional[str] = None,
) -> "TableResult":
    from natural_pdf.core.page_collection import PageCollection

    target_obj = target if target is not None else guides.context
    if target_obj is None:
        raise ValueError("No target object available. Provide target parameter or context.")

    if isinstance(target_obj, (PageCollection, ElementCollection, list)):
        return guides._extract_table_from_collection(
            elements=target_obj,
            header=header,
            skip_repeating_headers=skip_repeating_headers,
            method=method,
            table_settings=table_settings,
            use_ocr=use_ocr,
            ocr_config=ocr_config,
            text_options=text_options,
            cell_extraction_func=cell_extraction_func,
            show_progress=show_progress,
            content_filter=content_filter,
            apply_exclusions=apply_exclusions,
            structure_engine=structure_engine,
        )

    cleanup_pages = resolve_cleanup_pages(target_obj)

    has_verticals = len(guides.vertical) > 0
    has_horizontals = len(guides.horizontal) > 0

    if (has_verticals and not has_horizontals) or (has_horizontals and not has_verticals):
        return guides._extract_with_table_service(
            target_obj,
            method=method,
            table_settings=table_settings,
            use_ocr=use_ocr,
            ocr_config=ocr_config,
            text_options=text_options,
            cell_extraction_func=cell_extraction_func,
            show_progress=show_progress,
            content_filter=content_filter,
            apply_exclusions=apply_exclusions,
            verticals=list(guides.vertical) if has_verticals else None,
            horizontals=list(guides.horizontal) if has_horizontals else None,
            structure_engine=structure_engine,
        )

    try:
        grid_result = guides.build_grid(
            target=target_obj,
            source=source,
            cell_padding=cell_padding,
            include_outer_boundaries=include_outer_boundaries,
            multi_page=multi_page,
        )

        table_region = grid_result["regions"]["table"]
        if table_region is None:
            raise ValueError(
                "No table region was created from the guides. Check that you have both vertical and horizontal guides."
            )

        if isinstance(table_region, list):
            if not table_region:
                raise ValueError("No table regions were created from the guides.")
            table_region = table_region[0]

        table_result = guides._extract_with_table_service(
            table_region,
            method=method,
            table_settings=table_settings,
            use_ocr=use_ocr,
            ocr_config=ocr_config,
            text_options=text_options,
            cell_extraction_func=cell_extraction_func,
            show_progress=show_progress,
            content_filter=content_filter,
            apply_exclusions=apply_exclusions,
            structure_engine=structure_engine,
        )
        guides._assign_headers_from_rows(table_result, header)
        return table_result
    finally:
        remove_temporary_grid_regions(cleanup_pages, source=source)
