"""
Export structured extraction results as annotated PDFs with highlight annotations.

Each field's citation elements become /Highlight annotations on the
corresponding PDF pages, preserving the native text layer.  A native
vector sidebar legend (colored rectangles + Helvetica text) is drawn
on each annotated page.
"""

import logging
import unicodedata
from collections import OrderedDict, defaultdict
from typing import TYPE_CHECKING, Dict, List

from natural_pdf.utils.optional_imports import require
from natural_pdf.utils.visualization import _BASE_HIGHLIGHT_COLORS, DEFAULT_FILL_ALPHA

if TYPE_CHECKING:
    from natural_pdf.extraction.result import FieldResult

logger = logging.getLogger(__name__)

SIDEBAR_WIDTH = 250  # PDF points
_FONT_SIZE = 11
_LINE_HEIGHT = 14  # vertical advance per text line
_SWATCH_SIZE = 12
_PADDING_TOP = 10
_ITEM_GAP = 6
_SWATCH_X_OFFSET = 10  # from sidebar left edge
_TEXT_X_OFFSET = 30  # from sidebar left edge


def _pdf_escape(text: str) -> str:
    """Escape and sanitise a string for use in a PDF string literal ``(...)``.

    Handles backslashes, parentheses, newlines, carriage returns, and
    non-ASCII characters (normalised to closest ASCII via NFKD + stripped).
    """
    # Normalise unicode to closest ASCII (e.g. é → e, € → EUR)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = text.strip()
    return (
        text.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )


def _draw_sidebar_legend(target_doc, target_page, legend_items, sidebar_width):
    """Draw a native vector sidebar legend on the right side of a PDF page.

    Extends the page's MediaBox (and CropBox if present), draws a white
    background, colored rectangles, and Helvetica text for each field.

    Args:
        target_doc: The pikepdf.Pdf document.
        target_page: The pikepdf page to modify.
        legend_items: List of dicts with 'label' (str with \\n) and 'rgba' tuple.
        sidebar_width: Width in PDF points for the sidebar.
    """
    pikepdf = require("pikepdf")
    from pikepdf import Array, Dictionary, Name

    from natural_pdf.exporters.region_pdf import _append_content_stream

    # Read current page box
    if "/CropBox" in target_page:
        page_box = target_page.CropBox
    else:
        page_box = target_page.MediaBox
    box_x0, box_y0, box_x1, box_y1 = [float(c) for c in page_box]
    page_height = box_y1 - box_y0

    # Extend page boxes to the right
    target_page.MediaBox = Array(
        [
            float(target_page.MediaBox[0]),
            float(target_page.MediaBox[1]),
            float(target_page.MediaBox[2]) + sidebar_width,
            float(target_page.MediaBox[3]),
        ]
    )
    if "/CropBox" in target_page:
        target_page.CropBox = Array(
            [
                float(target_page.CropBox[0]),
                float(target_page.CropBox[1]),
                float(target_page.CropBox[2]) + sidebar_width,
                float(target_page.CropBox[3]),
            ]
        )

    # Register Helvetica font in page Resources
    if "/Resources" not in target_page:
        target_page["/Resources"] = Dictionary()
    if "/Font" not in target_page.Resources:
        target_page.Resources["/Font"] = Dictionary()
    if "/HelvLegend" not in target_page.Resources.Font:
        font_dict = Dictionary(
            Type=Name.Font,
            Subtype=Name.Type1,
            BaseFont=Name.Helvetica,
        )
        target_page.Resources.Font["/HelvLegend"] = target_doc.make_indirect(font_dict)

    # Build the content stream
    sidebar_x = box_x1
    ops = ["q"]

    # White background for entire sidebar
    ops.append("1 1 1 rg")
    ops.append(f"{sidebar_x:.2f} {box_y0:.2f} {sidebar_width:.2f} {page_height:.2f} re f")

    # Draw each legend item top-down
    # PDF y-axis goes up, so we start from box_y1 (top) and subtract
    cursor_y = box_y1 - _PADDING_TOP

    for item in legend_items:
        label = item["label"]
        r, g, b, _alpha = item["rgba"]

        lines = label.split("\n")

        # Draw color swatch aligned to first line of text
        swatch_x = sidebar_x + _SWATCH_X_OFFSET
        swatch_top = cursor_y
        swatch_bottom = swatch_top - _SWATCH_SIZE
        ops.append(f"{r / 255.0:.4f} {g / 255.0:.4f} {b / 255.0:.4f} rg")
        ops.append(
            f"{swatch_x:.2f} {swatch_bottom:.2f} " f"{_SWATCH_SIZE:.2f} {_SWATCH_SIZE:.2f} re f"
        )

        # Draw text lines
        text_x = sidebar_x + _TEXT_X_OFFSET
        ops.append("0 0 0 rg")  # black text
        ops.append("BT")
        ops.append(f"/HelvLegend {_FONT_SIZE} Tf")
        ops.append("0 Tr")  # Fill mode (reset from possible invisible text layer)
        # Position at first line baseline (approx font_size * 0.8 below top)
        baseline_y = cursor_y - _FONT_SIZE * 0.8
        ops.append(f"{text_x:.2f} {baseline_y:.2f} Td")
        ops.append(f"({_pdf_escape(lines[0])}) Tj")
        for line in lines[1:]:
            ops.append(f"0 -{_LINE_HEIGHT} Td")
            ops.append(f"({_pdf_escape(line)}) Tj")
        ops.append("ET")

        # Advance cursor past all lines + gap
        total_item_height = max(_SWATCH_SIZE, _LINE_HEIGHT * len(lines))
        cursor_y -= total_item_height + _ITEM_GAP

    ops.append("Q")

    stream_data = "\n".join(ops).encode("utf-8")
    content_stream = pikepdf.Stream(target_doc, stream_data)
    _append_content_stream(target_page, content_stream)


def create_annotated_pdf(
    fields: Dict[str, "FieldResult"],
    output_path: str,
    pages: str = "all",
) -> None:
    """Create a PDF with /Highlight annotations and a native sidebar legend.

    For each field with citations, adds highlight annotations on the
    corresponding pages and draws a sidebar legend with colored rectangles
    and text labels using standard PDF drawing operators.

    Args:
        fields: Dict mapping field names to :class:`FieldResult` objects.
        output_path: Path to write the annotated PDF.

    Raises:
        ValueError: If no citation elements are found.
        ImportError: If pikepdf is not installed.
    """
    pikepdf = require("pikepdf")
    from pikepdf import Array, Dictionary, Name

    from natural_pdf.exporters.region_pdf import _open_source_pdf, _translate_bbox_to_pdf_coords
    from natural_pdf.extraction.result import build_enriched_label

    # Collect highlight info grouped by page
    page_annotations: Dict[int, list] = defaultdict(list)
    # Track which fields (with colors) appear on each page, preserving order
    page_fields: Dict[int, OrderedDict] = defaultdict(OrderedDict)
    source_page_obj = None

    color_cycle = list(_BASE_HIGHLIGHT_COLORS)

    # Try to reuse colors assigned during .show() so PDF matches interactive view
    interactive_colors: dict = {}
    try:
        for fr in fields.values():
            if fr.citations and len(fr.citations) > 0:
                page = getattr(fr.citations[0], "page", None)
                if page and hasattr(page, "_pdf"):
                    hs = getattr(page._pdf, "highlighter", None)
                    if hs:
                        interactive_colors = hs.get_labels_and_colors()
                    break
    except Exception:
        pass

    for field_idx, (field_name, fr) in enumerate(fields.items()):
        if fr.citations is None or len(fr.citations) == 0:
            continue

        # Check if .show() already assigned a color for this field's label
        label = build_enriched_label(field_name, fr.value)
        existing = interactive_colors.get(label)
        rgb = existing[:3] if existing else color_cycle[field_idx % len(color_cycle)]
        # Compute pastel color by alpha-blending with white, matching .show() appearance
        blend = DEFAULT_FILL_ALPHA / 255.0
        pastel_rgb = tuple(int(c * blend + 255 * (1 - blend)) for c in rgb)
        # Use moderate opacity for PDF highlights (colors are already soft)
        pdf_alpha = 128
        rgba = (*pastel_rgb, 255)  # Full opacity for legend swatches

        # Build popup text for highlight annotation
        parts = [f"{field_name}: {fr.value}"]
        if fr.confidence is not None:
            parts.append(f"confidence: {fr.confidence}")
        popup_text = "\n".join(parts)

        for elem in fr.citations:
            page = getattr(elem, "page", None)
            if page is None:
                continue
            if source_page_obj is None:
                source_page_obj = page

            bbox = elem.bbox
            page_annotations[page.index].append(
                {
                    "bbox": bbox,
                    "rgb": pastel_rgb,
                    "alpha": pdf_alpha,
                    "field_name": field_name,
                    "popup_text": popup_text,
                }
            )

            # Track field for this page's legend
            if field_name not in page_fields[page.index]:
                page_fields[page.index][field_name] = {
                    "label": label,
                    "rgba": rgba,
                }

    if source_page_obj is None:
        raise ValueError("No citation elements with page references found.")

    # Open source PDF
    source_cache: Dict[str, object] = {}
    source_doc = _open_source_pdf(source_page_obj, cache=source_cache)

    target_doc = pikepdf.Pdf.new()

    # Determine which source pages to include
    cited_indices = set(page_annotations.keys())
    if pages == "cited":
        include_indices = sorted(cited_indices)
    else:
        include_indices = list(range(len(source_doc.pages)))

    # Map from original page index to target page index
    idx_map = {}
    try:
        for target_idx, source_idx in enumerate(include_indices):
            target_doc.pages.append(source_doc.pages[source_idx])
            idx_map[source_idx] = target_idx

        # Add annotations + sidebar per page
        for page_idx, annots_info in page_annotations.items():
            if page_idx not in idx_map:
                logger.warning(f"Page index {page_idx} not in output. Skipping annotations.")
                continue

            target_page = target_doc.pages[idx_map[page_idx]]

            # --- Highlight annotations ---
            if "/Annots" in target_page:
                annots = target_page.Annots
            else:
                annots = Array()
                target_page.Annots = annots

            for info in annots_info:
                pdf_bbox = _translate_bbox_to_pdf_coords(info["bbox"], target_page)
                pdf_x0, pdf_y0, pdf_x1, pdf_y1 = pdf_bbox

                r, g, b = info["rgb"]

                quad_points = Array(
                    [
                        pdf_x0,
                        pdf_y1,
                        pdf_x1,
                        pdf_y1,
                        pdf_x0,
                        pdf_y0,
                        pdf_x1,
                        pdf_y0,
                    ]
                )

                annot = Dictionary(
                    Type=Name.Annot,
                    Subtype=Name.Highlight,
                    Rect=Array([pdf_x0, pdf_y0, pdf_x1, pdf_y1]),
                    QuadPoints=quad_points,
                    C=Array([r / 255.0, g / 255.0, b / 255.0]),
                    CA=info["alpha"] / 255.0,
                    T=pikepdf.String(info["field_name"]),
                    Contents=pikepdf.String(info["popup_text"]),
                    F=4,
                )
                annots.append(target_doc.make_indirect(annot))

            # --- Sidebar legend ---
            pf = page_fields.get(page_idx, {})
            if pf:
                legend_items = list(pf.values())
                _draw_sidebar_legend(target_doc, target_page, legend_items, SIDEBAR_WIDTH)

        target_doc.save(str(output_path))
        logger.info(f"Saved annotated PDF to: {output_path}")

    finally:
        for doc in source_cache.values():
            try:
                doc.close()
            except Exception:
                pass
