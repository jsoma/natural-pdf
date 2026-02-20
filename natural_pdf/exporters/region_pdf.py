"""
Module for exporting PDF regions (sub-page areas) as new PDF files.

Supports two methods:
- 'crop': Sets CropBox on copied pages so output is sized to the region.
- 'whiteout': Overlays white rectangles outside the region on full-size pages.

Also supports exclusion-aware PDF export that whites out exclusion zones.
"""

import io
import logging
import os
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from natural_pdf.utils.optional_imports import require

if TYPE_CHECKING:
    from natural_pdf.core.page import Page

logger = logging.getLogger(__name__)


def _open_source_pdf(page: "Page", cache: Optional[Dict[str, object]] = None):
    """
    Open the source PDF for a page via pikepdf.

    Tries filesystem path first, falls back to in-memory bytes, then URL download.
    Results are cached by PDF path in the provided cache dict.

    Args:
        page: A Page object whose source PDF should be opened.
        cache: Optional dict mapping PDF path -> open pikepdf.Pdf for reuse.

    Returns:
        An open pikepdf.Pdf document.
    """
    pikepdf = require("pikepdf")

    pdf_obj = page.pdf
    pdf_path = getattr(pdf_obj, "path", None)

    # Check cache
    cache_key = str(pdf_path) if pdf_path else str(id(pdf_obj))
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    source_doc = None

    # Try filesystem path
    if pdf_path and os.path.exists(str(pdf_path)):
        source_doc = pikepdf.Pdf.open(str(pdf_path))
    else:
        # Try in-memory bytes
        if hasattr(pdf_obj, "_original_bytes") and pdf_obj._original_bytes:
            source_doc = pikepdf.Pdf.open(io.BytesIO(pdf_obj._original_bytes))
        elif isinstance(pdf_path, str) and pdf_path.startswith(("http://", "https://")):
            try:
                with urllib.request.urlopen(pdf_path) as resp:
                    data = resp.read()
                source_doc = pikepdf.Pdf.open(io.BytesIO(data))
            except Exception as dl_err:
                raise FileNotFoundError(
                    f"Source PDF download failed for {pdf_path}: {dl_err}"
                ) from dl_err
        else:
            raise FileNotFoundError(f"Cannot open source PDF: path={pdf_path}")

    if cache is not None:
        cache[cache_key] = source_doc

    return source_doc


def _translate_bbox_to_pdf_coords(
    plumber_bbox: Tuple[float, float, float, float],
    pikepdf_page,
) -> Tuple[float, float, float, float]:
    """
    Translate pdfplumber coordinates (top-left origin, relative to effective page box)
    to PDF coordinates (bottom-left origin, absolute).

    Args:
        plumber_bbox: (x0, top, x1, bottom) in pdfplumber coordinates.
        pikepdf_page: The pikepdf page object to read CropBox/MediaBox from.

    Returns:
        (pdf_x0, pdf_y0, pdf_x1, pdf_y1) in PDF coordinate space where y0 < y1.
    """
    pikepdf = require("pikepdf")

    # Get the effective page box (CropBox if present, else MediaBox)
    if "/CropBox" in pikepdf_page:
        page_box = pikepdf_page.CropBox
    else:
        page_box = pikepdf_page.MediaBox
    box_x0, box_y0, box_x1, box_y1 = [float(c) for c in page_box]

    plumber_x0, plumber_top, plumber_x1, plumber_bottom = plumber_bbox

    # pdfplumber coords are relative to top-left of the effective page box
    # PDF coords have origin at bottom-left
    pdf_x0 = plumber_x0 + box_x0
    pdf_x1 = plumber_x1 + box_x0
    pdf_y0 = box_y1 - plumber_bottom  # bottom in pdfplumber -> low-y in PDF
    pdf_y1 = box_y1 - plumber_top  # top in pdfplumber -> high-y in PDF

    return (pdf_x0, pdf_y0, pdf_x1, pdf_y1)


def _append_content_stream(pikepdf_page, new_stream):
    """
    Defensively append a content stream to a pikepdf page.

    Handles pages with no Contents, a single Stream, or an Array of Streams.
    """
    pikepdf = require("pikepdf")

    if "/Contents" not in pikepdf_page:
        pikepdf_page.Contents = new_stream
    elif isinstance(pikepdf_page.Contents, pikepdf.Stream):
        pikepdf_page.Contents = pikepdf.Array([pikepdf_page.Contents, new_stream])
    elif isinstance(pikepdf_page.Contents, pikepdf.Array):
        pikepdf_page.Contents.append(new_stream)
    else:
        # Fallback: replace
        pikepdf_page.Contents = new_stream


def _build_whiteout_stream(
    pikepdf_page,
    region_pdf_bbox: Tuple[float, float, float, float],
    target_doc,
) -> object:
    """
    Build a PDF content stream that draws white rectangles over the areas
    outside the given region bbox.

    Args:
        pikepdf_page: The pikepdf page to read dimensions from.
        region_pdf_bbox: (x0, y0, x1, y1) in PDF coordinates of the region to keep.
        target_doc: The pikepdf.Pdf document to create the stream in.

    Returns:
        A pikepdf.Stream object with the whiteout drawing commands.
    """
    pikepdf = require("pikepdf")

    # Get full page box
    if "/CropBox" in pikepdf_page:
        page_box = pikepdf_page.CropBox
    else:
        page_box = pikepdf_page.MediaBox
    px0, py0, px1, py1 = [float(c) for c in page_box]

    rx0, ry0, rx1, ry1 = region_pdf_bbox

    # Build rectangles for the 4 strips outside the region
    rects = []

    # Bottom strip (below region)
    if ry0 > py0:
        rects.append((px0, py0, px1 - px0, ry0 - py0))

    # Top strip (above region)
    if ry1 < py1:
        rects.append((px0, ry1, px1 - px0, py1 - ry1))

    # Left strip (between bottom and top strips, left of region)
    if rx0 > px0:
        rects.append((px0, ry0, rx0 - px0, ry1 - ry0))

    # Right strip (between bottom and top strips, right of region)
    if rx1 < px1:
        rects.append((rx1, ry0, px1 - rx1, ry1 - ry0))

    if not rects:
        # Region covers the full page, no whiteout needed
        return None

    # Build the content stream: save state, set white fill, draw rects, fill, restore
    parts = ["q", "1 1 1 rg"]
    for x, y, w, h in rects:
        parts.append(f"{x:.4f} {y:.4f} {w:.4f} {h:.4f} re")
    parts.append("f")
    parts.append("Q")
    stream_data = " ".join(parts).encode("ascii")

    return pikepdf.Stream(target_doc, stream_data)


def create_region_pdf(
    regions: List[Tuple["Page", Tuple[float, float, float, float]]],
    output_path: Union[str, Path],
    method: str = "crop",
):
    """
    Create a PDF from a list of (Page, bbox) tuples. Each region becomes one page.

    Args:
        regions: List of (Page, bbox) where bbox is (x0, top, x1, bottom) in
                 pdfplumber coordinates.
        output_path: Path to save the output PDF.
        method: 'crop' to set CropBox (default), or 'whiteout' to overlay white
                rectangles outside each region.

    Raises:
        ValueError: If regions list is empty or method is invalid.
        ImportError: If pikepdf is not installed.
    """
    if method not in ("crop", "whiteout"):
        raise ValueError(f"Invalid method '{method}'. Must be 'crop' or 'whiteout'.")

    if not regions:
        raise ValueError("No regions provided for PDF export.")

    pikepdf = require("pikepdf")
    output_path_str = str(output_path)

    # Cache open source documents by PDF path
    source_cache: Dict[str, object] = {}
    target_doc = pikepdf.Pdf.new()

    try:
        for page, plumber_bbox in regions:
            source_doc = _open_source_pdf(page, cache=source_cache)
            page_index = page.index

            if page_index < 0 or page_index >= len(source_doc.pages):
                logger.warning(f"Page index {page_index} out of bounds for source PDF. Skipping.")
                continue

            # Copy the source page into the target document
            target_doc.pages.append(source_doc.pages[page_index])
            target_page = target_doc.pages[-1]

            # Translate pdfplumber bbox to PDF coordinates
            pdf_bbox = _translate_bbox_to_pdf_coords(plumber_bbox, target_page)

            if method == "crop":
                # Set CropBox to the region bounds
                target_page.CropBox = pikepdf.Array([float(v) for v in pdf_bbox])

            elif method == "whiteout":
                # Draw white rectangles over areas outside the region
                stream = _build_whiteout_stream(target_page, pdf_bbox, target_doc)
                if stream is not None:
                    _append_content_stream(target_page, stream)

        if not target_doc.pages:
            raise RuntimeError("No valid pages were produced for the output PDF.")

        target_doc.save(output_path_str)
        logger.info(
            f"Saved region PDF ({len(target_doc.pages)} pages, method={method}) to: {output_path_str}"
        )

    finally:
        # Close all cached source documents
        for doc in source_cache.values():
            try:
                doc.close()
            except Exception:
                pass


def create_exclusion_aware_pdf(
    pages: List["Page"],
    output_path: Union[str, Path],
):
    """
    Create a PDF that whites out exclusion zones on each page.

    For each page, resolves exclusion regions via page._get_exclusion_regions()
    and overlays white rectangles over each exclusion bbox.

    Args:
        pages: List of Page objects to include.
        output_path: Path to save the output PDF.

    Raises:
        ValueError: If pages list is empty.
        ImportError: If pikepdf is not installed.
    """
    if not pages:
        raise ValueError("No pages provided for exclusion-aware PDF export.")

    pikepdf = require("pikepdf")
    output_path_str = str(output_path)

    source_cache: Dict[str, object] = {}
    target_doc = pikepdf.Pdf.new()

    try:
        for page in pages:
            source_doc = _open_source_pdf(page, cache=source_cache)
            page_index = page.index

            if page_index < 0 or page_index >= len(source_doc.pages):
                logger.warning(f"Page index {page_index} out of bounds for source PDF. Skipping.")
                continue

            # Copy the source page
            target_doc.pages.append(source_doc.pages[page_index])
            target_page = target_doc.pages[-1]

            # Get exclusion regions for this page
            exclusion_regions = page._get_exclusion_regions()

            if not exclusion_regions:
                continue

            # Build whiteout rectangles for each exclusion zone
            all_rects = []
            for exc_region in exclusion_regions:
                exc_bbox = exc_region.bbox  # (x0, top, x1, bottom) in pdfplumber coords
                pdf_bbox = _translate_bbox_to_pdf_coords(exc_bbox, target_page)
                # Each exclusion becomes a single rectangle to white out
                rx0, ry0, rx1, ry1 = pdf_bbox
                w = rx1 - rx0
                h = ry1 - ry0
                if w > 0 and h > 0:
                    all_rects.append((rx0, ry0, w, h))

            if not all_rects:
                continue

            # Build a single content stream for all exclusion whiteouts on this page
            parts = ["q", "1 1 1 rg"]
            for x, y, w, h in all_rects:
                parts.append(f"{x:.4f} {y:.4f} {w:.4f} {h:.4f} re")
            parts.append("f")
            parts.append("Q")
            stream_data = " ".join(parts).encode("ascii")

            stream = pikepdf.Stream(target_doc, stream_data)
            _append_content_stream(target_page, stream)

        if not target_doc.pages:
            raise RuntimeError("No valid pages were produced for the exclusion-aware PDF.")

        target_doc.save(output_path_str)
        logger.info(
            f"Saved exclusion-aware PDF ({len(target_doc.pages)} pages) to: {output_path_str}"
        )

    finally:
        for doc in source_cache.values():
            try:
                doc.close()
            except Exception:
                pass
