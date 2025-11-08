"""
Utilities for packaging data for external processes, like correction tasks.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import zipfile
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Tuple, Union, cast

from PIL import Image  # type: ignore[import-untyped]
from tqdm import tqdm

from natural_pdf.core.page import Page
from natural_pdf.core.pdf import PDF
from natural_pdf.core.pdf_collection import PDFCollection
from natural_pdf.elements.region import Region
from natural_pdf.elements.text import TextElement
from natural_pdf.utils.identifiers import generate_short_path_hash

PDFSource = Union[PDF, PDFCollection, Sequence[PDF]]
SuggestFunction = Callable[[Region, Optional[float]], Optional[str]]


logger = logging.getLogger(__name__)


def create_correction_task_package(
    source: PDFSource,
    output_zip_path: str,
    overwrite: bool = False,
    suggest: Optional[SuggestFunction] = None,
    resolution: int = 300,
) -> None:
    """
    Creates a zip package containing data for an OCR correction task.

    The package includes:
    - manifest.json: Metadata about pages and OCR regions (using original PDF coordinates).
    - images/ directory: Rendered full-page images.

    Args:
        source: The PDF object, PDFCollection, or list of PDF objects to process.
        output_zip_path: The full path where the output zip file should be saved.
        overwrite: If True, overwrite the output zip file if it already exists.
        suggest: Function that takes the region and returns an OCR suggestion

    Raises:
        FileNotFoundError: If the output directory cannot be created.
        FileExistsError: If the output zip file exists and overwrite is False.
        TypeError: If the source type is invalid.
        ValueError: If no valid pages with OCR data are found in the source.
    """
    if os.path.exists(output_zip_path) and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_zip_path}. Set overwrite=True to replace it."
        )

    # --- Resolve source to a list of PDF objects ---
    pdfs_to_process: List[PDF] = []
    if isinstance(source, PDF):
        pdfs_to_process = [source]
    elif isinstance(source, PDFCollection):
        pdfs_to_process = list(source.pdfs)
    elif isinstance(source, Sequence):
        if isinstance(source, (str, bytes)):
            raise TypeError(
                "String-like sources are not supported for correction packaging; provide PDF instances instead."
            )
        pdf_candidates = list(source)
        if not all(isinstance(item, PDF) for item in pdf_candidates):
            raise TypeError(
                "All items provided in the source sequence must be natural_pdf PDF instances."
            )
        pdfs_to_process = [cast(PDF, item) for item in pdf_candidates]
    else:
        raise TypeError(
            f"Unsupported source type: {type(source)}. Expected PDF, PDFCollection, or a sequence of PDF instances."
        )

    if not pdfs_to_process:
        logger.warning("No PDF documents provided in the source.")
        return

    manifest_data: Dict[str, List[Dict[str, Any]]] = {"pdfs": [], "pages": []}
    total_regions_found = 0

    # Use a temporary directory for staging files before zipping
    with tempfile.TemporaryDirectory() as temp_dir:
        images_dir = os.path.join(temp_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        logger.info(f"Using temporary directory for staging: {temp_dir}")

        # --- Process each PDF ---
        for pdf in pdfs_to_process:
            if not hasattr(pdf, "path") or not hasattr(pdf, "pages"):
                logger.warning(f"Skipping invalid PDF object: {pdf}")
                continue

            pdf_path = cast(str, getattr(pdf, "path"))
            pdf_short_id = generate_short_path_hash(pdf_path)
            logger.debug(f"Processing PDF: {pdf_path} (ID: {pdf_short_id})")

            for page in cast(Iterable[Page], getattr(pdf, "pages", [])):
                if (
                    not hasattr(page, "index")
                    or not hasattr(page, "number")
                    or not hasattr(page, "width")
                    or not hasattr(page, "height")
                    or not hasattr(page, "find_all")
                    or not hasattr(page, "to_image")
                ):
                    logger.warning(
                        f"Skipping invalid Page object in {pdf_path}: page index {getattr(page, 'index', 'unknown')}"
                    )
                    continue

                # 1. Extract OCR elements for this page
                try:
                    query_result = page.find_all("text[source=ocr]", apply_exclusions=False)
                    raw_elements = getattr(query_result, "elements", [])
                    ocr_elements = [
                        cast(TextElement, elem)
                        for elem in raw_elements
                        if isinstance(elem, TextElement)
                    ]
                except Exception as e:
                    logger.error(
                        f"Failed to extract OCR elements for {pdf_path} page {page.number}: {e}",
                        exc_info=True,
                    )
                    continue

                if not ocr_elements:
                    logger.debug(
                        f"No OCR elements found for {pdf_path} page {page.number}. Skipping page in manifest."
                    )
                    continue

                logger.debug(f"  Found {len(ocr_elements)} OCR elements on page {page.number}")
                total_regions_found += len(ocr_elements)

                # 2. Render and save page image
                image_filename = f"{pdf_short_id}_page_{page.index}.png"
                image_save_path = os.path.join(images_dir, image_filename)

                rendered_image: Optional[Image.Image] = None
                try:
                    rendered = page.render(resolution=resolution)
                    if not isinstance(rendered, Image.Image):
                        raise ValueError("page.render returned None")
                    rendered_image = rendered
                    rendered_image.save(image_save_path, "PNG")
                except Exception as e:
                    logger.error(
                        f"Failed to render/save image for {pdf_path} page {page.number}: {e}",
                        exc_info=True,
                    )
                    # If image fails, we cannot proceed with this page for the task
                    continue
                if rendered_image is None:
                    continue
                assert rendered_image is not None

                # 3. Prepare region data for manifest
                page_regions_data: List[Dict[str, Any]] = []
                # Calculate scaling factor *from PDF points* to *actual image pixels*.
                # We prefer using the rendered image dimensions rather than the nominal
                # resolution value, because the image might have been resized (e.g. via
                # global `natural_pdf.options.image.width`). This guarantees that the
                # bounding boxes we write to the manifest always align with the exact
                # pixel grid of the exported image.

                try:
                    page_width = float(getattr(page, "width", 0.0) or 0.0)
                    page_height = float(getattr(page, "height", 0.0) or 0.0)
                    scale_x = rendered_image.width / page_width if page_width else 1.0
                    scale_y = rendered_image.height / page_height if page_height else 1.0
                except Exception as e:
                    logger.warning(
                        f"Could not compute per-axis scale factors for page {page.number}: {e}. "
                        "Falling back to resolution-based scaling."
                    )
                    scale_x = scale_y = resolution / 72.0

                for elem_index, elem in enumerate(tqdm(ocr_elements, leave=False), start=0):
                    # Basic check for necessary attributes
                    if not all(
                        hasattr(elem, attr) for attr in ["x0", "top", "x1", "bottom", "text"]
                    ):
                        logger.warning(
                            f"Skipping invalid OCR element {elem_index} on {pdf_path} page {page.number}"
                        )
                        continue
                    region_id = f"r_{page.index}_{elem_index}"  # ID unique within page

                    # Scale coordinates to match the **actual** image dimensions.
                    scaled_bbox = [
                        float(elem.x0) * scale_x,
                        float(elem.top) * scale_y,
                        float(elem.x1) * scale_x,
                        float(elem.bottom) * scale_y,
                    ]

                    corrected_text = elem.text
                    if suggest is not None:
                        try:
                            suggestion = suggest(
                                elem.to_region(), getattr(elem, "confidence", None)
                            )
                        except (
                            Exception
                        ) as suggest_error:  # pragma: no cover - user supplied callback
                            logger.warning(
                                "Suggestion callback raised an error for %s page %s region %s: %s",
                                pdf_path,
                                page.number,
                                region_id,
                                suggest_error,
                            )
                            suggestion = None
                        if suggestion is not None:
                            corrected_text = suggestion

                    page_regions_data.append(
                        {
                            "resolution": float(scale_x * 72.0),
                            "id": region_id,
                            "bbox": scaled_bbox,
                            "ocr_text": elem.text,
                            "confidence": getattr(
                                elem, "confidence", None
                            ),  # Include confidence if available
                            "corrected_text": corrected_text,
                            "modified": False,
                        }
                    )

                # 4. Add page data to manifest if it has regions
                if page_regions_data:
                    manifest_data["pages"].append(
                        {
                            "pdf_source": pdf_path,
                            "pdf_short_id": pdf_short_id,
                            "page_number": page.number,
                            "page_index": page.index,
                            "image_path": f"images/{image_filename}",  # Relative path within zip
                            "width": page.width,
                            "height": page.height,
                            "regions": page_regions_data,
                        }
                    )

        # --- Final Checks and Zipping ---
        if not manifest_data["pages"] or total_regions_found == 0:
            logger.error(
                "No pages with valid OCR regions and successfully rendered images found in the source PDFs. Cannot create task package."
            )
            # Consider raising ValueError here instead of just returning
            raise ValueError("No valid pages with OCR data found to create a task package.")

        manifest_path = os.path.join(temp_dir, "manifest.json")
        try:
            with open(manifest_path, "w", encoding="utf-8") as f_manifest:
                json.dump(manifest_data, f_manifest, indent=2)
        except Exception as e:
            logger.error(f"Failed to write manifest.json: {e}", exc_info=True)
            raise  # Re-raise error, cannot proceed

        # --- Copy SPA files into temp dir ---
        try:
            # Find the path to the spa template directory relative to this file
            # Using __file__ assumes this script is installed alongside the templates
            utils_dir = os.path.dirname(os.path.abspath(__file__))
            templates_dir = os.path.join(
                os.path.dirname(utils_dir), "templates"
            )  # Go up one level from utils
            spa_template_dir = os.path.join(templates_dir, "spa")

            if not os.path.isdir(spa_template_dir):
                raise FileNotFoundError(f"SPA template directory not found at {spa_template_dir}")

            logger.info(f"Copying SPA shell from: {spa_template_dir}")
            # Copy contents of spa_template_dir/* into temp_dir/
            # dirs_exist_ok=True handles merging if subdirs like js/ already exist (Python 3.8+)
            shutil.copytree(spa_template_dir, temp_dir, dirs_exist_ok=True)

        except Exception as e:
            logger.error(f"Failed to copy SPA template files: {e}", exc_info=True)
            raise RuntimeError("Could not package SPA files.") from e

        # --- Create the final zip file ---
        try:
            logger.info(f"Creating zip package at: {output_zip_path}")
            with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                # Add manifest.json
                zipf.write(manifest_path, arcname="manifest.json")
                # Add images directory
                for root, _, files in os.walk(images_dir):
                    for file in files:
                        full_path = os.path.join(root, file)
                        # Create the correct archive name (e.g., images/...)
                        arcname = os.path.relpath(full_path, temp_dir)
                        zipf.write(full_path, arcname=arcname)
            logger.info(
                f"Successfully created correction task package: {output_zip_path} ({total_regions_found} regions total)"
            )

        except Exception as e:
            logger.error(f"Failed to create zip file {output_zip_path}: {e}", exc_info=True)
            # Attempt to clean up existing zip if creation failed partially
            if os.path.exists(output_zip_path):
                try:
                    os.remove(output_zip_path)
                except:
                    pass
            raise  # Re-raise error

    # Temporary directory is automatically cleaned up by context manager


def import_ocr_from_manifest(pdf: "PDF", manifest_path: str) -> Dict[str, int]:
    """
    Imports OCR data into a PDF object from a manifest file.

    Reads a manifest.json file (typically generated by create_correction_task_package
    and potentially modified externally) and populates the corresponding pages
    of the PDF object with new TextElement objects based on the manifest data.
    It uses the 'corrected_text' field and bounding box from the manifest.

    This function assumes you want to replace or provide the primary OCR data
    from the manifest, rather than correcting existing elements.
    Existing OCR elements on the pages are NOT automatically cleared.

    Args:
        pdf: The natural_pdf.core.pdf.PDF object to populate with OCR data.
        manifest_path: Path to the manifest.json file.

    Returns:
        A dictionary containing counts of imported and skipped regions:
        {'imported': count, 'skipped': count}

    Raises:
        FileNotFoundError: If the manifest_path does not exist.
        ValueError: If the manifest is invalid or contains data for a different PDF.
        TypeError: If the input pdf object is not a valid PDF instance.
    """
    if not isinstance(pdf, PDF):
        raise TypeError(f"Input must be a natural_pdf PDF object, got {type(pdf)}")

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    logger.info(f"Importing OCR data into PDF '{pdf.path}' from manifest '{manifest_path}'")

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse manifest file: {e}")
        raise ValueError(f"Invalid JSON in manifest file: {manifest_path}") from e
    except Exception as e:
        logger.error(f"Failed to read manifest file: {e}")
        raise

    imported_count = 0
    skipped_count = 0
    processed_pages = 0

    manifest_pages = manifest_data.get("pages", [])
    if not manifest_pages:
        logger.warning("Manifest contains no page data.")
        return {"imported": 0, "skipped": 0}

    # --- Pre-check PDF source consistency ---
    first_manifest_pdf_path = manifest_pages[0].get("pdf_source")
    if first_manifest_pdf_path != pdf.path:
        # Allow matching based on just the filename if paths differ (e.g., absolute vs relative)
        if os.path.basename(first_manifest_pdf_path) != os.path.basename(pdf.path):
            logger.error(
                f"Manifest PDF source ('{first_manifest_pdf_path}') does not match target PDF path ('{pdf.path}'). Aborting."
            )
            raise ValueError("Manifest source PDF does not match the provided PDF object.")
        else:
            logger.warning(
                f"Manifest PDF source path ('{first_manifest_pdf_path}') differs from target PDF path ('{pdf.path}'), but filenames match. Proceeding cautiously."
            )

    pdf_pages_by_index: Dict[int, Page] = {
        cast(int, page.index): cast(Page, page) for page in getattr(pdf, "pages", [])
    }

    for page_data in tqdm(manifest_pages, desc="Importing OCR Data"):
        page_index = page_data.get("page_index")
        manifest_pdf_path = page_data.get("pdf_source")

        # Check consistency for every page? (Maybe overkill if pre-checked)
        if manifest_pdf_path != pdf.path and os.path.basename(
            manifest_pdf_path
        ) != os.path.basename(pdf.path):
            logger.warning(
                f"Skipping page index {page_index} due to PDF source mismatch ('{manifest_pdf_path}' vs '{pdf.path}')"
            )
            skipped_count += len(page_data.get("regions", []))  # Count all regions as skipped
            continue

        if page_index is None:
            logger.warning(
                f"Skipping page entry with missing 'page_index': {page_data.get('page_number')}"
            )
            skipped_count += len(page_data.get("regions", []))
            continue

        page = pdf_pages_by_index.get(page_index)
        if page is None:
            logger.warning(
                f"Could not find page with index {page_index} in the target PDF. Skipping."
            )
            skipped_count += len(page_data.get("regions", []))
            continue

        processed_pages += 1
        # We are adding elements, no need to fetch existing ones unless we want to prevent duplicates (not implemented here)

        regions_to_add: List[TextElement] = []
        for region_data in page_data.get("regions", []):
            # We import all regions, not just modified ones
            # if not region_data.get("modified", False):
            #     continue # Only process modified regions

            region_id = region_data.get("id", "unknown")
            manifest_bbox = region_data.get("bbox")
            # Use corrected_text as the primary text source for the new element
            text_to_import = region_data.get("corrected_text")
            # Fallback to ocr_text if corrected_text is missing (though unlikely from the SPA)
            if text_to_import is None:
                text_to_import = region_data.get("ocr_text")
            if text_to_import is None:
                text_to_import_str: Optional[str] = None
            elif isinstance(text_to_import, str):
                text_to_import_str = text_to_import
            else:
                text_to_import_str = str(text_to_import)

            resolution = region_data.get("resolution")  # Mandatory from export
            confidence = region_data.get("confidence")  # Optional

            if not all([manifest_bbox, text_to_import_str is not None, resolution]):
                logger.warning(
                    f"Skipping incomplete/invalid region data on page {page_index}, region id '{region_id}': Missing bbox, text, or resolution."
                )
                skipped_count += 1
                continue
            assert text_to_import_str is not None

            # Convert manifest bbox (image pixels) back to PDF coordinates (points @ 72 DPI)
            try:
                if not isinstance(manifest_bbox, (list, tuple)) or len(manifest_bbox) != 4:
                    raise ValueError("Bounding box must contain four coordinates.")
                scale_factor = 72.0 / float(resolution)
                pdf_x0 = float(manifest_bbox[0]) * scale_factor
                pdf_top = float(manifest_bbox[1]) * scale_factor
                pdf_x1 = float(manifest_bbox[2]) * scale_factor
                pdf_bottom = float(manifest_bbox[3]) * scale_factor
            except (ValueError, TypeError, IndexError, ZeroDivisionError):
                logger.warning(
                    f"Invalid bbox or resolution for region '{region_id}' on page {page_index}. Skipping."
                )
                skipped_count += 1
                continue

            # --- Create New Element ---
            try:
                element_payload: Dict[str, Any] = {
                    "text": text_to_import_str,
                    "x0": pdf_x0,
                    "top": pdf_top,
                    "x1": pdf_x1,
                    "bottom": pdf_bottom,
                    "bbox": (pdf_x0, pdf_top, pdf_x1, pdf_bottom),
                    "source": "manifest-import",
                }
                if confidence is not None:
                    element_payload["confidence"] = confidence

                new_element = TextElement(element_payload, page)
                original_ocr = region_data.get("ocr_text")
                if original_ocr and original_ocr != text_to_import_str:
                    new_element.metadata["original_ocr"] = original_ocr
                regions_to_add.append(new_element)
            except Exception as e:
                logger.error(
                    f"Error creating TextElement for region '{region_id}' on page {page_index}: {e}",
                    exc_info=True,
                )
                skipped_count += 1

        # --- Add Elements to Page ---
        # Add all created elements for this page in one go
        if regions_to_add:
            page.ensure_elements_loaded()
            for new_element in regions_to_add:
                try:
                    was_added = page.add_element(new_element, element_type="words")
                except Exception as add_error:
                    logger.error(
                        "Error adding imported OCR element to page %s: %s",
                        page.index,
                        add_error,
                        exc_info=True,
                    )
                    was_added = False

                if was_added:
                    imported_count += 1
                else:
                    skipped_count += 1

    logger.info(
        f"Import process finished. Imported: {imported_count}, Skipped: {skipped_count}. Processed {processed_pages} pages from manifest."
    )
    return {"imported": imported_count, "skipped": skipped_count}
