"""
Module for exporting original PDF pages without modification.
"""

import io
import logging
import os
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, List, Set, Union

from natural_pdf.exceptions import ExportError
from natural_pdf.utils.filesystem import atomic_output_path
from natural_pdf.utils.optional_imports import require

if TYPE_CHECKING:
    from natural_pdf.core.page import Page
    from natural_pdf.core.page_collection import PageCollection
    from natural_pdf.core.pdf import PDF

logger = logging.getLogger(__name__)


def create_original_pdf(
    source: Union["Page", "PageCollection", "PDF"], output_path: Union[str, Path]
):
    """
    Creates a new PDF file containing only the original, unmodified pages
    specified by the source object.

    Requires 'pikepdf'. Install with: pip install "natural-pdf[export]"

    Args:
        source: The Page, PageCollection, or PDF object indicating which pages to include.
        output_path: The path to save the resulting PDF file.

    Raises:
        ImportError: If 'pikepdf' is not installed.
        ValueError: If the source object is empty, pages are from different PDFs,
                    or the source PDF path cannot be determined.
        ExportError: If pikepdf fails to open the source or save the output, or
                     if a requested page index does not exist in the source PDF.
        pikepdf.PasswordError: If the source PDF is password-protected.
    """
    pikepdf = require("pikepdf")

    output_path_str = str(output_path)
    pages_to_extract: List["Page"] = []
    source_supported = False

    # Determine the list of pages from supported inputs
    if source is None:
        raise TypeError("Source cannot be None for create_original_pdf.")

    from collections.abc import Sequence

    from natural_pdf.core.page import Page

    pages_attr = getattr(source, "pages", None)
    if isinstance(source, Page):
        pages_to_extract = [source]
        source_supported = True
    elif pages_attr is not None:
        if hasattr(pages_attr, "pages"):
            # Handle PageCollection where .pages returns a sequence of Page objects
            pages_seq = getattr(pages_attr, "pages")
            pages_to_extract = list(pages_seq)
            source_supported = True
        elif isinstance(pages_attr, Sequence):
            pages_to_extract = list(pages_attr)
            source_supported = True
    elif isinstance(source, Sequence):
        filtered_pages = [page for page in source if isinstance(page, Page)]
        if filtered_pages:
            pages_to_extract = filtered_pages
            source_supported = True

    if not source_supported:
        raise TypeError(f"Unsupported source type for create_original_pdf: {type(source)}")

    if not pages_to_extract:
        raise ValueError("No valid pages found in the source object.")

    # Verify all pages come from the same PDF and get path
    first_page_pdf_path = None
    first_page_pdf_obj = None
    if hasattr(pages_to_extract[0], "pdf") and pages_to_extract[0].pdf:
        src_pdf = pages_to_extract[0].pdf
        first_page_pdf_path = getattr(src_pdf, "path", None)
        first_page_pdf_obj = src_pdf

    if not first_page_pdf_path:
        raise ValueError(
            "Cannot save original pages: Source PDF path not found for the first page."
        )

    page_indices_set: Set[int] = set()
    for page in pages_to_extract:
        page_pdf_path = getattr(getattr(page, "pdf", None), "path", None)
        if not page_pdf_path or page_pdf_path != first_page_pdf_path:
            raise ValueError(
                "Cannot save original pages: All pages must belong to the same source PDF document."
            )
        page_indices_set.add(page.index)  # 0-based index

    sorted_indices = sorted(list(page_indices_set))

    logger.info(
        f"Extracting original pages {sorted_indices} from '{first_page_pdf_path}' to '{output_path_str}'"
    )

    try:
        # Prefer opening via filesystem path when it exists locally
        if first_page_pdf_path and os.path.exists(first_page_pdf_path):
            source_handle = pikepdf.Pdf.open(first_page_pdf_path)
        else:
            # Fallback: attempt to open from in-memory bytes stored on PDF object
            if (
                first_page_pdf_obj is not None
                and hasattr(first_page_pdf_obj, "_original_bytes")
                and first_page_pdf_obj._original_bytes
            ):
                source_handle = pikepdf.Pdf.open(io.BytesIO(first_page_pdf_obj._original_bytes))
            else:
                # Attempt to download bytes directly if path looks like URL
                if isinstance(first_page_pdf_path, str) and first_page_pdf_path.startswith(
                    ("http://", "https://")
                ):
                    try:
                        request = first_page_pdf_obj._create_url_request(first_page_pdf_path)
                        ssl_context = first_page_pdf_obj._create_ssl_context()
                        with urllib.request.urlopen(
                            request, context=ssl_context, timeout=60
                        ) as resp:
                            data = resp.read()
                    except Exception as dl_err:
                        raise FileNotFoundError(
                            f"Source PDF bytes not available and download failed for {first_page_pdf_path}: {dl_err}"
                        ) from dl_err
                    # Parsing is intentionally outside the download handler:
                    # pikepdf.PasswordError must propagate for URL sources too.
                    source_handle = pikepdf.Pdf.open(io.BytesIO(data))
                else:
                    raise FileNotFoundError(
                        f"Source PDF bytes not available for {first_page_pdf_path}"
                    )

        with source_handle as source_pikepdf_doc:
            target_pikepdf_doc = pikepdf.Pdf.new()

            for page_index in sorted_indices:
                if 0 <= page_index < len(source_pikepdf_doc.pages):
                    # This correctly appends the pikepdf.Page object
                    target_pikepdf_doc.pages.append(source_pikepdf_doc.pages[page_index])
                else:
                    raise ExportError(
                        f"Page index {page_index} out of bounds for source PDF "
                        f"'{first_page_pdf_path}' ({len(source_pikepdf_doc.pages)} pages)."
                    )

            # Write to a temp path in the destination directory, then replace,
            # so a failure never leaves a truncated file at output_path.
            with atomic_output_path(output_path_str) as tmp_output_path:
                target_pikepdf_doc.save(str(tmp_output_path))
            logger.info(
                f"Successfully saved original pages PDF ({len(target_pikepdf_doc.pages)} pages) to: {output_path_str}"
            )

    except (ExportError, pikepdf.PasswordError):
        # PasswordError propagates as documented; ExportError already has context.
        raise
    except FileNotFoundError as e:
        raise ExportError(f"Failed to save original pages PDF: {e}") from e
    except Exception as e:
        logger.error(
            f"Failed to save original pages PDF to '{output_path_str}': {e}",
            exc_info=True,
        )
        raise ExportError(f"Failed to save original pages PDF: {e}") from e
