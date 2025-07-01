"""
Module for exporting original PDF pages without modification.
"""

import io
import logging
import os
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, List, Set, Union

# Lazy import for optional dependency
try:
    import pikepdf
except ImportError:
    pikepdf = None

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

    Requires 'pikepdf'. Install with: pip install "natural-pdf[ocr-export]"

    Args:
        source: The Page, PageCollection, or PDF object indicating which pages to include.
        output_path: The path to save the resulting PDF file.

    Raises:
        ImportError: If 'pikepdf' is not installed.
        ValueError: If the source object is empty, pages are from different PDFs,
                    or the source PDF path cannot be determined.
        RuntimeError: If pikepdf fails to open the source or save the output.
        pikepdf.PasswordError: If the source PDF is password-protected.
    """
    if pikepdf is None:
        raise ImportError(
            "Saving original PDF pages requires 'pikepdf'. "
            'Install with: pip install "natural-pdf[ocr-export]"'
        )

    output_path_str = str(output_path)
    pages_to_extract: List["Page"] = []

    # Determine the list of pages and the source PDF path
    if hasattr(source, "pages") and isinstance(source.pages, list):  # PDF or PageCollection
        if not source.pages:
            raise ValueError("Cannot save an empty collection/PDF.")
        pages_to_extract = source.pages
    elif hasattr(source, "page") and hasattr(source, "number"):  # Single Page object
        # Check if it's a natural_pdf.core.page.Page or similar duck-typed object
        if hasattr(source, "pdf") and source.pdf and hasattr(source.pdf, "path"):
            pages_to_extract = [source]
        else:
            raise ValueError("Input Page object does not have a valid PDF reference with a path.")
    else:
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
                        with urllib.request.urlopen(first_page_pdf_path) as resp:
                            data = resp.read()
                        source_handle = pikepdf.Pdf.open(io.BytesIO(data))
                    except Exception as dl_err:
                        raise FileNotFoundError(
                            f"Source PDF bytes not available and download failed for {first_page_pdf_path}: {dl_err}"
                        )
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
                    logger.warning(
                        f"Page index {page_index} out of bounds for source PDF '{first_page_pdf_path}'. Skipping."
                    )

            if not target_pikepdf_doc.pages:
                raise RuntimeError(f"No valid pages found to save from source PDF.")

            target_pikepdf_doc.save(output_path_str)
            logger.info(
                f"Successfully saved original pages PDF ({len(target_pikepdf_doc.pages)} pages) to: {output_path_str}"
            )

    except FileNotFoundError as e:
        logger.error(str(e))
        raise RuntimeError(f"Failed to save original pages PDF: {e}")
    except pikepdf.PasswordError:
        logger.error(f"Failed to open password-protected source PDF: {first_page_pdf_path}")
        raise RuntimeError(
            f"Source PDF '{first_page_pdf_path}' is password-protected."
        ) from None  # Raise specific error without chaining the generic Exception
    except Exception as e:
        logger.error(
            f"Failed to save original pages PDF to '{output_path_str}': {e}",
            exc_info=True,
        )
        # Re-raise as RuntimeError for consistent API error handling
        raise RuntimeError(f"Failed to save original pages PDF: {e}") from e
