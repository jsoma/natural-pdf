"""
Module for exporting PDF content to various formats.
"""

import logging
import os
import tempfile
from typing import TYPE_CHECKING, List, Dict, Any, Tuple

# Lazy imports for optional dependencies
try:
    from PIL import Image
except ImportError:
    Image = None # type: ignore

try:
    import pikepdf
except ImportError:
    pikepdf = None # type: ignore

try:
    from ocrmypdf.hocrtransform import HocrTransform
except ImportError:
    HocrTransform = None # type: ignore

if TYPE_CHECKING:
    from natural_pdf.core.pdf import PDF
    from natural_pdf.core.page import Page


logger = logging.getLogger(__name__)

# --- Constants ---
HOCR_TEMPLATE_HEADER = '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
    "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
 <head>
  <title></title>
  <meta http-equiv="Content-Type" content="text/html;charset=utf-8" />
  <meta name='ocr-system' content='natural-pdf' />
  <meta name='ocr-capabilities' content='ocr_page ocr_carea ocr_par ocr_line ocrx_word'/>
 </head>
 <body>
'''

HOCR_TEMPLATE_PAGE = '''  <div class='ocr_page' id='page_{page_num}' title='image "{image_path}"; bbox 0 0 {width} {height}; ppageno {page_num}'>
'''

HOCR_TEMPLATE_WORD = '''   <span class='ocrx_word' id='word_{page_num}_{word_id}' title='bbox {x0} {y0} {x1} {y1}; x_wconf {confidence}'>{text}</span>
'''

HOCR_TEMPLATE_LINE_START = '''   <span class='ocr_line' id='line_{page_num}_{line_id}' title='bbox {x0} {y0} {x1} {y1}'>
'''
HOCR_TEMPLATE_LINE_END = '''   </span>
'''

HOCR_TEMPLATE_FOOTER = '''  </div>
 </body>
</html>
'''
# --- End Constants ---


def _generate_hocr_for_page(page: 'Page', image_width: int, image_height: int) -> str:
    """
    Generates an hOCR string for a given Page object based on its OCR elements.

    Args:
        page: The Page object containing OCR elements (TextElements).
        image_width: The width of the rendered image for coordinate scaling.
        image_height: The height of the rendered image for coordinate scaling.

    Returns:
        An hOCR XML string.

    Raises:
        ValueError: If the page has no OCR elements.
    """
    # Attempt to get OCR elements (words) using find_all with selector
    # Use find_all which returns an ElementCollection
    ocr_elements_collection = page.find_all('text[source=ocr]')
    ocr_elements = ocr_elements_collection.elements # Get the list of elements

    if not ocr_elements:
        logger.warning(f"Page {page.number} has no OCR elements (text[source=ocr]) to generate hOCR from.")
        # Return minimal valid hOCR for an empty page
        hocr_content = HOCR_TEMPLATE_HEADER
        hocr_content += HOCR_TEMPLATE_PAGE.format(page_num=page.index, image_path="", width=image_width, height=image_height)
        hocr_content += HOCR_TEMPLATE_FOOTER
        return hocr_content


    # --- TODO: Implement logic to group words into lines if necessary ---
    # For now, just output words directly. A more advanced implementation
    # might group words geometrically into lines first.
    # Example (simple, assuming elements are somewhat sorted):
    # lines = []
    # current_line = []
    # last_y = -1
    # for word in ocr_elements:
    #     if not current_line or abs(word.y0 - last_y) < threshold: # Simple Y-based grouping
    #         current_line.append(word)
    #         last_y = word.y0
    #     else:
    #         lines.append(current_line)
    #         current_line = [word]
    #         last_y = word.y0
    # if current_line:
    #     lines.append(current_line)
    # --- End Line Grouping Placeholder ---


    hocr_content = HOCR_TEMPLATE_HEADER
    hocr_content += HOCR_TEMPLATE_PAGE.format(page_num=page.index, image_path="", width=image_width, height=image_height) # image_path is often unused

    # Scale factors from PDF points (page dims) to image pixels (rendered image dims)
    # Note: Assumes OCR element coordinates are in PDF points (page.width/height)
    scale_x = image_width / page.width if page.width > 0 else 1
    scale_y = image_height / page.height if page.height > 0 else 1

    word_id_counter = 0
    for word in ocr_elements:
        # Scale coordinates to image dimensions
        img_x0 = int(word.x0 * scale_x)
        img_y0 = int(word.y0 * scale_y)
        img_x1 = int(word.x1 * scale_x)
        img_y1 = int(word.y1 * scale_y)

        # Ensure coordinates are within image bounds
        img_x0 = max(0, img_x0)
        img_y0 = max(0, img_y0)
        img_x1 = min(image_width, img_x1)
        img_y1 = min(image_height, img_y1)

        # Basic escaping for XML - might need more robust escaping
        text = word.text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        # Confidence (assuming it exists, default to 99 if not)
        confidence = getattr(word, 'confidence', 0.99) * 100 # hOCR often uses 0-100

        hocr_content += HOCR_TEMPLATE_WORD.format(
            page_num=page.index,
            word_id=word_id_counter,
            x0=img_x0,
            y0=img_y0,
            x1=img_x1,
            y1=img_y1,
            confidence=int(confidence),
            text=text
        )
        word_id_counter += 1
        hocr_content += "\n" # Add newline for readability


    hocr_content += HOCR_TEMPLATE_FOOTER
    return hocr_content


def create_searchable_pdf(pdf_object: 'PDF', output_path: str, dpi: int = 300):
    """
    Creates a searchable PDF from a natural_pdf.PDF object using OCR results.

    Relies on ocrmypdf for hOCR transformation. Requires optional dependencies.

    Args:
        pdf_object: The natural_pdf.PDF instance (OCR should have been run).
        output_path: The path to save the resulting searchable PDF.
        dpi: The resolution (dots per inch) for rendering page images and hOCR.
    """
    # _check_dependencies() # Removed check

    # --- Ensure dependencies are loaded (they should be if installed) ---
    if Image is None or pikepdf is None or HocrTransform is None:
        # This should ideally not happen if dependencies are in main install,
        # but serves as a safeguard during development or if install is broken.
        raise ImportError(
            "Required dependencies (Pillow, pikepdf, ocrmypdf) are missing. "
            "Please ensure natural-pdf is installed correctly with all dependencies."
        )
    # --- End Safeguard Check ---

    logger.info(f"Starting searchable PDF creation for '{pdf_object.source_path}' -> '{output_path}' at {dpi} DPI.")

    temp_pdf_pages: List[str] = []
    output_abs_path = os.path.abspath(output_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        logger.debug(f"Using temporary directory: {tmpdir}")

        for i, page in enumerate(pdf_object.pages):
            logger.debug(f"Processing page {page.number} (index {i})...")
            page_base_name = f"page_{i}"
            img_path = os.path.join(tmpdir, f"{page_base_name}.png") # Use PNG for potentially better quality
            hocr_path = os.path.join(tmpdir, f"{page_base_name}.hocr")
            pdf_page_path = os.path.join(tmpdir, f"{page_base_name}.pdf")

            try:
                # 1. Render page image at target DPI
                logger.debug(f"  Rendering page {i} to image ({dpi} DPI)...")
                # Use the Page's to_image method
                pil_image = page.to_image(resolution=dpi, include_highlights=False)
                pil_image.save(img_path, format='PNG')
                img_width, img_height = pil_image.size
                logger.debug(f"  Image saved to {img_path} ({img_width}x{img_height})")

                # 2. Generate hOCR
                logger.debug(f"  Generating hOCR...")
                hocr_content = _generate_hocr_for_page(page, img_width, img_height)
                with open(hocr_path, 'w', encoding='utf-8') as f:
                    f.write(hocr_content)
                logger.debug(f"  hOCR saved to {hocr_path}")


                # 3. Use HocrTransform to create searchable PDF page
                logger.debug(f"  Running HocrTransform...")
                hocr_transform = HocrTransform(hocr_filename=hocr_path, dpi=dpi)
                # Pass image_filename explicitly
                hocr_transform.to_pdf(out_filename=pdf_page_path, image_filename=img_path)
                temp_pdf_pages.append(pdf_page_path)
                logger.debug(f"  Temporary PDF page saved to {pdf_page_path}")

            except Exception as e:
                 logger.error(f"  Failed to process page {page.number}: {e}", exc_info=True)
                 # Decide whether to skip or raise error
                 # For now, let's skip and continue
                 logger.warning(f"  Skipping page {page.number} due to error.")
                 continue # Skip to the next page

        # 4. Merge temporary PDF pages
        if not temp_pdf_pages:
            logger.error("No pages were successfully processed. Cannot create output PDF.")
            raise RuntimeError("Failed to process any pages for searchable PDF creation.")

        logger.info(f"Merging {len(temp_pdf_pages)} processed pages into final PDF...")
        try:
            # Use pikepdf for merging
            output_pdf = pikepdf.Pdf.new()
            for temp_pdf_path in temp_pdf_pages:
                 with pikepdf.Pdf.open(temp_pdf_path) as src_page_pdf:
                      # Assuming each temp PDF has exactly one page
                      if len(src_page_pdf.pages) == 1:
                           output_pdf.pages.append(src_page_pdf.pages[0])
                      else:
                           logger.warning(f"Temporary PDF '{temp_pdf_path}' had unexpected number of pages ({len(src_page_pdf.pages)}). Skipping.")
            output_pdf.save(output_abs_path)
            logger.info(f"Successfully saved merged searchable PDF to: {output_abs_path}")
        except Exception as e:
            logger.error(f"Failed to merge temporary PDFs into '{output_abs_path}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to save final PDF: {e}") from e

    logger.debug("Temporary directory cleaned up.") 