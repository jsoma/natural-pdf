"""
Module for exporting PDF content to various formats.
"""

import logging
import os
import tempfile
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union
from xml.etree.ElementTree import Element as ETElement
from xml.etree.ElementTree import SubElement

# Lazy imports for optional dependencies
try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

try:
    import pikepdf
except ImportError:
    pikepdf = None  # type: ignore

try:
    from natural_pdf.exporters.hocr import HocrTransform
except ImportError:
    HocrTransform = None  # type: ignore

if TYPE_CHECKING:
    from natural_pdf.core.page import Page
    from natural_pdf.core.page_collection import PageCollection
    from natural_pdf.core.pdf import PDF


logger = logging.getLogger(__name__)

# --- Constants ---
HOCR_TEMPLATE_HEADER = """<?xml version="1.0" encoding="UTF-8"?>
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
"""

HOCR_TEMPLATE_PAGE = """  <div class='ocr_page' id='page_{page_num}' title='image "{image_path}"; bbox 0 0 {width} {height}; ppageno {page_num}'>
"""

HOCR_TEMPLATE_WORD = """   <span class='ocrx_word' id='word_{page_num}_{word_id}' title='bbox {x0} {y0} {x1} {y1}; x_wconf {confidence}'>{text}</span>
"""

HOCR_TEMPLATE_LINE_START = """   <span class='ocr_line' id='line_{page_num}_{line_id}' title='bbox {x0} {y0} {x1} {y1}; baseline 0 0; x_size 0; x_descenders 0; x_ascenders 0'>
"""
HOCR_TEMPLATE_LINE_END = """   </span>
"""

HOCR_TEMPLATE_FOOTER = """  </div>
 </body>
</html>
"""
# --- End Constants ---


def _generate_hocr_for_page(page: "Page", image_width: int, image_height: int) -> str:
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
    ocr_elements_collection = page.find_all("text[source=ocr]")
    ocr_elements = ocr_elements_collection.elements  # Get the list of elements

    if not ocr_elements:
        logger.warning(
            f"Page {page.number} has no OCR elements (text[source=ocr]) to generate hOCR from."
        )
        # Return minimal valid hOCR for an empty page
        hocr_content = HOCR_TEMPLATE_HEADER
        hocr_content += HOCR_TEMPLATE_PAGE.format(
            page_num=page.index, image_path="", width=image_width, height=image_height
        )
        hocr_content += HOCR_TEMPLATE_FOOTER
        return hocr_content

    # --- Start Line Grouping Logic ---
    logger.debug(f"Page {page.index}: Grouping {len(ocr_elements)} words into lines.")
    ocr_elements.sort(key=lambda el: (el.bbox[1], el.bbox[0]))
    lines = []
    current_line = []
    if ocr_elements:
        current_line.append(ocr_elements[0])
        for i in range(1, len(ocr_elements)):
            current_word = ocr_elements[i]
            last_word = current_line[-1]
            last_word_y0, last_word_y1 = last_word.bbox[1], last_word.bbox[3]
            current_word_y0, current_word_y1 = current_word.bbox[1], current_word.bbox[3]
            last_word_center_y = (last_word_y0 + last_word_y1) / 2
            current_word_center_y = (current_word_y0 + current_word_y1) / 2
            last_word_height = last_word_y1 - last_word_y0
            current_word_height = current_word_y1 - current_word_y0
            avg_height = (last_word_height + current_word_height) / 2
            if avg_height <= 0:
                avg_height = 1
            tolerance_factor = 0.7
            threshold = avg_height * tolerance_factor
            delta_y = abs(current_word_center_y - last_word_center_y)
            # if delta_y < threshold:
            #     current_line.append(current_word)
            # else:
            lines.append(current_line)
            current_line = [current_word]
        if current_line:
            lines.append(current_line)
    logger.debug(f"Page {page.index}: Grouped into {len(lines)} lines.")
    # --- End Line Grouping Logic ---

    # --- Start ElementTree hOCR Generation ---
    scale_x = image_width / page.width if page.width > 0 else 1
    scale_y = image_height / page.height if page.height > 0 else 1

    # Create root element
    page_hocr = ETElement(
        "html", attrib={"xmlns": "http://www.w3.org/1999/xhtml", "xml:lang": "en"}
    )

    # Head
    head = SubElement(page_hocr, "head")
    SubElement(head, "title").text = ""
    SubElement(
        head, "meta", attrib={"http-equiv": "Content-Type", "content": "text/html;charset=utf-8"}
    )
    SubElement(head, "meta", attrib={"name": "ocr-system", "content": "natural-pdf"})
    SubElement(
        head,
        "meta",
        attrib={
            "name": "ocr-capabilities",
            "content": "ocr_page ocr_carea ocr_par ocr_line ocrx_word",
        },
    )

    # Body and Page
    body = SubElement(page_hocr, "body")
    page_div = SubElement(
        body,
        "div",
        attrib={
            "class": "ocr_page",
            "id": f"page_{page.index}",
            "title": f"image; bbox 0 0 {image_width} {image_height}; ppageno {page.index}",
        },
    )

    # Calculate overall bbox for carea/par (image coords)
    min_area_x0, min_area_y0 = image_width, image_height
    max_area_x1, max_area_y1 = 0, 0
    if lines:
        for line_words in lines:
            for word in line_words:
                (x0, y0, x1, y1) = word.bbox
                img_x0 = int(x0 * scale_x)
                img_y0 = int(y0 * scale_y)
                img_x1 = int(x1 * scale_x)
                img_y1 = int(y1 * scale_y)
                min_area_x0 = min(min_area_x0, img_x0)
                min_area_y0 = min(min_area_y0, img_y0)
                max_area_x1 = max(max_area_x1, img_x1)
                max_area_y1 = max(max_area_y1, img_y1)
        area_img_x0, area_img_y0 = max(0, min_area_x0), max(0, min_area_y0)
        area_img_x1, area_img_y1 = min(image_width, max_area_x1), min(image_height, max_area_y1)
        if area_img_x0 >= area_img_x1 or area_img_y0 >= area_img_y1:
            area_img_x0, area_img_y0, area_img_x1, area_img_y1 = 0, 0, image_width, image_height
    else:
        area_img_x0, area_img_y0, area_img_x1, area_img_y1 = 0, 0, image_width, image_height

    # Add Carea and Par wrappers (assuming one block/paragraph per page for simplicity)
    block_div = SubElement(
        page_div,  # Attach to page_div now
        "div",
        attrib={
            "class": "ocr_carea",
            "id": "block_0_1",  # Simple ID
            "title": f"bbox {area_img_x0} {area_img_y0} {area_img_x1} {area_img_y1}",
        },
    )
    par_div = SubElement(
        block_div,
        "p",
        attrib={
            "class": "ocr_par",
            "id": "par_0_1",  # Simple ID
            "title": f"bbox {area_img_x0} {area_img_y0} {area_img_x1} {area_img_y1}",
        },
    )

    # Loop through lines and words
    word_id_counter = 0
    line_id_counter = 0
    for current_line_words in lines:
        if not current_line_words:
            continue

        # Sort words in line by x0
        current_line_words.sort(key=lambda el: el.bbox[0])

        # Calculate line bbox (image coords)
        min_line_x0, min_line_y0 = image_width, image_height
        max_line_x1, max_line_y1 = 0, 0
        for word in current_line_words:
            (x0, y0, x1, y1) = word.bbox
            img_x0, img_y0 = int(x0 * scale_x), int(y0 * scale_y)
            img_x1, img_y1 = int(x1 * scale_x), int(y1 * scale_y)
            min_line_x0, min_line_y0 = min(min_line_x0, img_x0), min(min_line_y0, img_y0)
            max_line_x1, max_line_y1 = max(max_line_x1, img_x1), max(max_line_y1, img_y1)

        line_img_x0, line_img_y0 = max(0, min_line_x0), max(0, min_line_y0)
        line_img_x1, line_img_y1 = min(image_width, max_line_x1), min(image_height, max_line_y1)
        if line_img_x0 >= line_img_x1 or line_img_y0 >= line_img_y1:
            line_img_x0, line_img_y0, line_img_x1, line_img_y1 = 0, 0, 1, 1

        # Create ocr_line span
        line_span = SubElement(
            par_div,  # Attach line to paragraph
            "span",
            attrib={
                "class": "ocr_line",
                "id": f"line_{page.index}_{line_id_counter}",
                "title": f"bbox {line_img_x0} {line_img_y0} {line_img_x1} {line_img_y1}; baseline 0 0; x_size 0; x_descenders 0; x_ascenders 0",
            },
        )

        # Add words to line
        for word in current_line_words:
            (x0, y0, x1, y1) = word.bbox
            img_x0, img_y0 = int(x0 * scale_x), int(y0 * scale_y)
            img_x1, img_y1 = int(x1 * scale_x), int(y1 * scale_y)

            img_x0, img_y0 = max(0, img_x0), max(0, img_y0)
            img_x1, img_y1 = min(image_width, img_x1), min(image_height, img_y1)
            if img_x1 <= img_x0:
                img_x1 = img_x0 + 1
            if img_y1 <= img_y0:
                img_y1 = img_y0 + 1

            # --- Strip whitespace and check if word is empty --- #
            text = word.text.strip().replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            if not text:
                continue  # Skip adding this word if it becomes empty after stripping
            # --- End strip ---
            confidence = getattr(word, "confidence", 1.00)

            word_span = SubElement(
                line_span,  # Attach word to line
                "span",
                attrib={
                    "class": "ocrx_word",
                    "id": f"word_{page.index}_{word_id_counter}",
                    "title": f"bbox {img_x0} {img_y0} {img_x1} {img_y1}; x_wconf {confidence}",
                },
            )
            word_span.text = text
            word_id_counter += 1
        line_id_counter += 1

    # Convert ElementTree to string
    # xml_declaration = '<?xml version="1.0" encoding="UTF-8"?>\n' # No longer needed
    # doctype_declaration = '''<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
    # "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">\n''' # No longer needed
    # ET.indent(page_hocr) # Optional: for pretty printing, requires Python 3.9+
    # Need bytes for writing, then decode for HocrTransform if it needs str
    # Let's stick to unicode string output for now, as the file write expects it.
    hocr_content = ET.tostring(
        page_hocr, encoding="unicode", method="xml"
    )  # Revert back to method='xml'
    # hocr_content = xml_declaration + doctype_declaration + hocr_string_content # Removed string addition
    # --- End ElementTree hOCR Generation ---

    # --- Add code to save hOCR output for inspection ---
    try:
        hocr_output_path = "natural_pdf_hocr_output.hocr"
        with open(hocr_output_path, "w", encoding="utf-8") as f_out:
            f_out.write(hocr_content)
        logger.info(f"Saved hOCR content for page {page.index} to: {hocr_output_path}")
    except Exception as e:
        logger.error(f"Failed to save hOCR output to file: {e}")
    # --- End save hOCR ---

    return hocr_content


def create_searchable_pdf(
    source: Union["Page", "PageCollection", "PDF"], output_path: str, dpi: int = 300
):
    """
    Creates a searchable PDF from a natural_pdf.PDF object using OCR results.

    Relies on pikepdf for saving the PDF.

    Args:
        source: The natural_pdf.PDF, PageCollection, or Page object
        output_path: The path to save the resulting searchable PDF.
        dpi: The resolution (dots per inch) for rendering page images and hOCR.
    """

    # duck type to see if source has .pages, to populate pages =
    if hasattr(source, "pages"):
        pages = source.pages
    else:
        pages = [source]

    logger.info(f"Starting searchable PDF creation '{output_path}' at {dpi} DPI.")

    temp_pdf_pages: List[str] = []
    output_abs_path = os.path.abspath(output_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        logger.debug(f"Using temporary directory: {tmpdir}")

        for i, page in enumerate(pages):
            logger.debug(f"Processing page {i+1} of {len(pages)}...")
            page_base_name = f"page_{i}"
            img_path = os.path.join(
                tmpdir, f"{page_base_name}.png"
            )  # Use PNG for potentially better quality
            hocr_path = os.path.join(tmpdir, f"{page_base_name}.hocr")
            pdf_page_path = os.path.join(tmpdir, f"{page_base_name}.pdf")

            try:
                # 1. Render page image at target DPI
                logger.debug(f"  Rendering page {i} to image ({dpi} DPI)...")
                # Use the Page's to_image method
                # Use render() for clean image without highlights
                pil_image = page.render(resolution=dpi)
                pil_image.save(img_path, format="PNG")
                img_width, img_height = pil_image.size
                logger.debug(f"  Image saved to {img_path} ({img_width}x{img_height})")

                # 2. Generate hOCR
                logger.debug(f"  Generating hOCR...")
                hocr_content = _generate_hocr_for_page(page, img_width, img_height)
                with open(hocr_path, "w", encoding="utf-8") as f:
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
                continue  # Skip to the next page

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
                        logger.warning(
                            f"Temporary PDF '{temp_pdf_path}' had unexpected number of pages ({len(src_page_pdf.pages)}). Skipping."
                        )
            output_pdf.save(output_abs_path)
            logger.info(f"Successfully saved merged searchable PDF to: {output_abs_path}")
        except Exception as e:
            logger.error(
                f"Failed to merge temporary PDFs into '{output_abs_path}': {e}", exc_info=True
            )
            raise RuntimeError(f"Failed to save final PDF: {e}") from e

    logger.debug("Temporary directory cleaned up.")
