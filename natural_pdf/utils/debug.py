"""
OCR debug utilities for natural-pdf.
"""
import base64
import io
import json
import os
import importlib.util
import importlib.resources
import webbrowser
from typing import Dict, List, Any, Optional, Union, Tuple

from PIL import Image

# Assuming Page type hint is available or define a placeholder
try:
    from natural_pdf.core.page import Page
except ImportError:
    Page = Any # Placeholder

# Function to load the OCR debug HTML template and CSS
def _load_ocr_debug_assets():
    """
    Load the OCR debug HTML template and CSS stylesheet.

    Returns:
        Tuple[str, str]: The HTML template string and CSS string.

    Raises:
        FileNotFoundError: If either template or CSS file cannot be found.
    """
    html_template = ""
    css_content = ""
    package_dir = 'natural_pdf.templates'
    html_filename = 'ocr_debug.html'
    css_filename = 'ocr_debug.css'

    try:
        # Try using importlib.resources (Python 3.7+)
        try:
            # For Python 3.9+
            html_path = importlib.resources.files(package_dir).joinpath(html_filename)
            css_path = importlib.resources.files(package_dir).joinpath(css_filename)
            with html_path.open('r', encoding='utf-8') as f_html, \
                 css_path.open('r', encoding='utf-8') as f_css:
                html_template = f_html.read()
                css_content = f_css.read()
        except (AttributeError, TypeError):
            # Fallback for Python 3.7-3.8
            html_template = importlib.resources.read_text(package_dir, html_filename, encoding='utf-8')
            css_content = importlib.resources.read_text(package_dir, css_filename, encoding='utf-8')

    except (ImportError, FileNotFoundError, ModuleNotFoundError) as e1:
        # Fallback for direct file access (development or non-package environments)
        import inspect
        try:
            # Get the directory of the current file (debug.py)
            current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
            # Go up one level (to utils) and then into templates
            templates_dir = os.path.join(os.path.dirname(current_dir), 'templates')
            html_path = os.path.join(templates_dir, html_filename)
            css_path = os.path.join(templates_dir, css_filename)

            if os.path.exists(html_path) and os.path.exists(css_path):
                with open(html_path, 'r', encoding='utf-8') as f_html, \
                     open(css_path, 'r', encoding='utf-8') as f_css:
                    html_template = f_html.read()
                    css_content = f_css.read()
            else:
                missing = []
                if not os.path.exists(html_path): missing.append(html_path)
                if not os.path.exists(css_path): missing.append(css_path)
                raise FileNotFoundError(f"OCR debug assets not found (fallback): {', '.join(missing)}")
        except Exception as e2:
            # Log both errors if fallback also fails
            print(f"Error loading OCR debug assets: Primary error ({type(e1).__name__}): {e1}, Fallback error ({type(e2).__name__}): {e2}")
            raise FileNotFoundError("Could not load OCR debug template or CSS file.") from e2

    if not html_template or not css_content:
         raise FileNotFoundError("OCR debug template or CSS content is empty after loading.")

    return html_template, css_content

def debug_ocr_to_html(pages: List[Page], output_path: Optional[str] = None) -> str:
    """
    Generate an interactive HTML debug report for OCR results from a list of pages.

    Args:
        pages: List of natural_pdf Page objects.
        output_path: Path to save the HTML report. If None, returns HTML string.

    Returns:
        Path to the generated HTML file if output_path is provided, otherwise the HTML string.
    """
    # Prepare the data structure
    pages_data = {"pages": []}

    # Process each page
    for i, page in enumerate(pages):
        # Ensure the page has the necessary methods/attributes
        if not hasattr(page, 'find_all') or not hasattr(page, 'to_image'):
            print(f"Warning: Skipping page {getattr(page, 'number', i)} - does not appear to be a valid natural_pdf Page object.")
            continue

        # Extract OCR elements (assuming they exist or have been generated)
        try:
            # Find elements marked with source='ocr'
            # We rely on OCR having been run beforehand and elements being available via page.words or page.get_elements()
            # If no elements are marked with 'ocr', this might return empty.
            ocr_elements = page.find_all('text[source=ocr]', apply_exclusions=False).elements
            # Alternative: Directly check word elements if selector is slow/unreliable
            # ocr_elements = [w for w in page.words if getattr(w, 'source', None) == 'ocr']

        except Exception as e:
            print(f"Error extracting OCR elements from page {getattr(page, 'number', i)}: {e}")
            continue

        # Skip if no OCR elements found
        if not ocr_elements:
            print(f"No OCR elements with source='ocr' found on page {getattr(page, 'number', i)}. Skipping.")
            continue

        # Get page image as base64
        try:
            img_data = _get_page_image_base64(page)
        except Exception as e:
            print(f"Error generating image for page {getattr(page, 'number', i)}: {e}")
            continue # Skip page if image fails

        # Create page data
        page_data = {
            "page_number": page.number,
            "image": img_data,
            "regions": []
        }

        # Process OCR elements
        for j, elem in enumerate(ocr_elements):
            # Ensure element has necessary attributes
            if not all(hasattr(elem, attr) for attr in ['x0', 'top', 'x1', 'bottom', 'text']):
                print(f"Warning: Skipping OCR element {j} on page {page.number} due to missing attributes.")
                continue

            region = {
                "id": f"region_{page.index}_{j}", # Unique ID across pages
                "bbox": [elem.x0, elem.top, elem.x1, elem.bottom],
                "ocr_text": elem.text,
                # Corrected text might exist if user modified it in a previous session or if loaded from source
                "corrected_text": getattr(elem, 'corrected_text', elem.text),
                "confidence": getattr(elem, 'confidence', 0.0),
                "modified": getattr(elem, 'corrected_text', elem.text) != elem.text # Initial modified state
            }
            page_data["regions"].append(region)

        if page_data["regions"]: # Only add page if it has valid regions
            pages_data["pages"].append(page_data)
        else:
            print(f"No valid OCR regions processed for page {page.number}.")

    # Check if any pages were processed
    if not pages_data["pages"]:
        print("Warning: No pages with valid OCR data found to generate the report.")
        # Return an empty string or minimal HTML indicating no data?
        return "<html><body><h1>OCR Debug Report</h1><p>No pages with OCR data found.</p></body></html>"

    # Get the HTML template and CSS
    try:
        template, css_content = _load_ocr_debug_assets()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return "<html><body><h1>Error</h1><p>Could not load debug template files.</p></body></html>"

    # Prepare JSON data for injection, escaping potentially problematic characters for JS
    # Using json.dumps ensures proper escaping within the string literal
    pages_data_json = json.dumps(pages_data)

    # Inject CSS and JSON data using simple string replacement to avoid issues with JS braces
    html = template.replace("/* {css_content} */", css_content)
    html = html.replace("{pages_data_json}", pages_data_json)

    # Save to file if output path provided
    if output_path:
        try:
            # Get the directory part of the path
            dir_name = os.path.dirname(output_path)
            # Only create directories if dir_name is not empty (i.e., path includes a directory)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            # Now open and write the file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"OCR Debug Report saved to: {output_path}")
            # Try to open the file in browser
            try:
                webbrowser.open('file://' + os.path.abspath(output_path))
            except Exception as wb_err:
                print(f"(Could not automatically open report in browser: {wb_err})" )
            return output_path
        except Exception as e:
            print(f"Error saving HTML report to {output_path}: {e}")
            return html # Return HTML string on save error

    # Return as string otherwise
    return html


def _get_page_image_base64(page: Page) -> str:
    """Generate a base64 encoded image of the page."""
    # Create a clean image of the page without highlights for the base background
    # Use a fixed scale consistent with the HTML/JS rendering logic
    img = page.to_image(scale=2.0, include_highlights=False)
    if img is None:
        raise ValueError(f"Failed to render image for page {page.number}")

    # Convert to base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}" 