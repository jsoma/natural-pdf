"""
OCR debug utilities for natural-pdf.
"""

import base64
import importlib.resources
import importlib.util
import io
import json
import os
import webbrowser
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image

# Assuming Page type hint is available or define a placeholder
try:
    from natural_pdf.core.page import Page
except ImportError:
    Page = Any  # Placeholder


def _get_page_image_base64(page: Page) -> str:
    """Generate a base64 encoded image of the page."""
    # Create a clean image of the page without highlights for the base background
    # Use a fixed scale consistent with the HTML/JS rendering logic
    # Use render() for clean image without highlights
    img = page.render(resolution=144)
    if img is None:
        raise ValueError(f"Failed to render image for page {page.number}")

    # Convert to base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
