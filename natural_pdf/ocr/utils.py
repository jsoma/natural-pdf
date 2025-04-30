import base64
import io
import logging
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional

from tqdm.auto import tqdm

from natural_pdf.elements.text import TextElement

if TYPE_CHECKING:
    from natural_pdf.elements.base import Element

# Import the global PDF render lock from dedicated locks module
from natural_pdf.utils.locks import pdf_render_lock

logger = logging.getLogger(__name__)


def _apply_ocr_correction_to_elements(
    elements: Iterable["Element"],
    correction_callback: Callable[[Any], Optional[str]],
    caller_info: str = "Utility",
) -> None:
    """
    Applies OCR correction callback to a list of elements in place,
    showing a progress bar.

    Iterates through elements, checks if source starts with 'ocr', calls
    the callback, and updates element.text if a new string is returned.

    Args:
        elements: An iterable of Element objects.
        correction_callback: A function accepting an element and returning
                             Optional[str] (new text or None).
        caller_info: String identifying the calling context for logs.
    """
    if not callable(correction_callback):
        # Raise error here so individual methods don't need to repeat the check
        raise TypeError("`correction_callback` must be a callable function.")

    if not elements:
        logger.warning(f"{caller_info}: No elements provided for correction.")
        return

    corrections_applied = 0
    elements_checked = 0

    # Prepare the iterable with tqdm
    element_iterable = tqdm(elements, desc=f"Correcting OCR ({caller_info})", unit="element")

    for element in element_iterable:
        # Check if the element is likely from OCR and has text attribute
        element_source = getattr(element, "source", None)
        if (
            isinstance(element_source, str)
            and element_source.startswith("ocr")
            and hasattr(element, "text")
        ):
            elements_checked += 1
            current_text = getattr(element, "text")  # Already checked hasattr

            new_text = correction_callback(element)

            if new_text is not None:
                if new_text != current_text:
                    element.text = new_text  # Update in place
                    corrections_applied += 1

    logger.info(
        f"{caller_info}: OCR correction finished. Checked: {elements_checked}, Applied: {corrections_applied}"
    )
    # No return value needed, modifies elements in place


def direct_ocr_llm(
    element,
    client,
    model="",
    resolution=150,
    prompt="OCR this image. Return only the exact text from the image. Include misspellings, punctuation, etc. If you cannot see any text, return an empty string.",
    padding=2,
) -> str:
    """Convenience method to directly OCR a region of the page."""

    if isinstance(element, TextElement):
        region = element.expand(left=padding, right=padding, top=padding, bottom=padding)
    else:
        region = element

    buffered = io.BytesIO()
    # Use the global PDF render lock when rendering images
    with pdf_render_lock:
        region_img = region.to_image(resolution=resolution, include_highlights=False)

    # Handle cases where image creation might fail (e.g., zero-dim region)
    if region_img is None:
        logger.warning(f"Could not generate image for region {region.bbox}, skipping OCR.")
        return ""  # Return empty string if image creation failed

    region_img.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert OCR engineer. You will be given an image of a region of a page. You will return the exact text from the image.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            },
        ],
    )

    corrected = response.choices[0].message.content.strip()
    logger.debug(f"Corrected {region.extract_text()} to {corrected}")

    return corrected
