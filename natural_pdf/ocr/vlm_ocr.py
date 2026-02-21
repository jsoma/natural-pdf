"""VLM-based OCR with grounding (bounding-box) support.

Parses grounding model output into the standard OCR result format
(list of dicts with ``bbox``, ``text``, ``confidence`` keys) that
the existing OCR pipeline can consume.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from natural_pdf.utils.locks import pdf_render_lock

logger = logging.getLogger(__name__)


def parse_grounding_response(raw: str) -> List[Dict[str, Any]]:
    """Parse a VLM grounding response into OCR result dicts.

    Expects a JSON array of objects with ``bbox``, ``text``, and optionally
    ``confidence`` keys.  Handles common model quirks like markdown fences.

    Args:
        raw: The raw text response from the VLM.

    Returns:
        List of dicts, each with ``bbox`` (4-element list), ``text`` (str),
        and ``confidence`` (float, default 0.5).
    """
    text = raw.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    # Try to find a JSON array in the response
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        text = match.group(0)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse VLM grounding response as JSON: %s", text[:200])
        return []

    if not isinstance(data, list):
        logger.warning("VLM grounding response is not a JSON array.")
        return []

    results = []
    for item in data:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox")
        item_text = item.get("text", "")
        confidence = item.get("confidence", 0.5)

        if bbox is None or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        if not item_text:
            continue

        try:
            bbox = [float(v) for v in bbox]
            confidence = float(confidence)
        except (TypeError, ValueError):
            continue

        results.append(
            {
                "bbox": bbox,
                "text": str(item_text),
                "confidence": max(0.0, min(1.0, confidence)),
            }
        )

    return results


def scale_ocr_results(
    results: List[Dict[str, Any]],
    *,
    image_width: float,
    image_height: float,
    page_width: float,
    page_height: float,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> List[Dict[str, Any]]:
    """Scale bounding boxes from image coordinates to PDF coordinates.

    Args:
        results: OCR result dicts with ``bbox`` in image pixel coordinates.
        image_width: Width of the rendered image in pixels.
        image_height: Height of the rendered image in pixels.
        page_width: Width of the PDF page/region in points.
        page_height: Height of the PDF page/region in points.
        offset_x: X offset for region crops.
        offset_y: Y offset for region crops.

    Returns:
        New list of result dicts with scaled ``bbox`` values.
    """
    if not image_width or not image_height:
        return results

    scale_x = page_width / image_width
    scale_y = page_height / image_height

    scaled = []
    for r in results:
        bbox = r["bbox"]
        x0 = bbox[0] * scale_x + offset_x
        y0 = bbox[1] * scale_y + offset_y
        x1 = bbox[2] * scale_x + offset_x
        y1 = bbox[3] * scale_y + offset_y

        # Normalize inverted coordinates
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0

        # Clamp to page bounds
        x0 = max(0.0, min(x0, page_width))
        y0 = max(0.0, min(y0, page_height))
        x1 = max(0.0, min(x1, page_width))
        y1 = max(0.0, min(y1, page_height))

        # Skip degenerate boxes
        if x1 - x0 < 0.1 or y1 - y0 < 0.1:
            continue

        scaled.append(
            {
                **r,
                "bbox": [x0, y0, x1, y1],
            }
        )
    return scaled


def run_vlm_ocr(
    host: Any,
    *,
    model: Optional[str] = None,
    client: Optional[Any] = None,
    resolution: int = 144,
    render_kwargs: Optional[Dict[str, Any]] = None,
    max_new_tokens: int = 4096,
    prompt: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Tuple[int, int]]:
    """Run VLM-based OCR on a page/region and return scaled results.

    Args:
        host: Page or Region with a ``render()`` method.
        model: HuggingFace model ID or remote model name.
        client: OpenAI-compatible client.
        resolution: DPI for rendering.
        render_kwargs: Extra kwargs for ``host.render()``.
        max_new_tokens: Max tokens for VLM generation.
        prompt: Custom prompt (default uses the grounding prompt).

    Returns:
        Tuple of (ocr_results, (image_width, image_height)).
        ``ocr_results`` are in *image* pixel coordinates (not yet scaled).
    """
    from natural_pdf.core.vlm_client import generate
    from natural_pdf.core.vlm_prompts import build_ocr_prompt

    effective_prompt = prompt or build_ocr_prompt(grounding=True)

    # Render page/region to image
    render_fn = getattr(host, "render", None)
    if not callable(render_fn):
        raise AttributeError("Host object does not support render() for VLM OCR.")

    kwargs = dict(render_kwargs or {})
    with pdf_render_lock:
        image = render_fn(resolution=resolution, **kwargs)

    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected render() to return a PIL Image, got {type(image).__name__}")

    raw_response = generate(
        image,
        effective_prompt,
        model=model,
        client=client,
        max_new_tokens=max_new_tokens,
    )

    results = parse_grounding_response(raw_response)
    return results, image.size
