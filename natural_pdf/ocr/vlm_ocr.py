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


_FAMILY_DEFAULT_CONFIDENCE = {
    "qwen_vl": 0.75,
    "gemini": 0.75,
    "openai": 0.5,
    "gutenocr": 0.8,
    "generic": 0.5,
}


def normalize_qwen_coordinates(
    results: List[Dict[str, Any]],
    image_width: float,
    image_height: float,
) -> List[Dict[str, Any]]:
    """Convert Qwen-VL 0-1000 normalized coordinates to image pixel coordinates.

    Args:
        results: OCR result dicts with ``bbox`` in 0-1000 normalized range.
        image_width: Width of the rendered image in pixels.
        image_height: Height of the rendered image in pixels.

    Returns:
        New list with ``bbox`` values scaled to image pixel coordinates.
    """
    scaled = []
    for r in results:
        bbox = r["bbox"]
        x0 = max(0.0, min(float(bbox[0]), 1000.0)) / 1000.0 * image_width
        y0 = max(0.0, min(float(bbox[1]), 1000.0)) / 1000.0 * image_height
        x1 = max(0.0, min(float(bbox[2]), 1000.0)) / 1000.0 * image_width
        y1 = max(0.0, min(float(bbox[3]), 1000.0)) / 1000.0 * image_height
        scaled.append({**r, "bbox": [x0, y0, x1, y1]})
    return scaled


def normalize_gemini_coordinates(
    results: List[Dict[str, Any]],
    image_width: float,
    image_height: float,
) -> List[Dict[str, Any]]:
    """Convert Gemini box_2d coordinates to image pixel coordinates.

    Gemini uses ``[y_min, x_min, y_max, x_max]`` in 0-1000 normalized range.
    This swaps the axes to ``[x_min, y_min, x_max, y_max]`` and scales to
    image pixel coordinates.

    Args:
        results: OCR result dicts with ``bbox`` in Gemini's ``[y, x, y, x]`` order.
        image_width: Width of the rendered image in pixels.
        image_height: Height of the rendered image in pixels.

    Returns:
        New list with ``bbox`` values as ``[x0, y0, x1, y1]`` in pixel coordinates.
    """
    scaled = []
    for r in results:
        bbox = r["bbox"]
        # Gemini order: [y_min, x_min, y_max, x_max] → swap to [x_min, y_min, x_max, y_max]
        y0_norm = max(0.0, min(float(bbox[0]), 1000.0))
        x0_norm = max(0.0, min(float(bbox[1]), 1000.0))
        y1_norm = max(0.0, min(float(bbox[2]), 1000.0))
        x1_norm = max(0.0, min(float(bbox[3]), 1000.0))

        x0 = x0_norm / 1000.0 * image_width
        y0 = y0_norm / 1000.0 * image_height
        x1 = x1_norm / 1000.0 * image_width
        y1 = y1_norm / 1000.0 * image_height
        scaled.append({**r, "bbox": [x0, y0, x1, y1]})
    return scaled


def _extract_json_array(text: str) -> Optional[list]:
    """Extract the first JSON array from *text* using ``json.JSONDecoder``.

    Falls back to a regex search if the decoder approach fails.
    """
    # Try raw_decode starting from the first '['
    idx = text.find("[")
    if idx != -1:
        decoder = json.JSONDecoder()
        try:
            obj, _ = decoder.raw_decode(text, idx)
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            pass

    # Fallback: greedy regex (handles some edge cases)
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            pass

    return None


def _extract_item(item: dict, family: str) -> Optional[Dict[str, Any]]:
    """Extract bbox/text/confidence from a single response item.

    Uses adaptive key detection: tries the expected schema for the given
    *family* first, then falls back to the alternative schema.
    """
    default_conf = _FAMILY_DEFAULT_CONFIDENCE.get(family, 0.5)

    # Define key schemas in priority order based on family
    if family in ("qwen_vl", "gemini"):
        schemas = [
            ("box_2d", "label"),
            ("bbox_2d", "label"),
            ("bbox", "text"),
        ]
    else:
        schemas = [
            ("bbox", "text"),
            ("bbox_2d", "label"),
            ("box_2d", "label"),
        ]

    bbox = None
    item_text = None
    for bbox_key, text_key in schemas:
        candidate_bbox = item.get(bbox_key)
        candidate_text = item.get(text_key, "")
        if (
            candidate_bbox is not None
            and isinstance(candidate_bbox, (list, tuple))
            and len(candidate_bbox) == 4
            and candidate_text
        ):
            bbox = candidate_bbox
            item_text = candidate_text
            break

    if bbox is None or not item_text:
        return None

    confidence = item.get("confidence", default_conf)

    try:
        bbox = [float(v) for v in bbox]
        confidence = float(confidence)
    except (TypeError, ValueError):
        return None

    return {
        "bbox": bbox,
        "text": str(item_text),
        "confidence": max(0.0, min(1.0, confidence)),
    }


def parse_grounding_response(raw: str, family: str = "generic") -> List[Dict[str, Any]]:
    """Parse a VLM grounding response into OCR result dicts.

    Supports multiple output schemas via adaptive key detection:
    - Generic / GutenOCR: ``bbox`` + ``text``
    - Qwen-VL: ``bbox_2d`` + ``label``

    The *family* hint controls which schema is tried first, but the parser
    will fall back to the alternative schema per-item if the primary keys
    are missing.

    Args:
        raw: The raw text response from the VLM.
        family: Model family hint (``"generic"``, ``"qwen_vl"``, ``"gutenocr"``).

    Returns:
        List of dicts, each with ``bbox`` (4-element list), ``text`` (str),
        and ``confidence`` (float).
    """
    text = raw.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    data = _extract_json_array(text)
    if data is None:
        # Last resort: try the whole string
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
        extracted = _extract_item(item, family)
        if extracted is not None:
            results.append(extracted)

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
    instructions: Optional[str] = None,
    languages: Optional[List[str]] = None,
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
            Overrides the auto-generated prompt entirely.
        instructions: Additional instructions appended to the auto-generated
            prompt (ignored when *prompt* is set).
        languages: Optional language codes for VLM hint (ignored when *prompt* is set).

    Returns:
        Tuple of (ocr_results, (image_width, image_height)).
        ``ocr_results`` are in *image* pixel coordinates (not yet scaled).
    """
    from natural_pdf.core.vlm_client import generate, get_default_client
    from natural_pdf.core.vlm_prompts import build_ocr_prompt, detect_model_family

    # Resolve effective model name for family detection
    effective_model = model
    if effective_model is None:
        _, default_model = get_default_client()
        effective_model = default_model

    family = detect_model_family(effective_model)

    if family == "generic":
        logger.warning(
            "VLM OCR: model %r is not a recognized grounding model. "
            "Bounding-box accuracy may be poor. For best results use a "
            "Qwen-VL family model (e.g. Qwen/Qwen3-VL-2B-Instruct).",
            effective_model,
        )
    elif family == "openai":
        logger.info(
            "VLM OCR: OpenAI models are not optimized for bounding-box "
            "grounding. Coordinates may be imprecise. For accurate bbox "
            "output, use a Qwen-VL model via OpenRouter.",
        )

    # Use family-specific prompt unless the caller supplied a custom one
    if prompt is not None:
        effective_prompt = prompt
    else:
        effective_prompt = build_ocr_prompt(grounding=True, family=family, languages=languages)
        if instructions:
            effective_prompt = f"{effective_prompt}\n\n{instructions}"

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

    logger.info(
        "VLM OCR: received %d-char response from %s (family=%s).",
        len(raw_response),
        effective_model,
        family,
    )
    logger.debug("VLM OCR raw response (first 500 chars): %s", raw_response[:500])

    results = parse_grounding_response(raw_response, family=family)

    if not results and raw_response.strip():
        logger.warning(
            "VLM OCR: model returned text but parser found 0 results. "
            "The model may not support grounded/bbox output. "
            "Raw response (first 300 chars): %s",
            raw_response[:300],
        )

    # Convert normalized coords to image pixel coords
    if family == "gemini":
        results = normalize_gemini_coordinates(results, image.size[0], image.size[1])
    elif family == "qwen_vl":
        results = normalize_qwen_coordinates(results, image.size[0], image.size[1])

    return results, image.size
