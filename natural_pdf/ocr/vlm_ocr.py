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
from natural_pdf.utils.option_validation import resolve_auto_device

logger = logging.getLogger(__name__)


_FAMILY_DEFAULT_CONFIDENCE = {
    "qwen_vl": 0.75,
    "gemini": 0.75,
    "openai": 0.5,
    "gutenocr": 0.8,
    "glm_ocr": 0.9,
    "dots_mocr": 0.9,
    "chandra": 0.9,
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


# ---------------------------------------------------------------------------
# GLM-OCR: in-process layout detection + text recognition
# ---------------------------------------------------------------------------

_LAYOUT_MODEL = "PaddlePaddle/PP-DocLayoutV3_safetensors"
_GLM_OCR_PROMPT = "Text Recognition:"

# Layout labels to skip — purely visual content with no text to recognize.
_GLM_SKIP_LABELS = {
    "figure",
    "image",
    "chart",
    "seal",
}

_layout_detector_cache: Dict[str, Any] = {}
_layout_cache_lock = __import__("threading").Lock()


def _get_layout_detector(model_name: str = _LAYOUT_MODEL) -> Any:
    """Load and cache the PP-DocLayout-V3 layout detector."""
    with _layout_cache_lock:
        if model_name not in _layout_detector_cache:
            try:
                import torch
                from transformers import PPDocLayoutV3ForObjectDetection

                # transformers 5.x renamed to PPDocLayoutV3ImageProcessorFast
                try:
                    from transformers import (
                        PPDocLayoutV3ImageProcessorFast as PPDocLayoutV3ImageProcessor,
                    )
                except ImportError:
                    from transformers import PPDocLayoutV3ImageProcessor
            except ImportError as exc:
                raise RuntimeError(
                    "Layout detection requires transformers and torch. "
                    "Install with: pip install transformers torch"
                ) from exc

            logger.info("Loading layout model %r ...", model_name)
            processor = PPDocLayoutV3ImageProcessor.from_pretrained(model_name)
            model = PPDocLayoutV3ForObjectDetection.from_pretrained(model_name)
            model.eval()

            device = resolve_auto_device()
            model = model.to(device)

            _layout_detector_cache[model_name] = {
                "processor": processor,
                "model": model,
                "device": device,
            }
            logger.info("Layout model loaded on %s.", device)
        return _layout_detector_cache[model_name]


def _detect_layout_regions(
    image: Image.Image,
    threshold: float = 0.4,
) -> List[Dict[str, Any]]:
    """Run PP-DocLayout-V3 on an image, return detected regions.

    Each region is a dict with keys: ``label``, ``bbox`` (pixel coords),
    ``confidence``.
    """
    import torch

    det = _get_layout_detector()
    processor = det["processor"]
    model = det["model"]
    device = det["device"]

    inputs = processor(images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device=device)
    raw = processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes
    )[0]

    id2label = model.config.id2label
    regions = []
    for score, label_id, box in zip(
        raw["scores"].tolist(), raw["labels"].tolist(), raw["boxes"].tolist()
    ):
        label = id2label.get(label_id, str(label_id)).lower()
        regions.append(
            {
                "label": label,
                "bbox": box,  # [x0, y0, x1, y1] in pixels
                "confidence": score,
            }
        )
    return regions


def _suppress_contained_regions(
    regions: List[Dict[str, Any]],
    *,
    nms_iou_same: float = 0.6,
    nms_iou_diff: float = 0.95,
    overlap_threshold: float = 0.7,
    min_box_size: int = 6,
) -> List[Dict[str, Any]]:
    """Remove overlapping layout regions to avoid duplicate OCR.

    Reproduces the three-stage strategy used by PaddleOCR-VL's native
    pipeline (which only runs on the non-MLX path):

    1. **Class-aware NMS** — greedy suppression sorted by confidence.
       Same-label boxes suppressed at IoU >= *nms_iou_same* (0.6),
       cross-label at IoU >= *nms_iou_diff* (0.95).
    2. **Overlap filter** — pairwise check using "small" mode
       (intersection / smaller area).  When ratio > *overlap_threshold*
       (0.7), the smaller-area box is dropped.  Boxes smaller than
       *min_box_size* in either dimension are also dropped.
    """
    if len(regions) <= 1:
        return regions

    def _area(b: List[float]) -> float:
        return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    def _intersection_area(a: List[float], b: List[float]) -> float:
        ix0 = max(a[0], b[0])
        iy0 = max(a[1], b[1])
        ix1 = min(a[2], b[2])
        iy1 = min(a[3], b[3])
        return max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)

    def _iou(a: List[float], b: List[float]) -> float:
        inter = _intersection_area(a, b)
        union = _area(a) + _area(b) - inter
        return inter / union if union > 0 else 0.0

    # --- Stage 1: class-aware NMS (greedy, sorted by confidence desc) ---
    order = sorted(
        range(len(regions)),
        key=lambda i: regions[i].get("confidence", 0.0),
        reverse=True,
    )
    keep: List[bool] = [True] * len(regions)

    for pos, i in enumerate(order):
        if not keep[i]:
            continue
        bbox_i = regions[i]["bbox"]
        label_i = regions[i].get("label", "")
        for j in order[pos + 1 :]:
            if not keep[j]:
                continue
            bbox_j = regions[j]["bbox"]
            label_j = regions[j].get("label", "")
            iou_val = _iou(bbox_i, bbox_j)
            threshold = nms_iou_same if label_i == label_j else nms_iou_diff
            if iou_val >= threshold:
                keep[j] = False

    # --- Stage 2: pairwise overlap filter ("small" mode) + min-size ---
    survivors = [i for i in range(len(regions)) if keep[i]]

    dropped: set = set()
    for i in survivors:
        bbox_i = regions[i]["bbox"]
        w = bbox_i[2] - bbox_i[0]
        h = bbox_i[3] - bbox_i[1]
        if w < min_box_size or h < min_box_size:
            dropped.add(i)

    for ai, i in enumerate(survivors):
        if i in dropped:
            continue
        bbox_i = regions[i]["bbox"]
        area_i = _area(bbox_i)
        for j in survivors[ai + 1 :]:
            if j in dropped:
                continue
            bbox_j = regions[j]["bbox"]
            area_j = _area(bbox_j)
            inter = _intersection_area(bbox_i, bbox_j)
            smaller_area = min(area_i, area_j)
            if smaller_area <= 0:
                continue
            ratio = inter / smaller_area
            if ratio > overlap_threshold:
                # Drop the smaller box.
                if area_i >= area_j:
                    dropped.add(j)
                else:
                    dropped.add(i)
                    break  # i is dropped, stop comparing it

    final = [i for i in survivors if i not in dropped]
    return [regions[i] for i in final]


def _detect_cluster_regions(
    image: Image.Image,
    *,
    expand_px: int = 2,
    languages: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Detect text regions by clustering word-level boxes.

    Uses rapidocr in detect-only mode to find word-level bounding boxes,
    expands each by *expand_px* pixels so nearby boxes overlap, then
    dissolves overlapping boxes into natural clusters.

    This produces intermediate-granularity regions (typically 30-80 per page)
    that are well-suited for per-crop VLM recognition. It avoids the
    pathological case where PP-DocLayout-V3 detects a dense form as one
    giant "table" region that causes VLM hallucination.

    Args:
        image: Pre-rendered page image.
        expand_px: Pixels to expand each word box before dissolving.
        languages: Language codes for the detection engine.

    Returns:
        List of region dicts with ``label``, ``bbox``, ``confidence`` keys.
        Bounding boxes are in image pixel coordinates.
    """
    from natural_pdf.ocr.unified_dispatch import run_detection

    img_w, img_h = image.size
    logger.info(
        "layout+OCR: running cluster detection on %dx%d image (expand=%dpx) ...",
        img_w,
        img_h,
        expand_px,
    )

    # Step 1: Get word-level bounding boxes via rapidocr detect-only
    word_regions = run_detection(image=image, engine_name="rapidocr", languages=languages)
    if not word_regions:
        logger.info("layout+OCR: cluster detection found no words.")
        return []

    logger.info("layout+OCR: cluster detection found %d word boxes.", len(word_regions))

    # Step 2: Expand each box so nearby words overlap, then merge via union-find.
    boxes = []
    for r in word_regions:
        x0, y0, x1, y1 = r["bbox"]
        boxes.append(
            [
                max(0.0, x0 - expand_px),
                max(0.0, y0 - expand_px),
                min(float(img_w), x1 + expand_px),
                min(float(img_h), y1 + expand_px),
            ]
        )

    parent = list(range(len(boxes)))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(i: int, j: int) -> None:
        ri, rj = _find(i), _find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            bi, bj = boxes[i], boxes[j]
            if bi[0] < bj[2] and bi[2] > bj[0] and bi[1] < bj[3] and bi[3] > bj[1]:
                _union(i, j)

    # Group by cluster root and compute bounding hull
    from collections import defaultdict

    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(len(boxes)):
        groups[_find(i)].append(i)

    clusters = []
    for members in groups.values():
        x0 = min(boxes[m][0] for m in members)
        y0 = min(boxes[m][1] for m in members)
        x1 = max(boxes[m][2] for m in members)
        y1 = max(boxes[m][3] for m in members)
        clusters.append(
            {
                "label": "text",
                "bbox": [x0, y0, x1, y1],
                "confidence": 0.9,
            }
        )

    logger.info(
        "layout+OCR: cluster detection merged %d words into %d clusters.",
        len(word_regions),
        len(clusters),
    )
    return clusters


def _run_layout_ocr_on_image(
    image: Image.Image,
    *,
    model: Optional[str],
    client: Optional[Any],
    max_new_tokens: int,
    prompt: str = _GLM_OCR_PROMPT,
    default_confidence: float = 0.9,
    crop_padding: int = 1,
    layout_threshold: float = 0.15,
    skip_labels: Optional[set] = None,
    detection_engine: Optional[str] = None,
    use_cluster: bool = False,
    languages: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], Tuple[int, int]]:
    """Run layout detection + per-crop VLM recognition on a pre-rendered image.

    1. Detect text regions via PP-DocLayout-V3 (default), a classic
       OCR engine in detect-only mode (when *detection_engine* is set),
       or cluster detection (when *use_cluster* is True).
    2. For each text region, crop the image (with padding) and send to VLM.
    3. Return results with bboxes from detection + text from VLM.

    When *use_cluster* is True, detection uses rapidocr in detect-only mode
    to find word-level boxes, expands each by a small margin, then dissolves
    overlapping boxes into natural clusters. This avoids the pathological case
    where PP-DocLayout-V3 detects a dense form as one giant "table" region.

    Returns results in image pixel coordinates.
    """
    from natural_pdf.core.vlm_client import generate

    if skip_labels is None:
        skip_labels = _GLM_SKIP_LABELS

    img_w, img_h = image.size

    # Step 1: Detection
    if use_cluster:
        text_regions = _detect_cluster_regions(image, languages=languages)
        regions = text_regions
    elif detection_engine is not None:
        from natural_pdf.ocr.unified_dispatch import run_detection

        logger.info(
            "layout+OCR: running %s detection on %dx%d image ...",
            detection_engine,
            img_w,
            img_h,
        )
        regions = run_detection(
            image=image,
            engine_name=detection_engine,
            languages=languages,
        )
        text_regions = regions
    else:
        logger.info("layout+OCR: running layout detection on %dx%d image ...", img_w, img_h)
        regions = _detect_layout_regions(image, threshold=layout_threshold)
        text_regions = [r for r in regions if r["label"] not in skip_labels]

        skipped = [r for r in regions if r["label"] in skip_labels]
        if skipped:
            skipped_summary: Dict[str, int] = {}
            for r in skipped:
                skipped_summary[r["label"]] = skipped_summary.get(r["label"], 0) + 1
            logger.info(
                "layout+OCR: skipped %d visual-only regions: %s", len(skipped), skipped_summary
            )

    if not text_regions:
        logger.info("layout+OCR: no text regions detected, running on full page.")
        text_regions = [{"label": "text", "bbox": [0, 0, img_w, img_h], "confidence": 1.0}]

    if not use_cluster:
        # Suppress overlapping regions to avoid duplicate OCR results.
        # (Cluster detection already produces non-overlapping regions.)
        before_dedup = len(text_regions)
        text_regions = _suppress_contained_regions(text_regions)
        if len(text_regions) < before_dedup:
            logger.info(
                "layout+OCR: suppressed %d overlapping regions (%d → %d).",
                before_dedup - len(text_regions),
                before_dedup,
                len(text_regions),
            )

    logger.info(
        "layout+OCR: detected %d text regions (of %d total). Running recognition on each ...",
        len(text_regions),
        len(regions),
    )

    # Step 2: Run VLM recognition on each text region
    results = []
    image_area = img_w * img_h

    from tqdm.auto import tqdm

    from natural_pdf.utils.text_utils import detect_repetition

    for region in tqdm(text_regions, desc="OCR regions", unit="region"):
        bbox = region["bbox"]
        x0, y0, x1, y1 = [int(round(v)) for v in bbox]

        # Clamp to image bounds with padding
        x0 = max(0, x0 - crop_padding)
        y0 = max(0, y0 - crop_padding)
        x1 = min(img_w, x1 + crop_padding)
        y1 = min(img_h, y1 + crop_padding)

        if x1 - x0 < 2 or y1 - y0 < 2:
            continue

        # Scale token budget by region area relative to full image.
        # 2x safety margin for dense tables/forms; floor of 256 tokens.
        region_area = (x1 - x0) * (y1 - y0)
        area_ratio = region_area / image_area
        region_max = max(256, min(max_new_tokens, int(max_new_tokens * area_ratio * 2)))

        crop = image.crop((x0, y0, x1, y1))
        raw_text = generate(
            crop,
            prompt,
            model=model,
            client=client,
            max_new_tokens=region_max,
        ).strip()

        if not raw_text:
            continue

        # Detect VLM hallucination: repetitive output that compresses unusually well.
        if detect_repetition(raw_text):
            label = region.get("label", "unknown")
            logger.warning(
                "Region '%s' produced repetitive output (%d chars, budget %d tokens) "
                "— discarding. If this region contains dense text, pass a higher "
                "max_new_tokens.",
                label,
                len(raw_text),
                region_max,
            )
            continue

        results.append(
            {
                "bbox": [float(x0), float(y0), float(x1), float(y1)],
                "text": raw_text,
                "confidence": default_confidence,
            }
        )

    logger.info("layout+OCR: recognized text in %d of %d regions.", len(results), len(text_regions))
    return results, (img_w, img_h)


def _run_glm_ocr_on_image(
    image: Image.Image,
    *,
    model: Optional[str],
    client: Optional[Any],
    max_new_tokens: int,
) -> Tuple[List[Dict[str, Any]], Tuple[int, int]]:
    """Run layout detection + GLM-OCR text recognition on a pre-rendered image.

    Convenience wrapper around :func:`_run_layout_ocr_on_image` with
    GLM-OCR defaults.
    """
    return _run_layout_ocr_on_image(
        image,
        model=model,
        client=client,
        max_new_tokens=max_new_tokens,
        prompt=_GLM_OCR_PROMPT,
        default_confidence=_FAMILY_DEFAULT_CONFIDENCE["glm_ocr"],
    )


def _run_glm_ocr_with_layout(
    host: Any,
    *,
    image: Any,
    model: Optional[str],
    client: Optional[Any],
    resolution: int,
    render_kwargs: Optional[Dict[str, Any]],
    max_new_tokens: int,
) -> Tuple[List[Dict[str, Any]], Tuple[int, int]]:
    """Run layout detection + GLM-OCR on a host (renders internally).

    Thin wrapper around :func:`_run_glm_ocr_on_image`.
    """
    render_fn = getattr(host, "render", None)
    if not callable(render_fn):
        raise AttributeError("Host object does not support render() for VLM OCR.")

    kwargs = dict(render_kwargs or {})
    with pdf_render_lock:
        image = render_fn(resolution=resolution, **kwargs)

    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected render() to return a PIL Image, got {type(image).__name__}")

    return _run_glm_ocr_on_image(image, model=model, client=client, max_new_tokens=max_new_tokens)


def resolve_glm_ocr_model() -> str:
    """Pick the best GLM-OCR model variant for the current platform.

    Returns the MLX 4-bit model on Apple Silicon (requires ``pip install
    mlx-vlm``), otherwise the full HuggingFace model for GPU/CPU.
    """
    import platform

    if platform.machine() == "arm64" and platform.system() == "Darwin":
        return "mlx-community/GLM-OCR-4bit"
    return "zai-org/GLM-OCR"


def resolve_chandra_model() -> str:
    """Pick the best Chandra model variant for the current platform.

    Returns the MLX 4-bit model on Apple Silicon (requires ``pip install
    mlx-vlm``), otherwise the full HuggingFace model for GPU/CPU.
    """
    import platform

    if platform.machine() == "arm64" and platform.system() == "Darwin":
        return "mlx-community/chandra-4bit"
    return "datalab-to/chandra"


def resolve_dots_model() -> str:
    """Pick the best dots.mocr model variant for the current platform.

    Returns the MLX 4-bit model on Apple Silicon (requires ``pip install
    mlx-vlm``), otherwise the full HuggingFace model for GPU/CPU.
    """
    import platform

    if platform.machine() == "arm64" and platform.system() == "Darwin":
        return "mlx-community/dots.mocr-4bit"
    return "rednote-hilab/dots.mocr"


# ---------------------------------------------------------------------------
# dots.mocr: single-model layout detection + text recognition
# ---------------------------------------------------------------------------

# Layout labels that contain no useful text to return as OCR results.
_DOTS_SKIP_LABELS = {"Picture"}


def _html_table_to_tsv(html: str) -> str:
    """Convert an HTML table to tab-separated values."""
    from html.parser import HTMLParser

    class _TableParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.rows: List[List[str]] = []
            self._current_row: List[str] = []
            self._current_cell: List[str] = []
            self._in_cell = False

        def handle_starttag(self, tag, attrs):
            if tag == "tr":
                self._current_row = []
            elif tag in ("td", "th"):
                self._in_cell = True
                self._current_cell = []

        def handle_endtag(self, tag):
            if tag in ("td", "th"):
                self._in_cell = False
                self._current_row.append("".join(self._current_cell).strip())
            elif tag == "tr":
                if self._current_row:
                    self.rows.append(self._current_row)

        def handle_data(self, data):
            if self._in_cell:
                self._current_cell.append(data)

    parser = _TableParser()
    parser.feed(html)
    return "\n".join("\t".join(cells) for cells in parser.rows)


def _parse_dots_mocr_response(raw: str) -> List[Dict[str, Any]]:
    """Parse dots.mocr structured JSON output into OCR result dicts.

    dots.mocr returns a JSON object with a ``layouts`` array, where each
    element has ``bbox`` ([x1, y1, x2, y2] in pixel coords), ``category``,
    and ``text``.  Tables are converted from HTML to TSV.
    """
    text = raw.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try raw_decode from the first '{' (precise, avoids greedy regex)
        decoder = json.JSONDecoder()
        idx = text.find("{")
        if idx != -1:
            try:
                data, _ = decoder.raw_decode(text, idx)
            except json.JSONDecodeError:
                logger.warning("dots.mocr: failed to parse response as JSON: %s", text[:200])
                return []
        else:
            logger.warning("dots.mocr: no JSON object found in response: %s", text[:200])
            return []

    layouts = data.get("layouts") if isinstance(data, dict) else None
    if not isinstance(layouts, list):
        # Maybe the response is the array directly
        if isinstance(data, list):
            layouts = data
        else:
            logger.warning("dots.mocr: response has no 'layouts' array.")
            return []

    default_conf = _FAMILY_DEFAULT_CONFIDENCE["dots_mocr"]
    results = []
    for item in layouts:
        if not isinstance(item, dict):
            continue

        category = str(item.get("category") or item.get("type") or "")
        if category.strip().lower() in {"picture"}:
            continue

        bbox = item.get("bbox")
        if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue

        item_text = item.get("text") or item.get("content") or ""
        if not item_text:
            continue

        # For tables, keep raw HTML and convert to TSV for display
        raw_html = None
        if category.strip().lower() == "table" and "<table" in item_text.lower():
            raw_html = item_text
            item_text = _html_table_to_tsv(item_text)

        try:
            x0, y0, x1, y1 = [float(v) for v in bbox]
        except (TypeError, ValueError):
            continue

        # Normalize inverted coordinates
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0

        # Skip degenerate boxes
        if x1 - x0 < 0.1 or y1 - y0 < 0.1:
            continue

        result_dict = {
            "bbox": [x0, y0, x1, y1],
            "text": str(item_text),
            "confidence": default_conf,
        }
        cat = category.strip()
        if cat:
            result_dict["source_category"] = cat
        if raw_html:
            result_dict["raw_html"] = raw_html
        results.append(result_dict)

    return results


def _run_dots_mocr_on_image(
    image: Image.Image,
    *,
    model: Optional[str],
    client: Optional[Any],
    max_new_tokens: int,
) -> Tuple[List[Dict[str, Any]], Tuple[int, int]]:
    """Run dots.mocr on a pre-rendered image.

    Returns results in image pixel coordinates.
    """
    from natural_pdf.core.vlm_client import generate
    from natural_pdf.core.vlm_prompts import DOTS_MOCR_PROMPT

    img_w, img_h = image.size
    logger.info("dots.mocr: running on %dx%d image ...", img_w, img_h)

    raw_response = generate(
        image,
        DOTS_MOCR_PROMPT,
        model=model,
        client=client,
        max_new_tokens=max_new_tokens,
    )

    logger.info("dots.mocr: received %d-char response.", len(raw_response))
    logger.debug("dots.mocr raw response (first 500 chars): %s", raw_response[:500])

    results = _parse_dots_mocr_response(raw_response)
    logger.info("dots.mocr: parsed %d text regions.", len(results))

    return results, (img_w, img_h)


def _run_dots_mocr(
    host: Any,
    *,
    model: Optional[str],
    client: Optional[Any],
    resolution: int,
    render_kwargs: Optional[Dict[str, Any]],
    max_new_tokens: int,
) -> Tuple[List[Dict[str, Any]], Tuple[int, int]]:
    """Run dots.mocr on a host (renders internally).

    Thin wrapper around :func:`_run_dots_mocr_on_image`.
    """
    render_fn = getattr(host, "render", None)
    if not callable(render_fn):
        raise AttributeError("Host object does not support render() for VLM OCR.")

    kwargs = dict(render_kwargs or {})
    with pdf_render_lock:
        image = render_fn(resolution=resolution, **kwargs)

    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected render() to return a PIL Image, got {type(image).__name__}")

    return _run_dots_mocr_on_image(image, model=model, client=client, max_new_tokens=max_new_tokens)


def create_table_regions_from_ocr(
    page: Any,
    results: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """Split OCR results: table items become regions, rest stay as text results.

    Table results are converted into Region objects with ``region_type="table"``,
    ``alt_text`` set to the TSV content (used by ``extract_text()``), and
    ``raw_html`` stashed in ``region.metadata`` for ``extract_table()``.
    Regions are registered on the page so they appear in selector queries.

    Args:
        page: The Page object to create regions on.
        results: Scaled OCR result dicts (in PDF coordinates).

    Returns:
        Tuple of (non_table_results, created_regions).
    """
    from natural_pdf.elements.region import Region

    non_table = []
    regions = []

    for r in results:
        cat = r.get("source_category", "").lower()
        if cat == "table":
            bbox = r["bbox"]
            region = Region(page, (bbox[0], bbox[1], bbox[2], bbox[3]))
            region.region_type = "table"
            region.source = "dots.mocr"
            region.alt_text = r["text"]  # TSV content
            raw_html = r.get("raw_html")
            if raw_html:
                region.metadata["raw_html"] = raw_html
            # Register on the page so extract_text() and selectors can find it
            page.add_region(region, source="dots.mocr")
            regions.append(region)
        else:
            non_table.append(r)

    return non_table, regions


# ---------------------------------------------------------------------------
# Chandra: VLM-based OCR with HTML layout output
# ---------------------------------------------------------------------------

# Labels whose div content should be skipped (no useful text).
_CHANDRA_SKIP_LABELS = {"image", "figure"}

# Chandra v0.1.x uses data-bbox="[x0, y0, x1, y1]" (bracket notation).
# We accept both bracket and bare formats for robustness.
_RE_DIV_BBOX = re.compile(
    r'<div[^>]*\bdata-bbox="([^"]+)"[^>]*\bdata-label="([^"]+)"[^>]*>(.*?)</div>',
    re.DOTALL | re.IGNORECASE,
)
_RE_DIV_BBOX_ALT = re.compile(
    r'<div[^>]*\bdata-label="([^"]+)"[^>]*\bdata-bbox="([^"]+)"[^>]*>(.*?)</div>',
    re.DOTALL | re.IGNORECASE,
)
_RE_HTML_TAGS = re.compile(r"<[^>]+>")


def _parse_bbox_str(bbox_str: str) -> Optional[Tuple[float, float, float, float]]:
    """Parse a bbox string in either ``[x0, y0, x1, y1]`` or ``x0 y0 x1 y1`` format."""
    s = bbox_str.strip().strip("[]")
    # Handle both comma-separated and space-separated
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
    else:
        parts = s.split()
    if len(parts) != 4:
        return None
    try:
        return tuple(float(p) for p in parts)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def _parse_chandra_response(raw: str, img_w: int, img_h: int) -> List[Dict[str, Any]]:
    """Parse Chandra's HTML layout output into OCR result dicts.

    Chandra v0.1.x returns HTML divs with ``data-bbox="[x0, y0, x1, y1]"``
    (normalized 0–1024) and ``data-label="Label"`` attributes.  We extract
    the text content, strip HTML tags, and convert bbox to pixel coordinates.
    """
    default_conf = _FAMILY_DEFAULT_CONFIDENCE["chandra"]
    results: List[Dict[str, Any]] = []

    # Try both attribute orderings
    matches = _RE_DIV_BBOX.findall(raw)
    if matches:
        parsed = [(bbox_str, label, content) for bbox_str, label, content in matches]
    else:
        matches_alt = _RE_DIV_BBOX_ALT.findall(raw)
        parsed = [(bbox_str, label, content) for label, bbox_str, content in matches_alt]

    for bbox_str, label, content in parsed:
        label_lower = label.strip().lower()
        if label_lower in _CHANDRA_SKIP_LABELS:
            continue

        # Parse bbox — "[x0, y0, x1, y1]" normalized 0-1024
        coords = _parse_bbox_str(bbox_str)
        if coords is None:
            continue
        nx0, ny0, nx1, ny1 = coords

        # Convert from 0-1024 normalized to pixel coordinates
        x0 = nx0 / 1024.0 * img_w
        y0 = ny0 / 1024.0 * img_h
        x1 = nx1 / 1024.0 * img_w
        y1 = ny1 / 1024.0 * img_h

        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0
        if x1 - x0 < 0.1 or y1 - y0 < 0.1:
            continue

        # Check if this is a table — keep raw HTML, convert to TSV for text
        is_table = label_lower == "table" and "<table" in content.lower()
        raw_html = None
        if is_table:
            raw_html = content.strip()
            text = _html_table_to_tsv(content)
        else:
            # Strip HTML tags for plain text
            text = _RE_HTML_TAGS.sub("", content)
            text = re.sub(r"  +", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = text.strip()

        if not text:
            continue

        result_dict: Dict[str, Any] = {
            "bbox": [x0, y0, x1, y1],
            "text": text,
            "confidence": default_conf,
            "source_category": label.strip(),
        }
        if raw_html:
            result_dict["raw_html"] = raw_html
        results.append(result_dict)

    return results


def _run_chandra_on_image(
    image: Image.Image,
    *,
    model: Optional[str],
    client: Optional[Any],
    max_new_tokens: int,
) -> Tuple[List[Dict[str, Any]], Tuple[int, int]]:
    """Run Chandra VLM on a pre-rendered image.

    Returns results in image pixel coordinates.
    """
    from natural_pdf.core.vlm_client import generate
    from natural_pdf.core.vlm_prompts import CHANDRA_OCR_LAYOUT_PROMPT

    img_w, img_h = image.size
    logger.info("Chandra: running on %dx%d image ...", img_w, img_h)

    raw_response = generate(
        image,
        CHANDRA_OCR_LAYOUT_PROMPT,
        model=model,
        client=client,
        max_new_tokens=max_new_tokens,
    )

    logger.info("Chandra: received %d-char response.", len(raw_response))
    logger.debug("Chandra raw response (first 500 chars): %s", raw_response[:500])

    results = _parse_chandra_response(raw_response, img_w, img_h)
    logger.info("Chandra: parsed %d text regions.", len(results))

    return results, (img_w, img_h)


def _run_chandra(
    host: Any,
    *,
    model: Optional[str],
    client: Optional[Any],
    resolution: int,
    render_kwargs: Optional[Dict[str, Any]],
    max_new_tokens: int,
) -> Tuple[List[Dict[str, Any]], Tuple[int, int]]:
    """Run Chandra VLM on a host (renders internally).

    Thin wrapper around :func:`_run_chandra_on_image`.
    """
    render_fn = getattr(host, "render", None)
    if not callable(render_fn):
        raise AttributeError("Host object does not support render() for VLM OCR.")

    kwargs = dict(render_kwargs or {})
    with pdf_render_lock:
        image = render_fn(resolution=resolution, **kwargs)

    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected render() to return a PIL Image, got {type(image).__name__}")

    return _run_chandra_on_image(image, model=model, client=client, max_new_tokens=max_new_tokens)


def run_vlm_ocr_on_image(
    image: Image.Image,
    *,
    model: Optional[str] = None,
    client: Optional[Any] = None,
    max_new_tokens: Optional[int] = None,
    prompt: Optional[str] = None,
    instructions: Optional[str] = None,
    languages: Optional[List[str]] = None,
    layout: Optional[bool | str] = None,
) -> Tuple[List[Dict[str, Any]], Tuple[int, int]]:
    """Run VLM-based OCR on a pre-rendered image.

    Dispatches to family-specific handlers based on the detected model family.

    Args:
        image: Pre-rendered PIL Image.
        model: HuggingFace model ID or remote model name.
        client: OpenAI-compatible client.
        max_new_tokens: Max tokens for VLM generation. In layout+OCR mode,
            each region's budget is scaled by its area relative to the
            full image to prevent VLM hallucination loops.
        prompt: Custom prompt (overrides auto-generated prompt).
        instructions: Additional instructions appended to auto-generated prompt
            (ignored when *prompt* is set).
        languages: Optional language codes for VLM hint.
        layout: If ``True``, run layout detection (PP-DocLayout-V3) first
            and send each cropped region to the VLM for recognition.
            If ``"cluster"``, use rapidocr detect-only to find word boxes,
            expand them slightly, and dissolve overlapping boxes into
            natural clusters. Best for dense forms where PP-DocLayout-V3
            detects the whole page as one giant "table" region.
            If another string (e.g. ``"rapidocr"``, ``"paddle"``), use that
            classic OCR engine in detect-only mode for line-level boxes
            instead of PP-DocLayout-V3.
            If ``None`` (default), auto-detect based on model family
            (``glm_ocr`` uses layout; grounding models don't).
            If ``False``, always use full-page prompt.

    Returns:
        Tuple of (ocr_results, (image_width, image_height)).
        ``ocr_results`` are in *image* pixel coordinates.
    """
    from natural_pdf.core.vlm_client import DEFAULT_VLM_MAX_TOKENS, generate, get_default_client
    from natural_pdf.core.vlm_prompts import build_ocr_prompt, detect_model_family

    if max_new_tokens is None:
        max_new_tokens = DEFAULT_VLM_MAX_TOKENS

    # Resolve effective model name for family detection
    effective_model = model
    if effective_model is None:
        _, default_model = get_default_client()
        effective_model = default_model

    family = detect_model_family(effective_model)

    # Unpack layout parameter:
    #   "cluster" → rapidocr detect-only → expand → dissolve into clusters
    #   str       → use that engine for detection, force layout pipeline
    #   True      → PP-DocLayout-V3
    #   None      → auto-detect per family
    #   False     → full-page, no layout
    detection_engine: Optional[str] = None
    use_cluster = False
    if isinstance(layout, str):
        if layout == "cluster":
            use_cluster = True
        else:
            detection_engine = layout
        use_layout = True
    elif layout is not None:
        use_layout = layout
    else:
        use_layout = family == "glm_ocr"

    if use_layout:
        effective_prompt = prompt or _GLM_OCR_PROMPT
        default_conf = _FAMILY_DEFAULT_CONFIDENCE.get(family, 0.9)
        return _run_layout_ocr_on_image(
            image,
            model=model,
            client=client,
            max_new_tokens=max_new_tokens,
            prompt=effective_prompt,
            default_confidence=default_conf,
            detection_engine=detection_engine,
            use_cluster=use_cluster,
            languages=languages,
        )

    # Family-specific single-pass handlers
    if family == "dots_mocr":
        return _run_dots_mocr_on_image(
            image, model=model, client=client, max_new_tokens=max_new_tokens
        )
    elif family == "chandra":
        return _run_chandra_on_image(
            image, model=model, client=client, max_new_tokens=max_new_tokens
        )
    elif family == "generic":
        logger.warning(
            "VLM OCR: model %r is not a recognized grounding model. "
            "Bounding-box accuracy may be poor. For best results use a "
            "Qwen-VL family model (e.g. Qwen/Qwen3-VL-2B-Instruct), "
            "or pass layout=True to use layout detection + per-crop recognition.",
            effective_model,
        )
    elif family == "openai":
        logger.info(
            "VLM OCR: OpenAI models are not optimized for bounding-box "
            "grounding. Coordinates may be imprecise. For accurate bbox "
            "output, use a Qwen-VL model via OpenRouter.",
        )

    # Full-page grounding prompt path
    if prompt is not None:
        effective_prompt = prompt
    else:
        effective_prompt = build_ocr_prompt(grounding=True, family=family, languages=languages)
        if instructions:
            effective_prompt = f"{effective_prompt}\n\n{instructions}"

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


def run_vlm_ocr(
    host: Any,
    *,
    model: Optional[str] = None,
    client: Optional[Any] = None,
    resolution: int = 144,
    render_kwargs: Optional[Dict[str, Any]] = None,
    max_new_tokens: Optional[int] = None,
    prompt: Optional[str] = None,
    instructions: Optional[str] = None,
    languages: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], Tuple[int, int]]:
    """Run VLM-based OCR on a page/region (renders internally).

    Thin wrapper around :func:`run_vlm_ocr_on_image`.
    """
    render_fn = getattr(host, "render", None)
    if not callable(render_fn):
        raise AttributeError("Host object does not support render() for VLM OCR.")

    kwargs = dict(render_kwargs or {})
    with pdf_render_lock:
        image = render_fn(resolution=resolution, **kwargs)

    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected render() to return a PIL Image, got {type(image).__name__}")

    return run_vlm_ocr_on_image(
        image,
        model=model,
        client=client,
        max_new_tokens=max_new_tokens,
        prompt=prompt,
        instructions=instructions,
        languages=languages,
    )
