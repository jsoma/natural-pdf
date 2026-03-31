"""OCR alignment: spatially aligns OCR elements from multiple engines into comparison regions."""

from __future__ import annotations

import logging
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

from natural_pdf.ocr.comparison import (
    ComparisonRegion,
    _edit_distance_ratio,
    classify_region,
    compute_consensus,
    find_outlier,
    normalize_text,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tagged element wrapper
# ---------------------------------------------------------------------------


class _TaggedBox:
    """Lightweight wrapper around an OCR element with engine tag."""

    __slots__ = (
        "engine",
        "element",
        "x0",
        "top",
        "x1",
        "bottom",
        "height",
        "y_center",
        "text",
        "confidence",
    )

    def __init__(self, engine: str, element: Any):
        self.engine = engine
        self.element = element
        self.x0 = float(element.x0)
        self.top = float(element.top)
        self.x1 = float(element.x1)
        self.bottom = float(element.bottom)
        self.height = self.bottom - self.top
        self.y_center = (self.top + self.bottom) / 2.0
        self.text = getattr(element, "text", "") or ""
        conf = getattr(element, "confidence", None)
        self.confidence = float(conf) if conf is not None else None


# ---------------------------------------------------------------------------
# Row-based alignment
# ---------------------------------------------------------------------------


def _vertical_overlap_ratio(a_top: float, a_bottom: float, b_top: float, b_bottom: float) -> float:
    """Compute vertical overlap ratio: intersection / min(height_a, height_b)."""
    intersection = max(0, min(a_bottom, b_bottom) - max(a_top, b_top))
    min_height = min(a_bottom - a_top, b_bottom - b_top)
    if min_height <= 0:
        return 0.0
    return intersection / min_height


def _split_into_columns(boxes: List[_TaggedBox], median_width: float) -> List[List[_TaggedBox]]:
    """Split a horizontal band into column segments by detecting large x-gaps."""
    if not boxes:
        return []
    sorted_boxes = sorted(boxes, key=lambda b: b.x0)
    segments: List[List[_TaggedBox]] = [[sorted_boxes[0]]]

    gap_threshold = median_width * 3.0 if median_width > 0 else 50.0

    for i in range(1, len(sorted_boxes)):
        prev = sorted_boxes[i - 1]
        curr = sorted_boxes[i]
        gap = curr.x0 - prev.x1
        if gap > gap_threshold:
            segments.append([curr])
        else:
            segments[-1].append(curr)

    return segments


def _build_region_from_segment(
    segment: List[_TaggedBox],
    engines: List[str],
    normalize_mode: str,
    total_engines: int,
) -> ComparisonRegion:
    """Build a ComparisonRegion from a list of tagged boxes in one segment."""
    # Union bbox
    x0 = min(b.x0 for b in segment)
    top = min(b.top for b in segment)
    x1 = max(b.x1 for b in segment)
    bottom = max(b.bottom for b in segment)

    # Per-engine text and confidence
    texts: Dict[str, str] = {}
    confidences: Dict[str, Optional[float]] = {}
    elements_map: Dict[str, list] = {}

    for engine in engines:
        engine_boxes = sorted(
            [b for b in segment if b.engine == engine],
            key=lambda b: (b.top, b.x0),
        )
        if engine_boxes:
            texts[engine] = " ".join(b.text for b in engine_boxes if b.text)
            confs = [b.confidence for b in engine_boxes if b.confidence is not None]
            confidences[engine] = sum(confs) / len(confs) if confs else None
            elements_map[engine] = [b.element for b in engine_boxes]

    # Normalize texts
    normalized_texts = {e: normalize_text(t, normalize_mode) for e, t in texts.items()}

    # Consensus
    consensus = compute_consensus(normalized_texts)

    # Edit distances
    edit_distances: Dict[str, float] = {}
    if consensus is not None:
        for engine, norm_text in normalized_texts.items():
            edit_distances[engine] = _edit_distance_ratio(consensus, norm_text)

    # Classification
    classification = classify_region(
        edit_distances,
        present_engines=len(texts),
        total_engines=total_engines,
        text_length=len(consensus) if consensus else 0,
    )

    # Outlier
    outlier = find_outlier(edit_distances)

    return ComparisonRegion(
        bbox=(x0, top, x1, bottom),
        texts=texts,
        normalized_texts=normalized_texts,
        confidences=confidences,
        consensus=consensus,
        classification=classification,
        edit_distances=edit_distances,
        outlier_engine=outlier,
        elements=elements_map,
    )


def align_by_rows(
    engine_elements: Dict[str, list],
    normalize: str = "collapse",
) -> List[ComparisonRegion]:
    """Row-based alignment: group elements into horizontal bands, split by columns.

    Handles mixed granularity by setting aside tall (block-level) boxes
    and mapping them to regions by overlap afterward.
    """
    engines = list(engine_elements.keys())
    total_engines = len(engines)

    # 1. Collect all boxes
    all_boxes: List[_TaggedBox] = []
    for engine, elems in engine_elements.items():
        for elem in elems:
            all_boxes.append(_TaggedBox(engine, elem))

    if not all_boxes:
        return []

    # 2. Compute median height and separate tall boxes
    heights = [b.height for b in all_boxes if b.height > 0]
    if not heights:
        return []
    med_height = median(heights)

    tall_threshold = med_height * 3.0
    normal_boxes: List[_TaggedBox] = []
    tall_boxes: List[_TaggedBox] = []
    for b in all_boxes:
        if b.height > tall_threshold:
            tall_boxes.append(b)
        else:
            normal_boxes.append(b)

    if not normal_boxes:
        # All boxes are "tall" — treat them all as normal
        normal_boxes = tall_boxes
        tall_boxes = []

    # 3. Sort by y-center and group into horizontal bands via vertical overlap
    normal_boxes.sort(key=lambda b: b.y_center)

    bands: List[List[_TaggedBox]] = []
    band_top = normal_boxes[0].top
    band_bottom = normal_boxes[0].bottom

    current_band: List[_TaggedBox] = [normal_boxes[0]]

    for box in normal_boxes[1:]:
        overlap = _vertical_overlap_ratio(band_top, band_bottom, box.top, box.bottom)
        if overlap >= 0.5:
            current_band.append(box)
            band_top = min(band_top, box.top)
            band_bottom = max(band_bottom, box.bottom)
        else:
            bands.append(current_band)
            current_band = [box]
            band_top = box.top
            band_bottom = box.bottom

    bands.append(current_band)

    # 4. Compute median width for column splitting
    widths = [b.x1 - b.x0 for b in normal_boxes if b.x1 - b.x0 > 0]
    med_width = median(widths) if widths else 20.0

    # 5. Split bands into column segments and build regions
    regions: List[ComparisonRegion] = []

    for band in bands:
        segments = _split_into_columns(band, med_width)
        for segment in segments:
            if segment:
                region = _build_region_from_segment(segment, engines, normalize, total_engines)
                # Only include regions that have text from at least one engine
                if region.texts:
                    regions.append(region)

    # 6. Map tall boxes to existing regions by overlap
    for tall in tall_boxes:
        # Find regions this tall box overlaps with
        overlapping_regions = []
        for region in regions:
            r_top = region.bbox[1]
            r_bottom = region.bbox[3]
            overlap = _vertical_overlap_ratio(tall.top, tall.bottom, r_top, r_bottom)
            if overlap > 0.1:  # any meaningful overlap
                overlapping_regions.append(region)

        if overlapping_regions:
            # Add to first overlapping region if engine not already represented there
            for region in overlapping_regions:
                if tall.engine not in region.texts:
                    region.texts[tall.engine] = tall.text
                    region.normalized_texts[tall.engine] = normalize_text(tall.text, normalize)
                    region.confidences[tall.engine] = tall.confidence
                    if tall.engine not in region.elements:
                        region.elements[tall.engine] = []
                    region.elements[tall.engine].append(tall.element)
                    # Recompute consensus and classification
                    region.consensus = compute_consensus(region.normalized_texts)
                    if region.consensus:
                        for e, nt in region.normalized_texts.items():
                            region.edit_distances[e] = _edit_distance_ratio(region.consensus, nt)
                    region.classification = classify_region(
                        region.edit_distances,
                        present_engines=len(region.texts),
                        total_engines=total_engines,
                        text_length=len(region.consensus) if region.consensus else 0,
                    )
                    region.outlier_engine = find_outlier(region.edit_distances)
                    break  # Only add to first matching region
        else:
            # Create a standalone region for this tall box
            standalone = _build_region_from_segment([tall], engines, normalize, total_engines)
            if standalone.texts:
                regions.append(standalone)

    # Sort regions by position (top-to-bottom, left-to-right)
    regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))

    return regions


# ---------------------------------------------------------------------------
# Tile-based alignment (fallback)
# ---------------------------------------------------------------------------


def align_by_tiles(
    engine_elements: Dict[str, list],
    page_bbox: Tuple[float, float, float, float],
    normalize: str = "collapse",
    tile_size: float = 80.0,
) -> List[ComparisonRegion]:
    """Tile-based alignment: divide page into grid, assign elements to tiles."""
    engines = list(engine_elements.keys())
    total_engines = len(engines)

    px0, ptop, px1, pbottom = page_bbox
    page_w = px1 - px0
    page_h = pbottom - ptop

    cols = max(1, int(page_w / tile_size))
    rows = max(1, int(page_h / tile_size))
    cell_w = page_w / cols
    cell_h = page_h / rows

    # Build tile grid
    tiles: Dict[Tuple[int, int], List[_TaggedBox]] = {}

    for engine, elems in engine_elements.items():
        for elem in elems:
            box = _TaggedBox(engine, elem)
            # Assign to tile containing the element's center
            cx = (box.x0 + box.x1) / 2.0
            cy = (box.top + box.bottom) / 2.0
            col = min(cols - 1, max(0, int((cx - px0) / cell_w)))
            row = min(rows - 1, max(0, int((cy - ptop) / cell_h)))
            key = (row, col)
            if key not in tiles:
                tiles[key] = []
            tiles[key].append(box)

    # Build regions from non-empty tiles
    regions: List[ComparisonRegion] = []
    for (row, col), boxes in sorted(tiles.items()):
        tile_x0 = px0 + col * cell_w
        tile_top = ptop + row * cell_h
        tile_x1 = tile_x0 + cell_w
        tile_bottom = tile_top + cell_h

        # Per-engine text
        texts: Dict[str, str] = {}
        confidences: Dict[str, Optional[float]] = {}
        elements_map: Dict[str, list] = {}

        for engine in engines:
            engine_boxes = sorted(
                [b for b in boxes if b.engine == engine],
                key=lambda b: (b.top, b.x0),
            )
            if engine_boxes:
                texts[engine] = " ".join(b.text for b in engine_boxes if b.text)
                confs = [b.confidence for b in engine_boxes if b.confidence is not None]
                confidences[engine] = sum(confs) / len(confs) if confs else None
                elements_map[engine] = [b.element for b in engine_boxes]

        if not texts:
            continue

        normalized_texts = {e: normalize_text(t, normalize) for e, t in texts.items()}
        consensus = compute_consensus(normalized_texts)
        edit_distances: Dict[str, float] = {}
        if consensus is not None:
            for engine, nt in normalized_texts.items():
                edit_distances[engine] = _edit_distance_ratio(consensus, nt)

        classification = classify_region(
            edit_distances,
            present_engines=len(texts),
            total_engines=total_engines,
            text_length=len(consensus) if consensus else 0,
        )
        outlier = find_outlier(edit_distances)

        regions.append(
            ComparisonRegion(
                bbox=(tile_x0, tile_top, tile_x1, tile_bottom),
                texts=texts,
                normalized_texts=normalized_texts,
                confidences=confidences,
                consensus=consensus,
                classification=classification,
                edit_distances=edit_distances,
                outlier_engine=outlier,
                elements=elements_map,
            )
        )

    return regions


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


def check_alignment_health(
    regions: List[ComparisonRegion],
    n_engines: int,
) -> Tuple[bool, Dict[str, Any]]:
    """Check whether alignment produced healthy regions.

    Returns (is_healthy, diagnostics_dict).
    """
    if not regions:
        return False, {"reason": "no_regions", "n_regions": 0}

    # Avg engines per region
    engines_per_region = [len(r.texts) for r in regions]
    avg_engines = sum(engines_per_region) / len(engines_per_region)

    # Orphan rate: regions with only 1 engine
    orphan_count = sum(1 for n in engines_per_region if n == 1)
    orphan_rate = orphan_count / len(regions) if regions else 0

    # Fill ratio: how well do element boxes fill the region bbox
    fill_ratios = []
    for r in regions:
        region_area = (r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1])
        if region_area <= 0:
            continue
        elem_area = 0
        for elems in r.elements.values():
            for elem in elems:
                w = getattr(elem, "x1", 0) - getattr(elem, "x0", 0)
                h = getattr(elem, "bottom", 0) - getattr(elem, "top", 0)
                elem_area += max(0, w) * max(0, h)
        fill_ratios.append(min(elem_area / region_area, 5.0))  # cap at 5x for overlaps

    avg_fill = sum(fill_ratios) / len(fill_ratios) if fill_ratios else 0

    # Length ratio outliers
    length_outlier_count = 0
    for r in regions:
        if len(r.texts) < 2:
            continue
        lengths = [len(t) for t in r.texts.values() if t]
        if lengths and max(lengths) > 5 * min(max(min(lengths), 1), 100):
            length_outlier_count += 1

    diagnostics = {
        "n_regions": len(regions),
        "avg_engines_per_region": round(avg_engines, 2),
        "orphan_rate": round(orphan_rate, 3),
        "avg_fill_ratio": round(avg_fill, 2),
        "length_outlier_count": length_outlier_count,
    }

    # Health decision
    is_healthy = (
        avg_engines >= 1.5
        and orphan_rate <= 0.3
        and avg_fill >= 0.05  # very low fill = merged distant things
    )

    if not is_healthy:
        reasons = []
        if avg_engines < 1.5:
            reasons.append(f"avg_engines_per_region={avg_engines:.2f} < 1.5")
        if orphan_rate > 0.3:
            reasons.append(f"orphan_rate={orphan_rate:.2f} > 0.3")
        if avg_fill < 0.05:
            reasons.append(f"avg_fill_ratio={avg_fill:.2f} < 0.05")
        diagnostics["reason"] = "; ".join(reasons)

    return is_healthy, diagnostics


# ---------------------------------------------------------------------------
# Auto alignment
# ---------------------------------------------------------------------------


def align_ocr_outputs(
    engine_elements: Dict[str, list],
    page_bbox: Tuple[float, float, float, float],
    *,
    strategy: str = "auto",
    normalize: str = "collapse",
) -> Tuple[List[ComparisonRegion], str, Dict[str, Any]]:
    """Align OCR outputs from multiple engines into comparison regions.

    Args:
        engine_elements: engine_name → list of TextElements.
        page_bbox: (x0, top, x1, bottom) of the page.
        strategy: "auto" (try rows, fallback to tiles), "rows", or "tiles".
        normalize: Text normalization mode.

    Returns:
        (regions, strategy_used, diagnostics).
    """
    n_engines = len(engine_elements)

    if strategy == "tiles":
        regions = align_by_tiles(engine_elements, page_bbox, normalize)
        return regions, "tiles", {"forced": True}

    if strategy == "rows" or strategy == "auto":
        regions = align_by_rows(engine_elements, normalize)

        if strategy == "auto" and n_engines > 1:
            is_healthy, diagnostics = check_alignment_health(regions, n_engines)
            if not is_healthy:
                logger.info(
                    "Row alignment unhealthy (%s), falling back to tiles.",
                    diagnostics.get("reason", "unknown"),
                )
                regions = align_by_tiles(engine_elements, page_bbox, normalize)
                diagnostics["fallback"] = True
                return regions, "tiles", diagnostics
            return regions, "rows", diagnostics

        # Forced "rows" or single engine
        _, diagnostics = check_alignment_health(regions, n_engines)
        return regions, "rows", diagnostics

    raise ValueError(f"Unknown strategy: {strategy!r}. Use 'auto', 'rows', or 'tiles'.")
