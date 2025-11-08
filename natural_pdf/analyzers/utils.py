from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Iterable, List

from natural_pdf.utils.page_context import resolve_page_context

if TYPE_CHECKING:  # pragma: no cover
    from natural_pdf.elements.region import Region


def convert_to_regions(
    page: Any,
    detections: Iterable[Dict[str, Any]],
    scale_factor: float = 1.0,
) -> List["Region"]:
    """Convert detection dictionaries into ``Region`` objects."""

    from natural_pdf.elements.region import Region

    conversion_logger = logging.getLogger("natural_pdf.analyzers.layout.convert")

    detections_list = list(detections)
    conversion_logger.debug(
        "Converting %d detections to regions with scale %s", len(detections_list), scale_factor
    )

    regions: List[Region] = []

    page_obj, _ = resolve_page_context(page)

    for det in detections_list:
        x_min, y_min, x_max, y_max = det["bbox"]

        if x_min > x_max:
            x_min, x_max = x_max, x_min
        if y_min > y_max:
            y_min, y_max = y_max, y_min

        if scale_factor != 1.0:
            x_min *= scale_factor
            y_min *= scale_factor
            x_max *= scale_factor
            y_max *= scale_factor

        region = Region(page_obj, (x_min, y_min, x_max, y_max))
        region.region_type = det.get("class")
        region.confidence = det.get("confidence")
        region.normalized_type = det.get("normalized_class")
        region.source = det.get("source", "detected")
        region.model = det.get("model", "unknown")

        for key, value in det.items():
            if key not in {"bbox", "class", "confidence", "normalized_class", "source", "model"}:
                setattr(region, key, value)

        regions.append(region)

    conversion_logger.debug("Created %d region objects from detections", len(regions))
    return regions
