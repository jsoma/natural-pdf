"""Use direct top-k local maxima selection for capped pixel guide lines."""

from experiments.performance.patches.pixel_detection._common import (
    find_lines_fast_topk,
    install_find_lines_patch,
)

METADATA = {
    "track": "pixel_detection",
    "candidate": "fast_topk_peaks",
    "cache_only": False,
    "hypothesis": "When max_lines is set, direct profile top-k plus NMS may avoid scipy peak/prominence overhead.",
}


def install():
    return install_find_lines_patch(find_lines_fast_topk)
