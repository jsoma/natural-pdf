"""Use a leaner uint8 projection pipeline for pixel line detection."""

from experiments.performance.patches.pixel_detection._common import (
    find_lines_uint8,
    install_find_lines_patch,
)

METADATA = {
    "track": "pixel_detection",
    "candidate": "uint8_profile_pipeline",
    "cache_only": False,
    "hypothesis": "Avoid float-normalized binary image copies and use integer/bool profiles.",
}


def install():
    return install_find_lines_patch(find_lines_uint8)
