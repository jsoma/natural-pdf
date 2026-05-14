"""Lower default guide pixel line detection resolution to 150 DPI."""

from experiments.performance.patches.pixel_detection._common import install_resolution_patch

METADATA = {
    "track": "pixel_detection",
    "candidate": "resolution_150",
    "cache_only": False,
    "hypothesis": "150 DPI may preserve guide line quality while reducing render and pixel processing cost.",
}


def install():
    return install_resolution_patch(150)
