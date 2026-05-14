"""Lower default guide pixel line detection resolution to 120 DPI."""

from experiments.performance.patches.pixel_detection._common import install_resolution_patch

METADATA = {
    "track": "pixel_detection",
    "candidate": "resolution_120",
    "cache_only": False,
    "hypothesis": "120 DPI tests the aggressive lower-resolution speed/quality tradeoff for guide line detection.",
}


def install():
    return install_resolution_patch(120)
