"""Use OpenCV thresholding and morphology for pixel line detection when available."""

from experiments.performance.patches.pixel_detection._common import (
    find_lines_cv2,
    install_find_lines_patch,
)

METADATA = {
    "track": "pixel_detection",
    "candidate": "cv2_adaptive_pipeline",
    "cache_only": False,
    "hypothesis": "OpenCV's C-backed grayscale, adaptive threshold, Otsu, and morphology may beat the scipy/numpy implementation.",
}


def install():
    return install_find_lines_patch(find_lines_cv2)
