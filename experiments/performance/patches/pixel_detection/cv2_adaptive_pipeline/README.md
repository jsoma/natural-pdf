# Pixel Detection: OpenCV Adaptive Pipeline

Hypothesis: OpenCV's C-backed grayscale, adaptive threshold, Otsu threshold, and
morphology can reduce pixel-processing time versus the scipy/numpy path.

Risk: medium. OpenCV adaptive thresholding is not exactly the same algorithm as
the current gaussian-filter local mean implementation, so line coordinates and
counts need correctness checks.
