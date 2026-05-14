# Guide Lines: Vector NumPy Classify

Track: guide line detection / vector geometry

Hypothesis: vector guide detection is location-heavy enough that array-based
classification and top-k selection may beat repeated Python property checks on
line-heavy pages.

This patch only affects vector or auto-vector branches. Pixel detection falls
back to current behavior.

Primary cases:

- `micro:guide-lines-vector-both:01-practice`
- `micro:guide-lines-vector-both:policy-lines`
- `real:policy-lines`
