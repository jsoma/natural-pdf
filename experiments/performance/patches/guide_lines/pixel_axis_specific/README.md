# Guide Lines: Pixel Axis-Specific

Track: guide line detection / rendering

Hypothesis: when callers request only horizontal or only vertical guides with
pixel detection, Natural PDF should not run the opposite-axis morphology and peak
detection. This patch rewrites the `detect_lines()` flags for single-axis guide
calls only.

It intentionally leaves `axis="both"` unchanged because the current production
path relies on two sequential calls and `replace=True`; changing each call to one
axis would alter the final detected line elements on the page.

Primary cases:

- `micro:guide-lines-pixels-horizontal:01-practice`
- `micro:guide-lines-pixels-vertical:01-practice`
- `micro:guide-table:01-practice`
