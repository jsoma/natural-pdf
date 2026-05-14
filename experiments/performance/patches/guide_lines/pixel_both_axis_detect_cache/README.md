# Guide Lines: Pixel Both-Axis Detect Cache

Track: guide line detection / rendering

Hypothesis: `Guides.from_lines(axis="both", detection_method="pixels")` currently
can call pixel line detection once per axis. A short-lived cache around one guide
operation estimates the upside of removing duplicate renders/detection without
changing guide coordinate logic.

This is intentionally labelled cache-only. It is useful as a measurement
comparator, but the production fix should prefer a structural one-pass path if
the result is strong.

Primary case: `micro:guide-lines-pixels-both:01-practice`.
