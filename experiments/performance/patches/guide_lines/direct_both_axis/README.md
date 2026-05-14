# Guide Lines: Direct Both Axis

Track: guide line detection

Hypothesis: `Guides.from_lines(axis="both")` should not execute the line provider
twice. A one-pass implementation can collect vector lines once or run pixel line
detection once, then derive vertical and horizontal guide coordinates from the
same line set.

This is a structural candidate, not a cache candidate. It should help cold
single-use guide workflows where both axes are requested.

Primary cases:

- `micro:guide-lines-vector-both:01-practice`
- `micro:guide-lines-vector-both:policy-lines`
- `micro:guide-lines-pixels-both:01-practice`
- `micro:guide-table:01-practice`
