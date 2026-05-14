# Guide Lines: Vector Single-Pass Cache

Track: guide line detection / vector geometry

Hypothesis: some vector branch cost may come from collecting the same line
elements twice when both axes are requested. This patch caches only line
collection for the duration of one both-axis guide operation.

This is a cache comparator. If it helps meaningfully, the structural follow-up
would be the direct both-axis path.

Primary cases:

- `micro:guide-lines-vector-both:01-practice`
- `micro:guide-lines-vector-both:policy-lines`
