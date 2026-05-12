# Rendering: Render Cache

Track: rendering inside extraction workflows

Hypothesis: Some workflows render the same page or region repeatedly. A simple
per-process render cache estimates repeated-rasterization cost, but it is a
cache-only comparison and not a production recommendation by itself.

Primary cases: `real:0500000US42001`, `micro:guide-table:01-practice`, optional
render cases.

Risk: medium. Cached PIL images may increase memory and can become stale if
render inputs change.
