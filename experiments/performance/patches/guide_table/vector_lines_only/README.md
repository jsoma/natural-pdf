# Guide/Table: Vector Lines Only

Track: guide/table extraction / rendering

Hypothesis: Some guide workflows call pixel line detection, which renders pages.
For PDFs with usable vector lines, defaulting guide line detection to vector mode
avoids rasterization and isolates render-driven overhead.

Primary cases: `micro:guide-table:01-practice`, `real:0500000US42001`,
`real:multipage-table`.

Risk: high for scanned or raster-only PDFs. This is a no-render comparator, not a
universal default.
