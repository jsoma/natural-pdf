# Guide/Table: Default Word Cells

Track: guide/table extraction

Hypothesis: Recent guide-table work added fast batched word assignment for
indexed table cells. Guide-built tables that do not use OCR or custom callbacks
should often use `cell_extract="words"` by default to avoid slower per-cell text
extraction.

Primary cases: `real:0500000US42001`, `micro:guide-table:01-practice`,
`real:multipage-table`.

Risk: medium. Cell text spacing/newline behavior can differ from existing
`cell_extract="text"` output.
