# Guide/Table: Skip Alt Text Scan

Track: guide/table extraction

Hypothesis: Most benchmark table regions do not contain alt-text child regions.
Skipping the alt-text scan estimates how much table extraction pays for that
capability.

Primary cases: `real:guides-expenses-sample`,
`micro:guide-table:01-practice`, `real:0500000US42001`,
`real:multipage-table`.

Risk: high for documents that rely on alt-text regions inside table cells.
