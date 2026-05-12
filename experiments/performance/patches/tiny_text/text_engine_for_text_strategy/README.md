# Tiny Text: Natural PDF Text Engine For Text Strategy

Track: tiny text / pdfplumber table extraction

Hypothesis: When callers request pdfplumber with both strategies set to `text`,
the workload is asking pdfplumber to infer table structure from words. Natural
PDF's text table engine may avoid pdfplumber's expensive `char_in_bbox` cell
assignment on tiny glyphs.

Primary cases: `real:tiny-text-tables`, `real:guides-expenses-sample`.

Risk: high for table shape and cell text fidelity. This is a structural
comparator, not a direct production patch.
