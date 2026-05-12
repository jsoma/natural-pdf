# Tiny Text: Prefilter Pdfplumber Chars

Track: tiny text / pdfplumber table extraction

Hypothesis: Tiny-text table extraction spends most time assigning millions of
chars to candidate cells. Filtering pdfplumber objects to the requested region
before extraction may reduce broad char scans without changing public behavior.

Primary cases: `real:tiny-text-tables`, `micro:tiny-text-layout`,
`real:guides-expenses-sample`.

Risk: medium. This patch changes the pdfplumber page object passed to table
extraction and must be checked against table correctness.
