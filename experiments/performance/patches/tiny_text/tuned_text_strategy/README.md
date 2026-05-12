# Tiny Text: Tuned Text Strategy

Track: tiny text / pdfplumber table extraction

Hypothesis: Tiny-text tables may benefit from stricter pdfplumber text-strategy
settings that reduce candidate line/cell combinations. This is a comparator, not
a preferred structural fix.

Primary cases: `real:tiny-text-tables`, `real:guides-expenses-sample`.

Risk: high for correctness. Treat wins as workload-specific unless tests prove
the settings generalize.
