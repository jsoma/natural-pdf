# Selectors: Exact Query Cache

Track: repeated selectors / exclusions

Hypothesis: Some workflows repeatedly run identical selectors against the same
page. Exact-query caching should help warm repeated workloads but must be marked
as an amortized repeated-work improvement, not a cold single-use structural fix.

Primary cases: `real:Atlanta_Public_Schools_GA_sample`,
`micro:repeated-selectors:atlanta`, `warm:repeated-page:m27`.

Risk: medium. Cache invalidation is intentionally unsolved in this prototype; do
not promote directly to production.
