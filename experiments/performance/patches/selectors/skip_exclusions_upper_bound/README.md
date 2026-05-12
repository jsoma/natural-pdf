# Selectors: Skip Exclusions Upper Bound

Track: repeated selectors / exclusions

Hypothesis: Repeated exclusion filtering is a measurable cost in selector-heavy
workflows. Disabling it estimates the upper bound for a structural exclusion-mask
or batching implementation.

Primary cases: `real:Atlanta_Public_Schools_GA_sample`,
`micro:repeated-selectors:atlanta`, `warm:repeated-page:m27`,
`real:01-practice`.

Risk: intentionally high. This changes semantics and is not a production patch.
