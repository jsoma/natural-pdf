# no_exclusion_typed_pool

Hypothesis: most page workflows have no exclusions, so a production fast path can
be narrower and simpler by falling back when page/PDF exclusions exist.

This patch is otherwise the same idea as `typed_pool`: run the selector against a
region-prefiltered element pool. It is expected to show less benefit on Atlanta,
which intentionally has PDF-level exclusions.
