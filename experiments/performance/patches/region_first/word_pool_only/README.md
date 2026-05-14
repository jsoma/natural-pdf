# word_pool_only

Hypothesis: a text-only region-first path captures most user-facing region
queries while avoiding broader selector surface area.

This patch only changes `Region.find_all()` for `text`/`word` selectors and
falls back for rects, lines, images, regions, wildcard selectors, and OR queries
that include unsupported types.
