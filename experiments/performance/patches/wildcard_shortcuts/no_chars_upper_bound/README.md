# no_chars_upper_bound

Hypothesis: broad `any` element paths can accidentally force native char wrapper
materialization even when the user-facing operation only needs words, rects,
lines, images, and regions.

This patch is intentionally marked as an unsafe upper bound because it changes
wildcard semantics: `page.get_elements()` and `find_all("*")`-style paths no
longer include native char elements. It is useful only to quantify whether this
shortcut is worth designing as an explicit internal-only pool later.
