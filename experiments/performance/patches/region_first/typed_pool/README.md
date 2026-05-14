# typed_pool

Hypothesis: `region.find_all("text")` and similar calls waste work by first
running a page-wide selector and then clipping to the region. This patch adds a
temporary `Region._get_element_pool()` and runs the selector against a
region-prefiltered pool.

This is a structural cold-path shortcut. It still honors exclusions, but if a
page has callable exclusions those costs remain.
