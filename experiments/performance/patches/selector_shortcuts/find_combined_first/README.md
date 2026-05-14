# find_combined_first

Hypothesis: real extraction configs often mix `text:contains(...)` anchors and
`text[size=max()]` title lookups, so a combined first-match shortcut may beat
either specialized patch alone.

This patch includes the `find_contains_first` and `find_aggregate_first`
behaviors. It remains a structural shortcut and does not cache query results.
