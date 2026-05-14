# find_aggregate_first

Hypothesis: selectors such as `text[size=max()]` are common anchor lookups and
can avoid full result collection when the caller only needs `find()`.

This patch scans text elements once, tracks the max/min aggregate value, and
returns the first reading-order winner. It falls back for exclusions, custom
engines, text tolerance overrides, and complex selector syntax.

This is a cold-path structural shortcut, not a cache.
