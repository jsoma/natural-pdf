# Page Materialization: Defer Decorations

Track: page materialization / element creation

Hypothesis: Decoration detection is useful behavior but can be expensive during
plain extraction workflows that never inspect strike, underline, or highlight
attributes. Skipping detection in a prototype estimates the upper bound of this
cost.

Primary cases: `real:multipage-table`, `real:0500000US42001`,
`real:01-practice`, `micro:page-materialize:m27`.

Risk: high for decoration-related correctness. A production version would need a
real lazy/deferred decoration design, not this blanket skip.
