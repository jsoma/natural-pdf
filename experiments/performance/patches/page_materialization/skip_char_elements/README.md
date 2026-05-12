# Page Materialization: Skip Char Elements

Track: page materialization / element creation

Hypothesis: Many non-OCR extraction workflows need words, rects, lines, and
images, but do not need every native char wrapped as a `TextElement`. Skipping
char-element wrapping estimates the structural cost of eager char elements.

Primary cases: `real:multipage-table`, `real:0500000US42001`,
`real:01-practice`, `micro:page-materialize:m27`.

Risk: high for APIs that use `page.chars` or char-level selectors. A production
version would need lazy char element creation.
