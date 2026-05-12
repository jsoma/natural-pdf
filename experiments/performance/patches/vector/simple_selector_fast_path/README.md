# Vector: Simple Selector Fast Path

Fast-paths simple selector branches such as `text`, `rect`, and boolean
decoration selectors. Complex text matching, regex/fuzzy/OCR selectors,
aggregates, relational pseudos, and post-pseudos fall back to production code.
