# lazy_chars_minimal_field

Stores prepared native char dictionaries on `ElementManager` and leaves
`store["chars"]` empty until a char-facing API is called.

This is the smallest implementation of lazy char materialization. It tests the
raw benefit of removing eager `TextElement` wrapping while preserving normal
`page.chars`, `get_elements("chars")`, and `get_all_elements()` behavior by
materializing on demand.
