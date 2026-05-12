# Vector: Lazy Text Elements Upper Bound

Stores prepared raw char dicts during page materialization but avoids creating
native char `TextElement` objects until `page.chars` or `get_elements("chars")`
is requested.

This is an upper-bound prototype for lazy materialization and is not a direct
production patch.
