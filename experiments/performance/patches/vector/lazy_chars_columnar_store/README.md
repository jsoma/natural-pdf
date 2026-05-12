# lazy_chars_columnar_store

Builds a structure-of-arrays view over prepared native char dictionaries during
page population, but still defers `TextElement` wrappers until char APIs are
requested.

This measures the upfront tax of a broader vector-backed page store. It is
expected to be slower than pure lazy variants on one-off workflows unless later
vector operations reuse the arrays enough to pay for construction.
