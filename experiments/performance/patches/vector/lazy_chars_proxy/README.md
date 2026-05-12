# lazy_chars_proxy

Stores a list-like proxy in `store["chars"]` after population. `len()` is cheap,
but iteration and indexing materialize `TextElement` wrappers.

This tests whether a proxy is a cleaner compatibility layer than an empty store
slot. It is slightly riskier because existing code may assume `store["chars"]`
is a concrete list.
