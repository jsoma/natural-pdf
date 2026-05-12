# lazy_chars_raw_store

Stores prepared native char dictionaries in a separate manager field and treats
the element store as the materialized view.

Compared with the minimal field variant, this is the production-shaped version:
mutating char APIs first materialize, text-layer clearing counts deferred chars,
and invalidation clears the raw backing store.
