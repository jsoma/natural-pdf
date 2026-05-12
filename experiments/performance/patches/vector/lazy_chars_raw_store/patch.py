"""Separate raw-char store with lazy native-char TextElement materialization."""

from experiments.performance.patches.vector._lazy_chars import install_lazy_chars_variant

METADATA = {
    "track": "vector",
    "candidate": "lazy_chars_raw_store",
    "cache_only": False,
    "hypothesis": "Keeping a distinct raw-char backing store gives a production-shaped lazy design.",
}


def install():
    return install_lazy_chars_variant(
        mode="raw_store",
        label="vector.lazy_chars_raw_store",
    )
