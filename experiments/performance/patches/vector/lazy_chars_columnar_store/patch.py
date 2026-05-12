"""Columnar raw-char store with lazy native-char TextElement materialization."""

from experiments.performance.patches.vector._lazy_chars import install_lazy_chars_variant

METADATA = {
    "track": "vector",
    "candidate": "lazy_chars_columnar_store",
    "cache_only": False,
    "hypothesis": "Building a structure-of-arrays char view during load may enable broader vector paths while still deferring wrappers.",
}


def install():
    return install_lazy_chars_variant(
        mode="columnar_store",
        label="vector.lazy_chars_columnar_store",
    )
