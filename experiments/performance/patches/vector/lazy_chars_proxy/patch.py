"""Proxy-backed lazy native-char TextElement materialization."""

from experiments.performance.patches.vector._lazy_chars import install_lazy_chars_variant

METADATA = {
    "track": "vector",
    "candidate": "lazy_chars_proxy",
    "cache_only": False,
    "hypothesis": "A list-like store proxy can preserve collection shape and avoid wrapping unless iterated.",
}


def install():
    return install_lazy_chars_variant(
        mode="proxy",
        label="vector.lazy_chars_proxy",
    )
