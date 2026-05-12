"""Minimal lazy native-char TextElement materialization."""

from experiments.performance.patches.vector._lazy_chars import install_lazy_chars_variant

METADATA = {
    "track": "vector",
    "candidate": "lazy_chars_minimal_field",
    "cache_only": False,
    "hypothesis": "A manager-side raw-char field can remove eager char wrapping with little machinery.",
}


def install():
    return install_lazy_chars_variant(
        mode="minimal",
        label="vector.lazy_chars_minimal_field",
    )
