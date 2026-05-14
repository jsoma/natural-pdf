"""Run only text/word region selectors against a region-prefiltered pool."""

from experiments.performance.patches.region_first._common import install_region_first

METADATA = {
    "track": "region_first",
    "candidate": "word_pool_only",
    "cache_only": False,
    "hypothesis": "The safest region-first production slice may be text-only, because most region navigation extracts text.",
}


def install():
    return install_region_first(candidate="word_pool_only", allowed_types={"text"})
