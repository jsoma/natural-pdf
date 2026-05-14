"""Run typed region selectors against a region-prefiltered pool."""

from experiments.performance.patches.region_first._common import install_region_first

METADATA = {
    "track": "region_first",
    "candidate": "typed_pool",
    "cache_only": False,
    "hypothesis": "Region.find_all should filter by region geometry before executing typed selectors over page-wide pools.",
}


def install():
    return install_region_first(candidate="typed_pool")
