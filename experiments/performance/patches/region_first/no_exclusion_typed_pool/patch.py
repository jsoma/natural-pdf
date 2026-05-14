"""Region-first selector pool only when no exclusions are present."""

from experiments.performance.patches.region_first._common import install_region_first

METADATA = {
    "track": "region_first",
    "candidate": "no_exclusion_typed_pool",
    "cache_only": False,
    "hypothesis": "A no-exclusion fast path may capture common one-shot region queries without paying callable exclusion setup.",
}


def install():
    return install_region_first(candidate="no_exclusion_typed_pool", require_no_exclusions=True)
