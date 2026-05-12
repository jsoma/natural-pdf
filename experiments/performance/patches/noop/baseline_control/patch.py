"""No-op patch used to validate experiment harness plumbing."""

from contextlib import nullcontext

METADATA = {
    "track": "control",
    "candidate": "baseline_control",
    "cache_only": False,
    "hypothesis": "Experiment patch loading should not change workload behavior.",
}


def install():
    return nullcontext()
