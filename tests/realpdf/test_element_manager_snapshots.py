from __future__ import annotations

import json

import pytest

from tests.realpdf.snapshot_utils import (
    DEFAULT_SNAPSHOT_PATH,
    REAL_PDF_CASES,
    capture_case,
    load_snapshots,
)

pytestmark = pytest.mark.realpdf


@pytest.mark.parametrize("case", REAL_PDF_CASES, ids=lambda case: case.name)
def test_element_manager_matches_snapshot(case):
    snapshots = load_snapshots(DEFAULT_SNAPSHOT_PATH)
    assert case.name in snapshots, f"Snapshot missing for {case.name}"
    expected = snapshots[case.name]
    actual = capture_case(case)

    # Dump mismatched payloads to help debugging without exposing entire snapshot file
    if actual != expected:
        debug_payload = {
            "expected": expected,
            "actual": actual,
        }
        raise AssertionError(
            f"Snapshot mismatch for {case.name}:\n{json.dumps(debug_payload, indent=2)[:2000]}"
        )
