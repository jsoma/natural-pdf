import json

from scripts import perf_natural_pdf as perf


def test_discover_workloads_includes_real_and_micro_cases():
    workloads = perf.discover_workloads()
    names = {workload.name for workload in workloads}

    assert "real:01-practice" in names
    assert "real:m27" in names
    assert "real:tiny-text-tables" in names
    assert "real:multipage-table" in names
    assert "real:pak-ks-expenses" in names
    assert "real:policy-lines" in names
    assert "micro:tiny-text-layout" in names
    assert "micro:guide-table:01-practice" in names
    assert "micro:repeated-selectors:atlanta" in names
    assert "micro:repeated-selectors:tiny-text" in names
    assert "warm:repeated-page:m27" in names


def test_run_workloads_writes_baseline_shape(tmp_path):
    workload = perf.Workload(
        name="fake:fast",
        kind="test",
        description="Fake fast workload",
        cache_mode="warm",
        runner=lambda: {"rows": [1, 2, 3], "label": "ok"},
    )

    result = perf.run_workloads(
        [workload],
        output_dir=tmp_path,
        iterations=1,
        warmups=0,
        trace_memory=True,
        collect_counts=False,
        profile_slowest=0,
        profile_top_n=5,
        include_pytest_durations=False,
    )
    paths = perf.write_artifacts(result, tmp_path)

    baseline = json.loads((tmp_path / "baseline.json").read_text())
    assert baseline["schema_version"] == "1.0"
    assert baseline["settings"]["iterations"] == 1
    assert baseline["cases"][0]["name"] == "fake:fast"
    assert baseline["cases"][0]["metrics"]["wall_ms"]["median"] is not None
    assert baseline["cases"][0]["runs"][0]["output"]["rows"]["size"] == 3
    assert baseline["cases"][0]["runs"][0]["vector_metrics"] == {
        "counts": {},
        "timings_ms": {},
        "bytes": {},
    }
    assert baseline["cases"][0]["vector_metrics"] == {
        "counts": {},
        "timings_ms": {},
        "bytes": {},
    }
    assert paths["baseline"].endswith("baseline.json")


def test_vector_metrics_are_recorded_and_summarized(tmp_path):
    def record_vector_metrics():
        from experiments.performance import vector_metrics

        vector_metrics.count("vector.test.fast_path")
        vector_metrics.timing("vector.test.build_ms", 2.5)
        vector_metrics.byte_count("vector.test.array_bytes", 128)
        return {"ok": True}

    workload = perf.Workload(
        name="fake:vector",
        kind="test",
        description="Fake vector workload",
        cache_mode="warm",
        runner=record_vector_metrics,
    )
    result = perf.run_workloads(
        [workload],
        output_dir=tmp_path,
        iterations=1,
        warmups=0,
        trace_memory=False,
        collect_counts=False,
        profile_slowest=0,
        profile_top_n=5,
        include_pytest_durations=False,
    )
    perf.write_artifacts(result, tmp_path)

    baseline = json.loads((tmp_path / "baseline.json").read_text())
    run_metrics = baseline["cases"][0]["runs"][0]["vector_metrics"]
    summary_metrics = baseline["cases"][0]["vector_metrics"]
    assert run_metrics["counts"]["vector.test.fast_path"] == 1
    assert run_metrics["timings_ms"]["vector.test.build_ms"] == 2.5
    assert run_metrics["bytes"]["vector.test.array_bytes"] == 128
    assert summary_metrics["counts"]["vector.test.fast_path"]["total"] == 1
    assert summary_metrics["timings_ms"]["vector.test.build_ms"]["median"] == 2.5
    assert summary_metrics["bytes"]["vector.test.array_bytes"]["total"] == 128


def test_profile_output_includes_self_time_and_allocations(tmp_path):
    def allocate_rows():
        rows = [{"value": str(index)} for index in range(100)]
        return {"rows": rows}

    workload = perf.Workload(
        name="fake:profiled",
        kind="test",
        description="Fake profiled workload",
        cache_mode="warm",
        runner=allocate_rows,
    )
    result = perf.run_workloads(
        [workload],
        output_dir=tmp_path,
        iterations=1,
        warmups=0,
        trace_memory=False,
        collect_counts=False,
        profile_slowest=1,
        profile_top_n=5,
        include_pytest_durations=False,
    )
    perf.write_artifacts(result, tmp_path)

    baseline = json.loads((tmp_path / "baseline.json").read_text())
    profile_summary = (tmp_path / "profile-summary.md").read_text()
    assert baseline["profiles"][0]["top_self_time"]
    assert "allocation_hotspots" in baseline["profiles"][0]
    assert "Top self time" in profile_summary
    assert "Allocation hotspots" in profile_summary


def test_reports_include_menu_categories(tmp_path):
    workload = perf.Workload(
        name="fake:selectors",
        kind="test",
        description="Fake selector-heavy workload",
        cache_mode="warm",
        runner=lambda: {"matches": 10},
    )
    result = perf.run_workloads(
        [workload],
        output_dir=tmp_path,
        iterations=1,
        warmups=0,
        trace_memory=False,
        collect_counts=False,
        profile_slowest=0,
        profile_top_n=5,
        include_pytest_durations=False,
    )
    perf.write_artifacts(result, tmp_path)

    profile_summary = (tmp_path / "profile-summary.md").read_text()
    improvement_menu = (tmp_path / "improvement-menu.md").read_text()

    assert "Natural PDF Performance Profile Summary" in profile_summary
    assert "Natural PDF Optimization Menu" in improvement_menu
    assert "Repeated selector/exclusion scans" in improvement_menu
    assert "Spatial indexing" in improvement_menu


def test_patch_loading_records_metadata(tmp_path):
    patch_path = tmp_path / "noop_patch.py"
    patch_path.write_text("""
from contextlib import contextmanager

METADATA = {"track": "test", "candidate": "noop"}

@contextmanager
def install():
    yield
""".strip() + "\n")
    output_dir = tmp_path / "run"

    exit_code = perf.main(
        [
            "--quick",
            "--cases",
            "micro:open:01-practice",
            "--output",
            str(output_dir),
            "--patch",
            str(patch_path),
            "--experiment-label",
            "noop-smoke",
            "--no-counts",
            "--no-tracemalloc",
            "--profile-slowest",
            "0",
        ]
    )

    baseline = json.loads((output_dir / "baseline.json").read_text())
    assert exit_code == 0
    assert baseline["experiment"]["label"] == "noop-smoke"
    assert baseline["experiment"]["patches"][0]["status"] == "installed"
    assert baseline["experiment"]["patches"][0]["metadata"]["candidate"] == "noop"


def test_failed_patch_install_writes_failure_artifacts(tmp_path):
    patch_path = tmp_path / "bad_patch.py"
    patch_path.write_text("""
METADATA = {"track": "test", "candidate": "bad"}

def install():
    raise RuntimeError("boom")
""".strip() + "\n")
    output_dir = tmp_path / "run"

    exit_code = perf.main(
        [
            "--quick",
            "--cases",
            "micro:open:01-practice",
            "--output",
            str(output_dir),
            "--patch",
            str(patch_path),
            "--experiment-label",
            "bad-smoke",
            "--no-counts",
            "--no-tracemalloc",
            "--profile-slowest",
            "0",
        ]
    )

    baseline = json.loads((output_dir / "baseline.json").read_text())
    assert exit_code == 3
    assert baseline["cases"] == []
    assert baseline["experiment"]["patches"][0]["status"] == "install_error"
    assert "boom" in baseline["experiment"]["patch_errors"][0]
