#!/usr/bin/env python3
"""Developer performance harness for non-AI, non-OCR Natural PDF flows.

This script is intentionally outside the public natural_pdf API. It measures real
PDF workflows and focused micro workloads, then writes artifacts that help decide
where optimization time should go next.

Typical usage:

    uv run python scripts/perf_natural_pdf.py --output /tmp/npdf-perf --quick
    uv run python scripts/perf_natural_pdf.py --list-cases
"""

from __future__ import annotations

import argparse
import cProfile
import importlib.util
import json
import math
import os
import platform
import pstats
import statistics
import subprocess
import sys
import time
import tracemalloc
from collections import Counter, defaultdict
from contextlib import AbstractContextManager, ExitStack, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_OUTPUT_DIR = REPO_ROOT / "perf_output"
DEFAULT_PROFILE_TOP_N = 25
DEFAULT_PROFILE_SLOWEST = 4

NON_AI_PYTEST_TARGETS = [
    "tests/test_selector_expressions.py",
    "tests/test_guides_integration.py",
    "tests/test_tables_integration.py",
    "tests/test_words_vs_find_all_text.py",
    "tests/test_arabic_performance.py",
    "tests/test_tiny_text_tables.py",
    "tests/test_tiny_text_tables_table.py",
    "tests/test_multipage_directional.py",
]

NON_AI_MARKER_EXPR = (
    "not tutorial and not qa and not qa_remote and not qa_local and not optional_deps "
    "and not ocr and not network and not slow"
)


@dataclass(frozen=True)
class Workload:
    """A benchmarkable unit of work."""

    name: str
    kind: str
    description: str
    cache_mode: str
    runner: Callable[[], Mapping[str, Any]]
    pdf_path: Optional[Path] = None
    count_pages: int = 1
    available: bool = True
    skip_reason: Optional[str] = None


class ExperimentPatchManager(AbstractContextManager["ExperimentPatchManager"]):
    """Load experiment-only monkey patches for the lifetime of a perf run."""

    def __init__(self, patch_paths: Iterable[Path]) -> None:
        self.patch_paths = [Path(path) for path in patch_paths]
        self.records: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._stack = ExitStack()

    def __enter__(self) -> "ExperimentPatchManager":
        self._stack.__enter__()
        for patch_path in self.patch_paths:
            self._install_patch(patch_path)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stack.__exit__(exc_type, exc, tb)

    def _install_patch(self, patch_path: Path) -> None:
        resolved_path = patch_path.expanduser().resolve()
        record: dict[str, Any] = {
            "path": _relative_path(resolved_path),
            "status": "pending",
            "metadata": {},
        }
        self.records.append(record)

        try:
            if not resolved_path.exists():
                raise FileNotFoundError(f"Patch module does not exist: {resolved_path}")
            module_name = (
                f"_npdf_perf_patch_{_safe_filename(str(resolved_path))}_{len(self.records)}"
            )
            spec = importlib.util.spec_from_file_location(module_name, resolved_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load module spec for {resolved_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            metadata_value = getattr(module, "METADATA", {})
            if isinstance(metadata_value, Mapping):
                record["metadata"] = dict(metadata_value)
            else:
                record["metadata"] = {"value": repr(metadata_value)}
            record["module_name"] = module_name
            record["doc"] = (getattr(module, "__doc__", "") or "").strip().splitlines()[:1]

            install = getattr(module, "install", None)
            if not callable(install):
                raise AttributeError("Patch module must define callable install()")

            context_manager = install()
            if context_manager is None:
                context_manager = nullcontext()
            if not hasattr(context_manager, "__enter__") or not hasattr(
                context_manager, "__exit__"
            ):
                raise TypeError("install() must return a context manager or None")

            self._stack.enter_context(context_manager)
            record["status"] = "installed"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            record["status"] = "install_error"
            record["error"] = error
            self.errors.append(f"{record['path']}: {error}")


class PerfInstrumentation(AbstractContextManager["PerfInstrumentation"]):
    """Monkeypatch selected internal methods to collect call counts and phase timings."""

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.phase_ms: defaultdict[str, float] = defaultdict(float)
        self._patches: list[tuple[Any, str, Any]] = []

    def __enter__(self) -> "PerfInstrumentation":
        self._install_patches()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for owner, attr, original in reversed(self._patches):
            setattr(owner, attr, original)
        self._patches.clear()

    def _patch(self, owner: Any, attr: str, label: str) -> None:
        original = getattr(owner, attr, None)
        if original is None:
            return

        instrumentation = self

        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                instrumentation.counts[label] += 1
                instrumentation.phase_ms[label] += elapsed_ms

        self._patches.append((owner, attr, original))
        setattr(owner, attr, wrapper)

    def _install_patches(self) -> None:
        try:
            from natural_pdf.core import pdf as pdf_module
            from natural_pdf.core.element_manager import ElementManager
            from natural_pdf.core.page import Page
            from natural_pdf.core.word_engine import WordEngine
            from natural_pdf.elements.region import Region
            from natural_pdf.services import selector_service as selector_service_module
            from natural_pdf.services.rendering_service import RenderingService
            from natural_pdf.services.selector_service import SelectorService
            from natural_pdf.services.table_service import TableService
            from natural_pdf.services.to_llm_service import ToLLMService
        except Exception:
            return

        self._patch(pdf_module.PDF, "__init__", "pdf_init")
        self._patch(pdf_module._LazyPageList, "_create_page", "page_create")
        self._patch(ElementManager, "load_elements", "element_load")
        self._patch(ElementManager, "_populate_store", "element_populate")
        self._patch(WordEngine, "generate_words", "word_generate")
        self._patch(Page, "_filter_elements_by_exclusions", "exclusion_filter")
        self._patch(Region, "_filter_elements_by_overlap_mode", "region_overlap_filter")
        self._patch(TableService, "extract_table", "table_extract")
        self._patch(RenderingService, "render", "render")
        self._patch(ToLLMService, "to_llm", "to_llm")
        self._patch(SelectorService, "find_all", "find_all")
        self._patch(selector_service_module, "execute_selector_query", "selector_query")


def _reset_vector_metrics() -> None:
    try:
        from experiments.performance import vector_metrics

        vector_metrics.reset()
    except Exception:
        return


def _snapshot_vector_metrics() -> dict[str, dict[str, float]]:
    try:
        from experiments.performance import vector_metrics

        return vector_metrics.snapshot()
    except Exception:
        return {"counts": {}, "timings_ms": {}, "bytes": {}}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return repr(value)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_subprocess(args: list[str], *, timeout: Optional[int] = None) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    try:
        completed = subprocess.run(
            args,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "args": args}
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "args": args,
    }


def get_git_info() -> dict[str, Any]:
    sha_result = _run_subprocess(["git", "rev-parse", "HEAD"])
    status_result = _run_subprocess(["git", "status", "--short"])
    sha = sha_result.get("stdout", "").strip() if sha_result.get("ok") else None
    status_lines = [line for line in status_result.get("stdout", "").splitlines() if line.strip()]
    return {
        "sha": sha,
        "dirty": bool(status_lines),
        "status_short": status_lines,
        "status_error": status_result.get("error") if not status_result.get("ok") else None,
    }


def get_environment_info() -> dict[str, Any]:
    packages = {}
    for package_name in [
        "natural-pdf",
        "pdfplumber",
        "numpy",
        "pandas",
        "scipy",
        "pillow",
        "pydantic",
        "pytest",
    ]:
        try:
            packages[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            packages[package_name] = None
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": packages,
    }


def _current_rss_kib() -> Optional[float]:
    try:
        import resource
    except Exception:
        return None
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss = float(usage.ru_maxrss)
    except Exception:
        return None
    if sys.platform == "darwin":
        return rss / 1024.0
    return rss


def _summary(values: Iterable[Optional[float]]) -> dict[str, Optional[float]]:
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return {"min": None, "median": None, "p95": None, "max": None}
    p95_index = max(0, min(len(clean) - 1, math.ceil(len(clean) * 0.95) - 1))
    return {
        "min": clean[0],
        "median": statistics.median(clean),
        "p95": clean[p95_index],
        "max": clean[-1],
    }


def _summarize_vector_metrics(runs: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, list[float]]] = {
        "counts": defaultdict(list),
        "timings_ms": defaultdict(list),
        "bytes": defaultdict(list),
    }
    for run in runs:
        vector_metrics = run.get("vector_metrics") or {}
        for group_name in groups:
            for label, value in (vector_metrics.get(group_name) or {}).items():
                try:
                    groups[group_name][label].append(float(value))
                except (TypeError, ValueError):
                    continue

    summarized: dict[str, Any] = {}
    for group_name, grouped_values in groups.items():
        summarized[group_name] = {
            label: _summary(values) | {"total": float(sum(values))}
            for label, values in sorted(grouped_values.items())
        }
    return summarized


def _summarize_counters(runs: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    labels = sorted({label for run in runs for label in run.get(key, {})})
    summary = {}
    for label in labels:
        values = [run.get(key, {}).get(label, 0) for run in runs]
        numeric_values = [float(value) for value in values]
        summary[label] = {
            "min": min(numeric_values),
            "median": statistics.median(numeric_values),
            "max": max(numeric_values),
            "total": sum(numeric_values),
        }
    return summary


def _output_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, (str, int, float, bool)) or item is None:
            summary[key] = item
        elif isinstance(item, Mapping):
            summary[key] = {"type": "mapping", "size": len(item)}
        elif isinstance(item, (list, tuple)):
            summary[key] = {"type": type(item).__name__, "size": len(item)}
        else:
            shape = getattr(item, "shape", None)
            if shape is not None:
                summary[key] = {"type": type(item).__name__, "shape": list(shape)}
            else:
                summary[key] = {"type": type(item).__name__}
    return summary


def _relative_path(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _relative_pdf_path(path: Optional[Path]) -> Optional[str]:
    return _relative_path(path)


def collect_pdf_counts(pdf_path: Optional[Path], max_pages: int = 1) -> dict[str, Any]:
    if pdf_path is None:
        return {}
    if not pdf_path.exists():
        return {"missing": True}

    import natural_pdf as npdf

    counts = {
        "pdf_path": _relative_pdf_path(pdf_path),
        "pages_total": None,
        "pages_counted": 0,
        "chars": 0,
        "words": 0,
        "rects": 0,
        "lines": 0,
        "images": 0,
    }
    pdf = npdf.PDF(str(pdf_path))
    try:
        counts["pages_total"] = len(pdf.pages)
        pages_to_count = min(max_pages, len(pdf.pages))
        for page_index in range(pages_to_count):
            page = pdf.pages[page_index]
            counts["pages_counted"] += 1
            counts["chars"] += len(page.chars)
            counts["words"] += len(page.words)
            counts["rects"] += len(page.rects)
            counts["lines"] += len(page.lines)
            counts["images"] += len(page.images)
    finally:
        pdf.close()
    return counts


def _open_pdf_result(pdf_path: Path) -> Mapping[str, Any]:
    import natural_pdf as npdf

    pdf = npdf.PDF(str(pdf_path))
    try:
        return {"pages": len(pdf.pages)}
    finally:
        pdf.close()


def _materialize_page_result(pdf_path: Path, page_index: int = 0) -> Mapping[str, Any]:
    import natural_pdf as npdf

    pdf = npdf.PDF(str(pdf_path))
    try:
        page = pdf.pages[page_index]
        return {
            "page_number": page.number,
            "width": page.width,
            "height": page.height,
            "words": len(page.words),
        }
    finally:
        pdf.close()


def _materialize_chars_result(pdf_path: Path, page_index: int = 0) -> Mapping[str, Any]:
    import natural_pdf as npdf

    pdf = npdf.PDF(str(pdf_path))
    try:
        page = pdf.pages[page_index]
        chars = page.chars
        first_text = chars[0].text if chars else ""
        last_text = chars[-1].text if chars else ""
        return {
            "page_number": page.number,
            "chars": len(chars),
            "first_text": first_text,
            "last_text": last_text,
        }
    finally:
        pdf.close()


def _find_all_chars_result(pdf_path: Path, page_index: int = 0) -> Mapping[str, Any]:
    import natural_pdf as npdf

    pdf = npdf.PDF(str(pdf_path))
    try:
        page = pdf.pages[page_index]
        chars = page.find_all("char")
        return {"matches": len(chars)}
    finally:
        pdf.close()


def _get_elements_result(pdf_path: Path, page_index: int = 0) -> Mapping[str, Any]:
    import natural_pdf as npdf

    pdf = npdf.PDF(str(pdf_path))
    try:
        page = pdf.pages[page_index]
        elements = page.get_elements()
        return {"matches": len(elements)}
    finally:
        pdf.close()


def _find_all_any_result(pdf_path: Path, page_index: int = 0) -> Mapping[str, Any]:
    import natural_pdf as npdf

    pdf = npdf.PDF(str(pdf_path))
    try:
        page = pdf.pages[page_index]
        elements = page.find_all("*")
        return {"matches": len(elements)}
    finally:
        pdf.close()


def _summary_output(summary: Any) -> dict[str, Any]:
    try:
        data = summary.to_dict()
    except Exception:
        data = {}
    return {
        "sections": len(data) if isinstance(data, dict) else 0,
        "type": type(summary).__name__,
    }


def _describe_page_result(pdf_path: Path, page_index: int = 0) -> Mapping[str, Any]:
    import natural_pdf as npdf

    pdf = npdf.PDF(str(pdf_path))
    try:
        return _summary_output(pdf.pages[page_index].describe())
    finally:
        pdf.close()


def _inspect_page_result(pdf_path: Path, page_index: int = 0) -> Mapping[str, Any]:
    import natural_pdf as npdf

    pdf = npdf.PDF(str(pdf_path))
    try:
        return _summary_output(pdf.pages[page_index].inspect())
    finally:
        pdf.close()


def _extract_text_result(pdf_path: Path, *, layout: bool, page_index: int = 0) -> Mapping[str, Any]:
    import natural_pdf as npdf

    pdf = npdf.PDF(str(pdf_path))
    try:
        text = pdf.pages[page_index].extract_text(layout=layout)
        return {"characters": len(text), "lines": text.count("\n") + 1 if text else 0}
    finally:
        pdf.close()


def _find_anchor_result(
    pdf_path: Path,
    *,
    selectors: list[str],
    page_index: int = 0,
) -> Mapping[str, Any]:
    import natural_pdf as npdf

    pdf = npdf.PDF(str(pdf_path))
    try:
        page = pdf.pages[page_index]
        found = 0
        text_chars = 0
        for selector in selectors:
            element = page.find(selector)
            if element:
                found += 1
                text_chars += len(getattr(element, "text", "") or "")
        return {"selector_calls": len(selectors), "found": found, "text_chars": text_chars}
    finally:
        pdf.close()


def _repeated_selector_result(pdf_path: Path, *, page_index: int = 0) -> Mapping[str, Any]:
    import natural_pdf as npdf

    selectors = [
        "text",
        "text[size=max()]",
        "text:contains(a)",
        "rect",
        "line",
        "text[x0>0]",
    ]
    pdf = npdf.PDF(str(pdf_path))
    try:
        page = pdf.pages[page_index]
        total_matches = 0
        for _ in range(8):
            for selector in selectors:
                total_matches += len(page.find_all(selector))
        return {"selector_calls": 8 * len(selectors), "matches": total_matches}
    finally:
        pdf.close()


def _region_navigation_result(pdf_path: Path, *, page_index: int = 0) -> Mapping[str, Any]:
    import natural_pdf as npdf

    anchors = [
        "text:contains(LICENSEE NAME)",
        "text:contains(Date submitted)",
        "text:contains(Summary)",
        "text[size=max()]",
    ]
    pdf = npdf.PDF(str(pdf_path))
    try:
        page = pdf.pages[page_index]
        anchor = None
        for selector in anchors:
            anchor = page.find(selector)
            if anchor:
                break
        if not anchor:
            return {"anchor_found": False, "matches": 0}
        region = anchor.below(until="text:contains(TOTAL)", include_endpoint=False)
        matches = region.find_all("text", overlap="partial") if region else []
        return {"anchor_found": True, "matches": len(matches)}
    finally:
        pdf.close()


def _dissolve_result(pdf_path: Path, *, page_index: int = 0) -> Mapping[str, Any]:
    import natural_pdf as npdf

    pdf = npdf.PDF(str(pdf_path))
    try:
        page = pdf.pages[page_index]
        text = page.find_all("text")
        dissolved = text.dissolve(padding=2.0)
        return {"input": len(text), "regions": len(dissolved)}
    finally:
        pdf.close()


def _merge_connected_result(pdf_path: Path, *, page_index: int = 0) -> Mapping[str, Any]:
    import natural_pdf as npdf

    pdf = npdf.PDF(str(pdf_path))
    try:
        page = pdf.pages[page_index]
        regions = page.find_all("text").dissolve(padding=1.0)
        merged = regions.merge_connected(proximity_threshold=2.0)
        return {"input": len(regions), "merged": len(merged)}
    finally:
        pdf.close()


def _render_result(pdf_path: Path, *, page_index: int = 0) -> Mapping[str, Any]:
    import natural_pdf as npdf

    pdf = npdf.PDF(str(pdf_path))
    try:
        image = pdf.pages[page_index].render(resolution=72)
        if image is None:
            return {"rendered": False}
        return {"rendered": True, "width": image.width, "height": image.height}
    finally:
        pdf.close()


def _to_llm_result(pdf_path: Path, *, page_index: int = 0) -> Mapping[str, Any]:
    import natural_pdf as npdf

    pdf = npdf.PDF(str(pdf_path))
    try:
        output = pdf.pages[page_index].to_llm(detail="brief")
        return {"characters": len(output)}
    finally:
        pdf.close()


def _table_shape(table: Any) -> dict[str, Any]:
    rows = 0
    columns = 0
    try:
        rows = len(table)
    except Exception:
        rows = 0
    try:
        columns = max(len(row) for row in table) if rows else 0
    except Exception:
        columns = 0
    return {"rows": rows, "columns": columns}


def _tiny_text_workflow_result(pdf_path: Path) -> Mapping[str, Any]:
    import natural_pdf as npdf

    dials = [
        {"y_tolerance": 1, "x_tolerance": 0.5},
        {"y_tolerance": 1, "x_tolerance": 0.3},
    ]
    output: dict[str, Any] = {"dial_text_chars": [], "dial_text_lines": []}

    for dial in dials:
        pdf = npdf.PDF(str(pdf_path), text_tolerance=dial, auto_text_tolerance=False)
        try:
            text = pdf.pages[0].extract_text()
            output["dial_text_chars"].append(len(text))
            output["dial_text_lines"].append(text.count("\n") + 1 if text else 0)
        finally:
            pdf.close()

    pdf = npdf.PDF(str(pdf_path))
    try:
        page = pdf.pages[0]
        region_text = page.create_region(50, 55, 200, 62).extract_text()
        table_settings = {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "snap_tolerance": 1,
            "join_tolerance": 1,
        }
        x_tolerance = page._config.get("x_tolerance")
        y_tolerance = page._config.get("y_tolerance")
        if x_tolerance is not None:
            table_settings["text_x_tolerance"] = x_tolerance
        if y_tolerance is not None:
            table_settings["text_y_tolerance"] = y_tolerance
        table = page.extract_table(method="pdfplumber", table_settings=table_settings)
        output.update(
            {
                "region_chars": len(region_text),
                "table": _table_shape(table),
            }
        )
        return output
    finally:
        pdf.close()


def _multipage_table_workflow_result(pdf_path: Path, *, max_pages: int = 3) -> Mapping[str, Any]:
    import natural_pdf as npdf

    pdf = npdf.PDF(str(pdf_path))
    try:
        pages_to_visit = min(max_pages, len(pdf.pages))
        text_chars = 0
        text_elements = 0
        words = 0
        table_rows = 0
        table_errors: list[str] = []
        for page_index in range(pages_to_visit):
            page = pdf.pages[page_index]
            text = page.extract_text(layout=False) or ""
            text_chars += len(text)
            text_matches = page.find_all("text")
            text_elements += len(text_matches)
            words += len(page.words)
            try:
                table = page.extract_table(method="pdfplumber")
                table_rows += _table_shape(table)["rows"]
            except Exception as exc:
                table_errors.append(f"page {page_index + 1}: {type(exc).__name__}: {exc}")
        return {
            "pages": pages_to_visit,
            "text_chars": text_chars,
            "text_elements": text_elements,
            "words": words,
            "table_rows": table_rows,
            "table_errors": table_errors,
        }
    finally:
        pdf.close()


def _pak_expenses_workflow_result(pdf_path: Path, *, page_index: int = 0) -> Mapping[str, Any]:
    import natural_pdf as npdf

    pdf = npdf.PDF(str(pdf_path))
    try:
        page = pdf.pages[page_index]
        words = len(page.words)
        rects = len(page.rects)
        selector_matches = len(page.find_all("text")) + len(page.find_all("rect"))
        table_rows = 0
        table_errors: list[str] = []
        try:
            table = page.extract_table(method="pdfplumber")
            table_rows = _table_shape(table)["rows"]
        except Exception as exc:
            table_errors.append(f"{type(exc).__name__}: {exc}")
        return {
            "words": words,
            "rects": rects,
            "selector_matches": selector_matches,
            "table_rows": table_rows,
            "table_errors": table_errors,
        }
    finally:
        pdf.close()


def _policy_lines_workflow_result(pdf_path: Path, *, page_index: int = 0) -> Mapping[str, Any]:
    import natural_pdf as npdf

    pdf = npdf.PDF(str(pdf_path))
    try:
        page = pdf.pages[page_index]
        text = page.extract_text(layout=False) or ""
        text_matches = page.find_all("text")
        line_matches = page.find_all("line")
        title = page.find("text[size=max()]")
        below_matches = []
        if title:
            below = title.below(height=150)
            below_matches = below.find_all("text", overlap="partial")
        return {
            "text_chars": len(text),
            "text_matches": len(text_matches),
            "line_matches": len(line_matches),
            "below_matches": len(below_matches),
        }
    finally:
        pdf.close()


def _guide_table_result(pdf_path: Path, *, page_index: int = 0) -> Mapping[str, Any]:
    import natural_pdf as npdf
    from natural_pdf.analyzers.guides import Guides

    pdf = npdf.PDF(str(pdf_path))
    try:
        page = pdf.pages[page_index]
        text_elements = page.find_all("text")
        guides = Guides(page)
        headers = text_elements[:5]
        if len(headers) >= 3:
            guides.vertical.from_headers(headers)
        else:
            guides.vertical.from_content(text_elements[:4])
        guides.horizontal.from_lines(n=5)
        table = guides.extract_table()
        return {
            "vertical_guides": len(guides.vertical),
            "horizontal_guides": len(guides.horizontal),
            "table": _table_shape(table),
        }
    finally:
        pdf.close()


def _guide_lines_result(
    pdf_path: Path,
    *,
    axis: str,
    detection_method: str,
    page_index: int = 0,
    max_lines_h: Optional[int] = 5,
    max_lines_v: Optional[int] = 5,
) -> Mapping[str, Any]:
    import warnings

    import natural_pdf as npdf
    from natural_pdf.analyzers.guides import Guides

    pdf = npdf.PDF(str(pdf_path))
    try:
        page = pdf.pages[page_index]
        before_lines = len(page.lines)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            guides = Guides.from_lines(
                page,
                axis=axis,  # type: ignore[arg-type]
                detection_method=detection_method,
                max_lines_h=max_lines_h,
                max_lines_v=max_lines_v,
            )
        detected_lines = [
            line for line in page.lines if getattr(line, "source", None) == "guides_detection"
        ]
        return {
            "axis": axis,
            "detection_method": detection_method,
            "lines_before": before_lines,
            "lines_after": len(page.lines),
            "detected_lines": len(detected_lines),
            "vertical_guides": len(guides.vertical),
            "horizontal_guides": len(guides.horizontal),
        }
    finally:
        pdf.close()


def _warm_repeated_page_result(pdf_path: Path, *, page_index: int = 0) -> Mapping[str, Any]:
    import natural_pdf as npdf

    pdf = npdf.PDF(str(pdf_path))
    try:
        page = pdf.pages[page_index]
        words = len(page.words)
        text_elements = len(page.find_all("text"))
        total_chars = 0
        selector_matches = 0
        for _ in range(12):
            total_chars += len(page.extract_text(layout=False) or "")
            selector_matches += len(page.find_all("text[size=max()]"))
            selector_matches += len(page.find_all("text:contains(a)"))
        return {
            "warm_iterations": 12,
            "words": words,
            "text_elements": text_elements,
            "text_chars": total_chars,
            "selector_matches": selector_matches,
        }
    finally:
        pdf.close()


def _config_extraction_result(config_name: str, pages: int) -> Mapping[str, Any]:
    from benchmark.configs import get_config

    config_class = get_config(config_name)
    if config_class is None:
        raise ValueError(f"Unknown benchmark config: {config_name}")
    config = config_class()
    rows = 0
    result_types: Counter[str] = Counter()
    for page_index in range(pages):
        result = config.extract_with_natural_pdf(config.pdf_path, page_index)
        result_types[type(result).__name__] += 1
        if hasattr(result, "shape"):
            rows += int(result.shape[0])
        elif isinstance(result, list):
            rows += len(result)
        elif isinstance(result, dict):
            rows += 1
    return {"pages": pages, "rows_or_objects": rows, "result_types": dict(result_types)}


def _workload_available(pdf_path: Path) -> tuple[bool, Optional[str]]:
    if pdf_path.exists():
        return True, None
    return False, f"Missing PDF: {_relative_pdf_path(pdf_path)}"


def _make_workload(
    *,
    name: str,
    kind: str,
    description: str,
    cache_mode: str,
    pdf_path: Optional[Path],
    runner: Callable[[], Mapping[str, Any]],
    count_pages: int = 1,
) -> Workload:
    available = True
    skip_reason = None
    if pdf_path is not None:
        available, skip_reason = _workload_available(pdf_path)
    return Workload(
        name=name,
        kind=kind,
        description=description,
        cache_mode=cache_mode,
        runner=runner,
        pdf_path=pdf_path,
        count_pages=count_pages,
        available=available,
        skip_reason=skip_reason,
    )


def discover_workloads(
    *,
    include_render: bool = False,
    include_to_llm: bool = False,
    selected_names: Optional[set[str]] = None,
) -> list[Workload]:
    pdfs = REPO_ROOT / "pdfs"
    workloads: list[Workload] = []

    real_configs = [
        ("01-practice", 1, "baseline small structured extraction"),
        ("Atlanta_Public_Schools_GA_sample", 1, "repeated selectors, exclusions, sections"),
        ("m27", 1, "dense license table extraction"),
        ("guides-expenses-sample", 1, "guide-based expense table extraction"),
        ("hebrew-table", 1, "RTL table and guide extraction"),
        ("0500000US42001", 3, "multi-page election hierarchy extraction"),
    ]

    for config_name, pages, description in real_configs:
        try:
            from benchmark.configs import get_config

            config_class = get_config(config_name)
            config = config_class() if config_class else None
            pdf_path = REPO_ROOT / config.pdf_path if config else pdfs / f"{config_name}.pdf"
        except Exception:
            pdf_path = pdfs / f"{config_name}.pdf"
        workloads.append(
            _make_workload(
                name=f"real:{config_name}",
                kind="real",
                description=description,
                cache_mode="cold",
                pdf_path=pdf_path,
                count_pages=pages,
                runner=lambda config_name=config_name, pages=pages: _config_extraction_result(
                    config_name, pages
                ),
            )
        )

    tiny_pdf = pdfs / "tiny-text-tables.pdf"
    pak_expenses_pdf = pdfs / "pak-ks-expenses.pdf"
    policy_pdf = pdfs / "sample-bop-policy-restaurant.pdf"
    multipage_pdf = (
        pdfs / "multipage-table-african-recipes.pdf"
        if (pdfs / "multipage-table-african-recipes.pdf").exists()
        else pdfs / "multipage-table.pdf"
    )
    m27_pdf = pdfs / "m27.pdf"
    practice_pdf = pdfs / "01-practice.pdf"
    atlanta_pdf = pdfs / "Atlanta_Public_Schools_GA_sample.pdf"

    manual_real_specs: list[tuple[str, str, Path, Callable[[], Mapping[str, Any]], int]] = [
        (
            "real:tiny-text-tables",
            "Tiny text, auto tolerance, region extraction, and pdfplumber table extraction",
            tiny_pdf,
            lambda path=tiny_pdf: _tiny_text_workflow_result(path),
            1,
        ),
        (
            "real:multipage-table",
            "Multi-page table traversal with text, selectors, and table extraction",
            multipage_pdf,
            lambda path=multipage_pdf: _multipage_table_workflow_result(path),
            3,
        ),
        (
            "real:pak-ks-expenses",
            "Dense one-page expenses table with many rectangle cells",
            pak_expenses_pdf,
            lambda path=pak_expenses_pdf: _pak_expenses_workflow_result(path),
            1,
        ),
        (
            "real:policy-lines",
            "Line-heavy policy page with common text and selector scans",
            policy_pdf,
            lambda path=policy_pdf: _policy_lines_workflow_result(path),
            1,
        ),
    ]

    for name, description, pdf_path, runner, count_pages in manual_real_specs:
        workloads.append(
            _make_workload(
                name=name,
                kind="real",
                description=description,
                cache_mode="cold",
                pdf_path=pdf_path,
                count_pages=count_pages,
                runner=runner,
            )
        )

    micro_specs: list[tuple[str, str, Path, Callable[[], Mapping[str, Any]], int]] = [
        (
            "micro:open:01-practice",
            "PDF open and close only",
            practice_pdf,
            lambda path=practice_pdf: _open_pdf_result(path),
            1,
        ),
        (
            "micro:page-materialize:m27",
            "Materialize a dense page, including eager element loading",
            m27_pdf,
            lambda path=m27_pdf: _materialize_page_result(path),
            1,
        ),
        (
            "micro:page-materialize:tiny-text",
            "Materialize the very dense tiny-text page",
            tiny_pdf,
            lambda path=tiny_pdf: _materialize_page_result(path),
            1,
        ),
        (
            "micro:page-materialize:pak-ks-expenses",
            "Materialize a rect-heavy expenses table page",
            pak_expenses_pdf,
            lambda path=pak_expenses_pdf: _materialize_page_result(path),
            1,
        ),
        (
            "micro:page-chars:m27",
            "Materialize native char TextElements on a dense page",
            m27_pdf,
            lambda path=m27_pdf: _materialize_chars_result(path),
            1,
        ),
        (
            "micro:page-chars:tiny-text",
            "Materialize native char TextElements on the tiny-text page",
            tiny_pdf,
            lambda path=tiny_pdf: _materialize_chars_result(path),
            1,
        ),
        (
            "micro:find-all-char:m27",
            "Selector path that explicitly requests character elements",
            m27_pdf,
            lambda path=m27_pdf: _find_all_chars_result(path),
            1,
        ),
        (
            "micro:get-elements:m27",
            "Broad all-element page access on a dense page",
            m27_pdf,
            lambda path=m27_pdf: _get_elements_result(path),
            1,
        ),
        (
            "micro:get-elements:tiny-text",
            "Broad all-element page access on a very dense tiny-text page",
            tiny_pdf,
            lambda path=tiny_pdf: _get_elements_result(path),
            1,
        ),
        (
            "micro:find-all-any:m27",
            "Wildcard selector over all element types on a dense page",
            m27_pdf,
            lambda path=m27_pdf: _find_all_any_result(path),
            1,
        ),
        (
            "micro:describe:m27",
            "Page describe summary on a dense page",
            m27_pdf,
            lambda path=m27_pdf: _describe_page_result(path),
            1,
        ),
        (
            "micro:inspect:m27",
            "Page inspect summary on a dense page",
            m27_pdf,
            lambda path=m27_pdf: _inspect_page_result(path),
            1,
        ),
        (
            "micro:describe:tiny-text",
            "Page describe summary on a very dense tiny-text page",
            tiny_pdf,
            lambda path=tiny_pdf: _describe_page_result(path),
            1,
        ),
        (
            "micro:find-anchors:01-practice",
            "Cold first-match anchor lookups on the practice form",
            practice_pdf,
            lambda path=practice_pdf: _find_anchor_result(
                path,
                selectors=[
                    "text:contains(Site)",
                    "text:contains(Date)",
                    "text:contains(Summary)",
                    "text:contains(Violations)[size=max()]",
                    "text[color~=red]",
                ],
            ),
            1,
        ),
        (
            "micro:find-anchors:atlanta",
            "Cold first-match anchor lookups on the Atlanta sample",
            atlanta_pdf,
            lambda path=atlanta_pdf: _find_anchor_result(
                path,
                selectors=[
                    "text:contains(Author)",
                    "text:contains(ISBN)",
                    "text:contains(Published)",
                    "text:contains(Barcode)",
                    "text[size=max()]",
                ],
            ),
            1,
        ),
        (
            "micro:find-anchors:policy-lines",
            "Cold first-match anchor lookups on a line-heavy policy page",
            policy_pdf,
            lambda path=policy_pdf: _find_anchor_result(
                path,
                selectors=[
                    "text[size=max()]",
                    "text:contains(POLICY)",
                    "text:contains(RESTAURANT)",
                    "text:contains(COVERAGE)",
                ],
            ),
            1,
        ),
        (
            "micro:extract-text-simple:m27",
            "Native text extraction without layout reconstruction",
            m27_pdf,
            lambda path=m27_pdf: _extract_text_result(path, layout=False),
            1,
        ),
        (
            "micro:extract-text-layout:m27",
            "Native text extraction with layout reconstruction",
            m27_pdf,
            lambda path=m27_pdf: _extract_text_result(path, layout=True),
            1,
        ),
        (
            "micro:repeated-selectors:atlanta",
            "Repeated selector scans over a section-heavy page",
            atlanta_pdf,
            lambda path=atlanta_pdf: _repeated_selector_result(path),
            1,
        ),
        (
            "micro:repeated-selectors:tiny-text",
            "Repeated selector scans over a very dense text page",
            tiny_pdf,
            lambda path=tiny_pdf: _repeated_selector_result(path),
            1,
        ),
        (
            "micro:repeated-selectors:pak-ks-expenses",
            "Repeated selector scans over a rect-heavy expenses table",
            pak_expenses_pdf,
            lambda path=pak_expenses_pdf: _repeated_selector_result(path),
            1,
        ),
        (
            "micro:region-navigation:m27",
            "Directional region navigation followed by region find_all",
            m27_pdf,
            lambda path=m27_pdf: _region_navigation_result(path),
            1,
        ),
        (
            "micro:dissolve:hebrew-table",
            "Dissolve text elements into connected regions",
            pdfs / "hebrew-table.pdf",
            lambda path=pdfs / "hebrew-table.pdf": _dissolve_result(path),
            1,
        ),
        (
            "micro:merge-connected:multi-page-table",
            "Dissolve then merge connected regions",
            multipage_pdf,
            lambda path=multipage_pdf: _merge_connected_result(path),
            1,
        ),
        (
            "micro:tiny-text-layout",
            "Tiny text extraction with layout reconstruction",
            tiny_pdf,
            lambda path=tiny_pdf: _extract_text_result(path, layout=True),
            1,
        ),
        (
            "micro:guide-table:01-practice",
            "Guide construction and guide-backed table extraction",
            practice_pdf,
            lambda path=practice_pdf: _guide_table_result(path),
            1,
        ),
        (
            "micro:guide-lines-vector-both:01-practice",
            "Guide line detection from existing vector lines, both axes",
            practice_pdf,
            lambda path=practice_pdf: _guide_lines_result(
                path, axis="both", detection_method="vector"
            ),
            1,
        ),
        (
            "micro:guide-lines-vector-both:policy-lines",
            "Guide line detection from existing vector lines on a line-heavy page",
            policy_pdf,
            lambda path=policy_pdf: _guide_lines_result(
                path, axis="both", detection_method="vector"
            ),
            1,
        ),
        (
            "micro:guide-lines-pixels-both:01-practice",
            "Pixel guide line detection, both axes",
            practice_pdf,
            lambda path=practice_pdf: _guide_lines_result(
                path, axis="both", detection_method="pixels"
            ),
            1,
        ),
        (
            "micro:guide-lines-pixels-horizontal:01-practice",
            "Pixel guide line detection, horizontal axis only",
            practice_pdf,
            lambda path=practice_pdf: _guide_lines_result(
                path, axis="horizontal", detection_method="pixels"
            ),
            1,
        ),
        (
            "micro:guide-lines-pixels-vertical:01-practice",
            "Pixel guide line detection, vertical axis only",
            practice_pdf,
            lambda path=practice_pdf: _guide_lines_result(
                path, axis="vertical", detection_method="pixels"
            ),
            1,
        ),
    ]

    for name, description, pdf_path, runner, count_pages in micro_specs:
        workloads.append(
            _make_workload(
                name=name,
                kind="micro",
                description=description,
                cache_mode="cold",
                pdf_path=pdf_path,
                count_pages=count_pages,
                runner=runner,
            )
        )

    warm_specs: list[tuple[str, str, Path, Callable[[], Mapping[str, Any]], int]] = [
        (
            "warm:repeated-page:m27",
            "Warm page repeated text extraction and selector scans",
            m27_pdf,
            lambda path=m27_pdf: _warm_repeated_page_result(path),
            1,
        ),
        (
            "warm:repeated-page:atlanta",
            "Warm section-heavy page repeated text extraction and selector scans",
            atlanta_pdf,
            lambda path=atlanta_pdf: _warm_repeated_page_result(path),
            1,
        ),
    ]

    for name, description, pdf_path, runner, count_pages in warm_specs:
        workloads.append(
            _make_workload(
                name=name,
                kind="warm",
                description=description,
                cache_mode="warm",
                pdf_path=pdf_path,
                count_pages=count_pages,
                runner=runner,
            )
        )

    if include_render:
        workloads.append(
            _make_workload(
                name="optional:render:01-practice",
                kind="optional",
                description="Page render timing kept separate from extraction",
                cache_mode="cold",
                pdf_path=practice_pdf,
                runner=lambda path=practice_pdf: _render_result(path),
            )
        )

    if include_to_llm:
        workloads.append(
            _make_workload(
                name="optional:to-llm:01-practice",
                kind="optional",
                description="Local .to_llm formatting only, no model calls",
                cache_mode="cold",
                pdf_path=practice_pdf,
                runner=lambda path=practice_pdf: _to_llm_result(path),
            )
        )

    if selected_names is not None:
        workloads = [workload for workload in workloads if workload.name in selected_names]

    return workloads


def run_single_workload(
    workload: Workload,
    *,
    iterations: int,
    warmups: int,
    trace_memory: bool,
    collect_counts: bool,
) -> dict[str, Any]:
    case_result: dict[str, Any] = {
        "name": workload.name,
        "kind": workload.kind,
        "description": workload.description,
        "cache_mode": workload.cache_mode,
        "pdf_path": _relative_pdf_path(workload.pdf_path),
        "available": workload.available,
        "skip_reason": workload.skip_reason,
        "counts": {},
        "runs": [],
        "metrics": {},
        "operation_counts": {},
        "phase_timings_ms": {},
        "vector_metrics": {},
        "errors": [],
    }

    if not workload.available:
        return case_result

    if collect_counts:
        try:
            case_result["counts"] = collect_pdf_counts(workload.pdf_path, workload.count_pages)
        except Exception as exc:
            case_result["counts_error"] = f"{type(exc).__name__}: {exc}"

    total_runs = warmups + iterations
    for run_index in range(total_runs):
        measured = run_index >= warmups
        run_record: dict[str, Any] = {
            "index": run_index - warmups if measured else None,
            "warmup": not measured,
        }
        error: Optional[str] = None
        output: Mapping[str, Any] = {}

        with PerfInstrumentation() as instrumentation:
            _reset_vector_metrics()
            if trace_memory:
                tracemalloc.start()
            wall_start = time.perf_counter()
            cpu_start = time.process_time()
            try:
                output = workload.runner()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            cpu_ms = (time.process_time() - cpu_start) * 1000.0
            wall_ms = (time.perf_counter() - wall_start) * 1000.0
            if trace_memory:
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                run_record["tracemalloc_peak_kib"] = peak / 1024.0
            else:
                run_record["tracemalloc_peak_kib"] = None

            run_record.update(
                {
                    "wall_ms": wall_ms,
                    "cpu_ms": cpu_ms,
                    "rss_peak_kib": _current_rss_kib(),
                    "operation_counts": dict(instrumentation.counts),
                    "phase_ms": dict(instrumentation.phase_ms),
                    "vector_metrics": _snapshot_vector_metrics(),
                    "output": _output_summary(output),
                }
            )
            if error:
                run_record["error"] = error

        if measured:
            case_result["runs"].append(run_record)
            if error:
                case_result["errors"].append(error)

    successful_runs = [run for run in case_result["runs"] if "error" not in run]
    case_result["metrics"] = {
        "wall_ms": _summary(run.get("wall_ms") for run in successful_runs),
        "cpu_ms": _summary(run.get("cpu_ms") for run in successful_runs),
        "tracemalloc_peak_kib": _summary(
            run.get("tracemalloc_peak_kib") for run in successful_runs
        ),
        "rss_peak_kib": _summary(run.get("rss_peak_kib") for run in successful_runs),
    }
    case_result["operation_counts"] = _summarize_counters(successful_runs, "operation_counts")
    case_result["phase_timings_ms"] = _summarize_counters(successful_runs, "phase_ms")
    case_result["vector_metrics"] = _summarize_vector_metrics(successful_runs)
    return case_result


def _format_function(function_key: tuple[str, int, str]) -> str:
    filename, line_number, function_name = function_key
    try:
        short_path = str(Path(filename).relative_to(REPO_ROOT))
    except ValueError:
        short_path = Path(filename).name
    return f"{short_path}:{line_number}({function_name})"


def _profile_rows(
    stats: pstats.Stats,
    *,
    sort_key: str,
    top_n: int,
) -> list[dict[str, Any]]:
    entries = []
    for function_key, stat in stats.stats.items():
        primitive_calls, total_calls, total_time, cumulative_time, _callers = stat
        entries.append(
            {
                "function": _format_function(function_key),
                "primitive_calls": primitive_calls,
                "total_calls": total_calls,
                "self_time_s": total_time,
                "cumulative_time_s": cumulative_time,
            }
        )
    if sort_key == "cumulative":
        entries.sort(key=lambda row: row["cumulative_time_s"], reverse=True)
    elif sort_key == "self":
        entries.sort(key=lambda row: row["self_time_s"], reverse=True)
    elif sort_key == "calls":
        entries.sort(key=lambda row: row["total_calls"], reverse=True)
    else:
        raise ValueError(f"Unsupported profile sort key: {sort_key}")
    return entries[:top_n]


def _format_traceback_frame(frame: tracemalloc.Frame) -> str:
    try:
        short_path = str(Path(frame.filename).relative_to(REPO_ROOT))
    except ValueError:
        short_path = Path(frame.filename).name
    return f"{short_path}:{frame.lineno}"


def _allocation_rows(
    statistics_diff: list[tracemalloc.StatisticDiff],
    *,
    top_n: int,
) -> list[dict[str, Any]]:
    rows = []
    for stat in statistics_diff:
        if stat.size_diff <= 0:
            continue
        rows.append(
            {
                "location": _format_traceback_frame(stat.traceback[0]),
                "size_kib": stat.size_diff / 1024.0,
                "count": stat.count_diff,
                "traceback": [_format_traceback_frame(frame) for frame in stat.traceback[:3]],
            }
        )
        if len(rows) >= top_n:
            break
    return rows


def profile_workload(
    workload: Workload,
    *,
    output_dir: Path,
    top_n: int,
) -> dict[str, Any]:
    profiles_dir = output_dir / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    profile_path = profiles_dir / f"{_safe_filename(workload.name)}.prof"

    profiler = cProfile.Profile()
    error = None
    allocation_hotspots: list[dict[str, Any]] = []
    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()
    try:
        profiler.enable()
        workload.runner()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        profiler.disable()
        snapshot_after = tracemalloc.take_snapshot()
        allocation_hotspots = _allocation_rows(
            snapshot_after.compare_to(snapshot_before, "lineno"),
            top_n=top_n,
        )
        tracemalloc.stop()
    profiler.dump_stats(profile_path)

    stats = pstats.Stats(profiler)
    return {
        "case": workload.name,
        "profile_path": str(profile_path),
        "error": error,
        "top_cumulative": _profile_rows(stats, sort_key="cumulative", top_n=top_n),
        "top_self_time": _profile_rows(stats, sort_key="self", top_n=top_n),
        "top_call_count": _profile_rows(stats, sort_key="calls", top_n=top_n),
        "allocation_hotspots": allocation_hotspots,
    }


def select_workloads_to_profile(
    workloads: list[Workload],
    case_results: list[dict[str, Any]],
    profile_slowest: int,
) -> list[Workload]:
    if profile_slowest <= 0:
        return []
    workload_by_name = {workload.name: workload for workload in workloads if workload.available}
    sortable = []
    for result in case_results:
        if result.get("errors"):
            continue
        median_wall = result.get("metrics", {}).get("wall_ms", {}).get("median")
        if median_wall is None:
            continue
        sortable.append((float(median_wall), result["name"], result.get("kind")))
    sortable.sort(reverse=True)

    real_names = [name for _wall, name, kind in sortable if kind == "real"]
    names: list[str] = real_names[:profile_slowest]
    if len(names) < profile_slowest:
        names.extend(name for _wall, name, _kind in sortable if name not in names)
        names = names[:profile_slowest]
    for _wall, name, _kind in sortable:
        if "tiny" in name and name not in names:
            names.append(name)
            break
    return [workload_by_name[name] for name in names if name in workload_by_name]


def run_pytest_durations(timeout: int = 300) -> dict[str, Any]:
    args = [
        "uv",
        "run",
        "pytest",
        "-q",
        "--durations=30",
        "-m",
        NON_AI_MARKER_EXPR,
        "-o",
        "cache_dir=/tmp/natural-pdf-pytest-cache",
        *NON_AI_PYTEST_TARGETS,
    ]
    result = _run_subprocess(args, timeout=timeout)
    result["duration_lines"] = _extract_duration_lines(result.get("stdout", ""))
    return result


def _extract_duration_lines(stdout: str) -> list[str]:
    lines = stdout.splitlines()
    duration_lines = []
    in_section = False
    for line in lines:
        if "slowest" in line and "durations" in line:
            in_section = True
            continue
        if in_section:
            if not line.strip():
                continue
            if line.startswith("="):
                continue
            if " passed" in line or " failed" in line:
                break
            duration_lines.append(line.rstrip())
    return duration_lines


def run_workloads(
    workloads: list[Workload],
    *,
    output_dir: Path,
    iterations: int,
    warmups: int,
    trace_memory: bool,
    collect_counts: bool,
    profile_slowest: int,
    profile_top_n: int,
    include_pytest_durations: bool,
    experiment_label: Optional[str] = None,
    patch_records: Optional[list[dict[str, Any]]] = None,
    patch_errors: Optional[list[str]] = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    case_results = [
        run_single_workload(
            workload,
            iterations=iterations,
            warmups=warmups,
            trace_memory=trace_memory,
            collect_counts=collect_counts,
        )
        for workload in workloads
    ]
    profile_targets = select_workloads_to_profile(workloads, case_results, profile_slowest)
    profile_results = [
        profile_workload(workload, output_dir=output_dir, top_n=profile_top_n)
        for workload in profile_targets
    ]
    pytest_result = run_pytest_durations() if include_pytest_durations else None

    return {
        "schema_version": "1.0",
        "generated_at": _now_iso(),
        "repo": get_git_info(),
        "environment": get_environment_info(),
        "settings": {
            "iterations": iterations,
            "warmups": warmups,
            "trace_memory": trace_memory,
            "collect_counts": collect_counts,
            "profile_slowest": profile_slowest,
            "profile_top_n": profile_top_n,
            "include_pytest_durations": include_pytest_durations,
        },
        "experiment": {
            "label": experiment_label,
            "patches": patch_records or [],
            "patch_errors": patch_errors or [],
        },
        "cases": case_results,
        "profiles": profile_results,
        "pytest_durations": pytest_result,
    }


def build_patch_failure_result(
    *,
    output_dir: Path,
    iterations: int,
    warmups: int,
    trace_memory: bool,
    collect_counts: bool,
    profile_slowest: int,
    profile_top_n: int,
    include_pytest_durations: bool,
    experiment_label: Optional[str],
    patch_records: list[dict[str, Any]],
    patch_errors: list[str],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "schema_version": "1.0",
        "generated_at": _now_iso(),
        "repo": get_git_info(),
        "environment": get_environment_info(),
        "settings": {
            "iterations": iterations,
            "warmups": warmups,
            "trace_memory": trace_memory,
            "collect_counts": collect_counts,
            "profile_slowest": profile_slowest,
            "profile_top_n": profile_top_n,
            "include_pytest_durations": include_pytest_durations,
        },
        "experiment": {
            "label": experiment_label,
            "patches": patch_records,
            "patch_errors": patch_errors,
        },
        "cases": [],
        "profiles": [],
        "pytest_durations": None,
    }


def _case_metric(case: Mapping[str, Any], metric: str, stat: str = "median") -> Optional[float]:
    value = case.get("metrics", {}).get(metric, {}).get(stat)
    if value is None:
        return None
    return float(value)


def _total_operation(case: Mapping[str, Any], label: str) -> float:
    return float(case.get("operation_counts", {}).get(label, {}).get("total", 0.0))


def _total_phase(case: Mapping[str, Any], label: str) -> float:
    return float(case.get("phase_timings_ms", {}).get(label, {}).get("total", 0.0))


def _profile_contains(profile: Mapping[str, Any], needles: tuple[str, ...]) -> bool:
    rows = (
        list(profile.get("top_cumulative", []))
        + list(profile.get("top_self_time", []))
        + list(profile.get("top_call_count", []))
    )
    for row in rows:
        function = row.get("function", "")
        if any(needle in function for needle in needles):
            return True
    return False


def _profile_functions_matching(
    profiles: Iterable[Mapping[str, Any]],
    needles: tuple[str, ...],
    *,
    limit: int = 3,
) -> list[str]:
    functions: list[str] = []
    for profile in profiles:
        rows = (
            list(profile.get("top_cumulative", []))
            + list(profile.get("top_self_time", []))
            + list(profile.get("top_call_count", []))
        )
        for row in rows:
            function = row.get("function", "")
            if any(needle in function for needle in needles) and function not in functions:
                functions.append(function)
                if len(functions) >= limit:
                    return functions
    return functions


def _top_profile_functions(profiles: Iterable[Mapping[str, Any]], *, limit: int = 3) -> list[str]:
    functions: list[str] = []
    for profile in profiles:
        for row in profile.get("top_cumulative", [])[:1]:
            function = row.get("function", "")
            if function and function not in functions:
                functions.append(function)
                if len(functions) >= limit:
                    return functions
    return functions


def _top_cases_by_operation(
    cases: Iterable[Mapping[str, Any]],
    label: str,
    *,
    limit: int = 3,
) -> list[str]:
    ranked = [
        (_total_operation(case, label), str(case.get("name")))
        for case in cases
        if _total_operation(case, label) > 0
    ]
    ranked.sort(reverse=True)
    return [f"{name} ({count:.0f})" for count, name in ranked[:limit]]


def _top_cases_by_phase(
    cases: Iterable[Mapping[str, Any]],
    labels: tuple[str, ...],
    *,
    limit: int = 3,
) -> list[str]:
    ranked = []
    for case in cases:
        phase_total = sum(_total_phase(case, label) for label in labels)
        if phase_total > 0:
            ranked.append((phase_total, str(case.get("name"))))
    ranked.sort(reverse=True)
    return [f"{name} ({phase_ms:.1f} ms)" for phase_ms, name in ranked[:limit]]


def _join_evidence(values: Iterable[str], *, fallback: str = "none") -> str:
    clean = [value for value in values if value]
    return ", ".join(clean) if clean else fallback


def build_improvement_options(run_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    cases = list(run_result.get("cases", []))
    profiles = list(run_result.get("profiles", []))
    successful_cases = [case for case in cases if case.get("available") and not case.get("errors")]
    slow_cases = sorted(
        successful_cases,
        key=lambda case: _case_metric(case, "wall_ms") or 0.0,
        reverse=True,
    )
    slow_case_names = [case["name"] for case in slow_cases[:3]]
    max_wall = _case_metric(slow_cases[0], "wall_ms") if slow_cases else None

    selector_total = sum(_total_operation(case, "find_all") for case in successful_cases)
    exclusion_total = sum(_total_operation(case, "exclusion_filter") for case in successful_cases)
    table_total = sum(_total_operation(case, "table_extract") for case in successful_cases)
    spatial_total = sum(
        _total_operation(case, "region_overlap_filter") for case in successful_cases
    )
    render_total = sum(_total_operation(case, "render") for case in successful_cases)
    to_llm_total = sum(_total_operation(case, "to_llm") for case in successful_cases)

    context_hits = _profile_functions_matching(
        profiles, ("context.py", "PDFContext", "_default_qa_factory")
    )
    element_hits = _profile_functions_matching(
        profiles, ("element_manager.py", "word_engine.py", "base.py", "text.py")
    )
    table_hits = _profile_functions_matching(
        profiles, ("table_service.py", "plumber.py", "guides", "table.py")
    )
    selector_hits = _profile_functions_matching(
        profiles, ("selector_service.py", "selector_utils.py", "spatial.py")
    )
    spatial_hits = _profile_functions_matching(
        profiles, ("spatial.py", "geometry_mixin.py", "region.py", "element_collection.py")
    )
    tiny_hits = _profile_functions_matching(
        [profile for profile in profiles if "tiny" in str(profile.get("case"))],
        ("text", "word", "table", "plumber", "operations.py"),
    )
    top_profile_functions = _top_profile_functions(profiles)

    tiny_cases = [case for case in successful_cases if "tiny" in case["name"]]
    tiny_wall = max((_case_metric(case, "wall_ms") or 0.0 for case in tiny_cases), default=0.0)
    element_phase_total = sum(
        _total_phase(case, "page_create")
        + _total_phase(case, "element_load")
        + _total_phase(case, "element_populate")
        + _total_phase(case, "word_generate")
        for case in successful_cases
    )
    selector_cases = _top_cases_by_operation(successful_cases, "find_all")
    exclusion_cases = _top_cases_by_operation(successful_cases, "exclusion_filter")
    table_cases = _top_cases_by_operation(successful_cases, "table_extract")
    spatial_cases = _top_cases_by_operation(successful_cases, "region_overlap_filter")
    element_cases = _top_cases_by_phase(
        successful_cases,
        ("page_create", "element_load", "element_populate", "word_generate"),
    )

    options = [
        {
            "title": "Repeated context/service setup",
            "score": (50 if context_hits else 0),
            "confidence": "measured" if context_hits else "needs evidence",
            "risk": "low to medium",
            "evidence": (
                f"profile hits={_join_evidence(context_hits)}; "
                f"top profiled functions={_join_evidence(top_profile_functions)}."
                if context_hits
                else "No profiled context setup hotspot detected yet."
            ),
            "next_step": (
                "Inspect eager default construction and non-AI service imports."
                if context_hits
                else "Do not prioritize until profiles show context setup cost."
            ),
        },
        {
            "title": "Repeated selector/exclusion scans",
            "score": selector_total + exclusion_total + (40 if selector_hits else 0),
            "confidence": "measured" if selector_total or exclusion_total else "needs evidence",
            "risk": "medium",
            "evidence": (
                f"find_all total={selector_total:.0f} cases={_join_evidence(selector_cases)}; "
                f"exclusion_filter total={exclusion_total:.0f} cases={_join_evidence(exclusion_cases)}; "
                f"profile hits={_join_evidence(selector_hits)}."
            ),
            "next_step": "Look for repeated equivalent queries before adding spatial indexes.",
        },
        {
            "title": "Element/word creation",
            "score": (70 if element_hits else 0) + element_phase_total / 100.0,
            "confidence": "measured" if element_hits else "partial",
            "risk": "medium",
            "evidence": (
                f"phase cases={_join_evidence(element_cases)}; "
                f"profile hits={_join_evidence(element_hits)}."
            ),
            "next_step": "Separate page creation, element wrapping, and word generation costs.",
        },
        {
            "title": "Tiny text tolerance path",
            "score": tiny_wall,
            "confidence": "measured" if tiny_wall else "needs evidence",
            "risk": "medium",
            "evidence": (
                f"Slowest tiny-text median wall time is {tiny_wall:.1f} ms; "
                f"profile hits={_join_evidence(tiny_hits)}."
                if tiny_wall
                else "Tiny text workload was not measured or did not complete."
            ),
            "next_step": "Profile auto tolerance and layout text reconstruction on tiny PDFs.",
        },
        {
            "title": "Table extraction",
            "score": table_total + (50 if table_hits else 0),
            "confidence": "measured" if table_total or table_hits else "needs evidence",
            "risk": "medium",
            "evidence": (
                f"table_extract total={table_total:.0f} cases={_join_evidence(table_cases)}; "
                f"slow cases={_join_evidence(slow_case_names)}; "
                f"profile hits={_join_evidence(table_hits)}."
            ),
            "next_step": "Compare guide setup, cell assignment, pdfplumber calls, and text extraction.",
        },
        {
            "title": "Spatial indexing",
            "score": spatial_total,
            "confidence": "defer unless scan counts dominate",
            "risk": "medium to high",
            "evidence": (
                f"region_overlap_filter total={spatial_total:.0f} "
                f"cases={_join_evidence(spatial_cases)}; "
                f"profile hits={_join_evidence(spatial_hits)}."
            ),
            "next_step": "Only prototype indexing if repeated broad scans dominate simpler fixes.",
        },
        {
            "title": "Rendering/to_llm",
            "score": render_total + to_llm_total,
            "confidence": "measured" if render_total or to_llm_total else "not in current run",
            "risk": "separate track",
            "evidence": f"render total={render_total:.0f}, to_llm total={to_llm_total:.0f}.",
            "next_step": "Keep separate from core extraction unless users report this path.",
        },
    ]

    for option in options:
        if max_wall:
            option["slowest_case_median_wall_ms"] = max_wall
    options.sort(key=lambda option: float(option["score"]), reverse=True)
    return options


def _safe_filename(name: str) -> str:
    safe = []
    for char in name:
        safe.append(char if char.isalnum() or char in ("-", "_") else "_")
    return "".join(safe).strip("_") or "case"


def _fmt_ms(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}"


def _escape_table(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def build_profile_summary_md(run_result: Mapping[str, Any]) -> str:
    cases = list(run_result.get("cases", []))
    successful_cases = [case for case in cases if case.get("available") and not case.get("errors")]
    slow_cases = sorted(
        successful_cases,
        key=lambda case: _case_metric(case, "wall_ms") or 0.0,
        reverse=True,
    )
    lines = [
        "# Natural PDF Performance Profile Summary",
        "",
        f"Generated: {run_result.get('generated_at')}",
        f"Git SHA: {run_result.get('repo', {}).get('sha')}",
        f"Dirty worktree: {run_result.get('repo', {}).get('dirty')}",
        "",
    ]

    experiment = run_result.get("experiment") or {}
    patches = experiment.get("patches") or []
    patch_errors = experiment.get("patch_errors") or []
    if experiment.get("label") or patches or patch_errors:
        lines.extend(
            [
                "## Experiment",
                "",
                f"Label: `{experiment.get('label') or 'baseline'}`",
            ]
        )
        if patches:
            lines.extend(["", "| Patch | Status | Metadata |", "|---|---|---|"])
            for patch in patches:
                lines.append(
                    "| {path} | {status} | {metadata} |".format(
                        path=_escape_table(patch.get("path")),
                        status=_escape_table(patch.get("status")),
                        metadata=_escape_table(
                            json.dumps(patch.get("metadata", {}), sort_keys=True)
                        ),
                    )
                )
        if patch_errors:
            lines.extend(["", "Patch install errors:"])
            for error in patch_errors:
                lines.append(f"- `{error}`")
        lines.append("")

    lines.extend(
        [
            "## Slowest Cases",
            "",
            "| Case | Kind | Median wall ms | P95 wall ms | Median CPU ms | Peak Python KiB |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for case in slow_cases[:12]:
        lines.append(
            "| {name} | {kind} | {wall} | {p95} | {cpu} | {mem} |".format(
                name=_escape_table(case["name"]),
                kind=_escape_table(case["kind"]),
                wall=_fmt_ms(_case_metric(case, "wall_ms")),
                p95=_fmt_ms(_case_metric(case, "wall_ms", "p95")),
                cpu=_fmt_ms(_case_metric(case, "cpu_ms")),
                mem=_fmt_ms(_case_metric(case, "tracemalloc_peak_kib")),
            )
        )

    lines.extend(
        [
            "",
            "## Operation Counts",
            "",
            "| Case | find_all | selector_query | exclusion_filter | region_overlap_filter | table_extract |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for case in successful_cases:
        lines.append(
            "| {name} | {find_all:.0f} | {selector:.0f} | {excl:.0f} | {region:.0f} | {table:.0f} |".format(
                name=_escape_table(case["name"]),
                find_all=_total_operation(case, "find_all"),
                selector=_total_operation(case, "selector_query"),
                excl=_total_operation(case, "exclusion_filter"),
                region=_total_operation(case, "region_overlap_filter"),
                table=_total_operation(case, "table_extract"),
            )
        )

    profiles = list(run_result.get("profiles", []))
    lines.extend(["", "## cProfile Highlights", ""])
    if not profiles:
        lines.append("No cProfile runs were requested.")
    for profile in profiles:
        lines.extend(
            [
                f"### {profile.get('case')}",
                "",
                f"Raw profile: `{profile.get('profile_path')}`",
                "",
                "Top cumulative time:",
            ]
        )
        for row in profile.get("top_cumulative", [])[:10]:
            lines.append(
                "- `{}`: {:.4f}s cumulative, {:.4f}s self, {} calls".format(
                    row["function"],
                    row["cumulative_time_s"],
                    row["self_time_s"],
                    row["total_calls"],
                )
            )
        lines.append("")
        lines.append("Top self time:")
        for row in profile.get("top_self_time", [])[:10]:
            lines.append(
                "- `{}`: {:.4f}s self, {:.4f}s cumulative, {} calls".format(
                    row["function"],
                    row["self_time_s"],
                    row["cumulative_time_s"],
                    row["total_calls"],
                )
            )
        lines.append("")
        lines.append("Top call counts:")
        for row in profile.get("top_call_count", [])[:10]:
            lines.append(
                "- `{}`: {} calls, {:.4f}s cumulative".format(
                    row["function"],
                    row["total_calls"],
                    row["cumulative_time_s"],
                )
            )
        lines.append("")
        lines.append("Allocation hotspots:")
        hotspots = profile.get("allocation_hotspots", [])[:10]
        if not hotspots:
            lines.append("- No positive net Python allocation hotspots recorded.")
        for row in hotspots:
            lines.append(
                "- `{}`: {:.1f} KiB, {} allocations".format(
                    row["location"],
                    row["size_kib"],
                    row["count"],
                )
            )
        lines.append("")

    pytest_result = run_result.get("pytest_durations")
    lines.extend(["", "## Pytest Duration Leads", ""])
    if not pytest_result:
        lines.append("Pytest duration collection was not requested.")
    elif not pytest_result.get("ok"):
        lines.append(f"Pytest duration collection failed: {pytest_result.get('error')}")
    else:
        duration_lines = pytest_result.get("duration_lines") or []
        if duration_lines:
            for line in duration_lines:
                lines.append(f"- `{line}`")
        else:
            lines.append("No duration lines were parsed from pytest output.")

    skipped = [case for case in cases if not case.get("available")]
    if skipped:
        lines.extend(["", "## Skipped Cases", ""])
        for case in skipped:
            lines.append(f"- `{case['name']}`: {case.get('skip_reason')}")

    return "\n".join(lines) + "\n"


def build_improvement_menu_md(run_result: Mapping[str, Any]) -> str:
    options = build_improvement_options(run_result)
    lines = [
        "# Natural PDF Optimization Menu",
        "",
        "This menu is generated from the current perf baseline. Re-run after each optimization",
        "and compare medians before choosing the next item.",
        "",
    ]
    experiment = run_result.get("experiment") or {}
    if experiment.get("label") or experiment.get("patches") or experiment.get("patch_errors"):
        lines.append(f"Experiment label: `{experiment.get('label') or 'baseline'}`.")
        if experiment.get("patch_errors"):
            lines.append("Patch install errors were recorded; do not compare this run as patched.")
        lines.append("")
    if options:
        top_score = float(options[0]["score"])
        if top_score > 0:
            lines.append(f"Recommended first pick: **{options[0]['title']}**.")
        else:
            lines.append("Recommended first pick: run a broader baseline before optimizing.")
        lines.append("")
    lines.extend(
        [
            "| Rank | Option | Score | Confidence | Risk | Evidence | Next step |",
            "|---:|---|---:|---|---|---|---|",
        ]
    )
    for index, option in enumerate(options, start=1):
        lines.append(
            "| {rank} | {title} | {score:.1f} | {confidence} | {risk} | {evidence} | {next_step} |".format(
                rank=index,
                title=_escape_table(option["title"]),
                score=float(option["score"]),
                confidence=_escape_table(option["confidence"]),
                risk=_escape_table(option["risk"]),
                evidence=_escape_table(option["evidence"]),
                next_step=_escape_table(option["next_step"]),
            )
        )

    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Do not implement spatial indexes until broad overlap scans are proven dominant.",
            "- Keep rendering and `.to_llm()` separate from core extraction timing.",
            "- Treat pytest durations as leads, not benchmark truth.",
            "- For every optimization, compare against `baseline.json` from the prior run.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_artifacts(run_result: Mapping[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = output_dir / "baseline.json"
    profile_summary_path = output_dir / "profile-summary.md"
    improvement_menu_path = output_dir / "improvement-menu.md"

    baseline_path.write_text(json.dumps(run_result, indent=2, default=_json_default) + "\n")
    profile_summary_path.write_text(build_profile_summary_md(run_result))
    improvement_menu_path.write_text(build_improvement_menu_md(run_result))
    return {
        "baseline": str(baseline_path),
        "profile_summary": str(profile_summary_path),
        "improvement_menu": str(improvement_menu_path),
    }


def _parse_case_selection(raw_cases: Optional[str]) -> Optional[set[str]]:
    if not raw_cases:
        return None
    return {case.strip() for case in raw_cases.split(",") if case.strip()}


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure non-AI, non-OCR Natural PDF performance workloads."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for baseline and reports (default: perf_output).",
    )
    parser.add_argument("--iterations", type=int, default=5, help="Measured runs per case.")
    parser.add_argument("--warmups", type=int, default=1, help="Warmup runs per case.")
    parser.add_argument(
        "--cases",
        help="Comma-separated case names. Use --list-cases to inspect available names.",
    )
    parser.add_argument(
        "--patch",
        action="append",
        type=Path,
        default=[],
        help=(
            "Path to an experiment patch module. May be passed multiple times. "
            "Each module must define install() returning a context manager or None."
        ),
    )
    parser.add_argument(
        "--experiment-label",
        help="Human-readable label recorded in baseline.json for patched experiment runs.",
    )
    parser.add_argument(
        "--profile-slowest",
        type=int,
        default=None,
        help=(
            "Number of slowest cases to profile with cProfile "
            f"(default: {DEFAULT_PROFILE_SLOWEST}; quick default: 1)."
        ),
    )
    parser.add_argument(
        "--profile-top-n",
        type=int,
        default=DEFAULT_PROFILE_TOP_N,
        help="Rows to keep for each profile summary section.",
    )
    parser.add_argument(
        "--include-render",
        action="store_true",
        help="Include rendering as a separate optional workload.",
    )
    parser.add_argument(
        "--include-to-llm",
        action="store_true",
        help="Include local .to_llm formatting as a separate optional workload.",
    )
    parser.add_argument(
        "--include-pytest-durations",
        action="store_true",
        help="Run a non-AI/non-OCR pytest subset with --durations=30.",
    )
    parser.add_argument(
        "--no-tracemalloc",
        action="store_true",
        help="Disable per-run tracemalloc peak measurements.",
    )
    parser.add_argument(
        "--no-counts",
        action="store_true",
        help="Skip separate PDF element counts.",
    )
    parser.add_argument("--list-cases", action="store_true", help="List case names and exit.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Shortcut for --iterations 1 --warmups 0 --profile-slowest 1.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.quick:
        args.iterations = 1
        args.warmups = 0
    if args.profile_slowest is None:
        args.profile_slowest = 1 if args.quick else DEFAULT_PROFILE_SLOWEST

    selected_names = _parse_case_selection(args.cases)
    workloads = discover_workloads(
        include_render=args.include_render,
        include_to_llm=args.include_to_llm,
        selected_names=selected_names,
    )

    if args.list_cases:
        for workload in workloads:
            status = "available" if workload.available else f"skipped: {workload.skip_reason}"
            print(f"{workload.name}\t{workload.kind}\t{status}\t{workload.description}")
        return 0

    if selected_names:
        found_names = {workload.name for workload in workloads}
        missing = sorted(selected_names - found_names)
        if missing:
            print(f"Unknown case(s): {', '.join(missing)}", file=sys.stderr)
            return 2

    iterations = max(1, args.iterations)
    warmups = max(0, args.warmups)
    trace_memory = not args.no_tracemalloc
    collect_counts = not args.no_counts
    profile_slowest = max(0, args.profile_slowest)
    profile_top_n = max(1, args.profile_top_n)

    with ExperimentPatchManager(args.patch) as patch_manager:
        if patch_manager.errors:
            run_result = build_patch_failure_result(
                output_dir=args.output,
                iterations=iterations,
                warmups=warmups,
                trace_memory=trace_memory,
                collect_counts=collect_counts,
                profile_slowest=profile_slowest,
                profile_top_n=profile_top_n,
                include_pytest_durations=args.include_pytest_durations,
                experiment_label=args.experiment_label,
                patch_records=patch_manager.records,
                patch_errors=patch_manager.errors,
            )
            paths = write_artifacts(run_result, args.output)
            print("Patch installation failed; artifacts written:", file=sys.stderr)
            for label, path in paths.items():
                print(f"  {label}: {path}", file=sys.stderr)
            for error in patch_manager.errors:
                print(f"  error: {error}", file=sys.stderr)
            return 3

        run_result = run_workloads(
            workloads,
            output_dir=args.output,
            iterations=iterations,
            warmups=warmups,
            trace_memory=trace_memory,
            collect_counts=collect_counts,
            profile_slowest=profile_slowest,
            profile_top_n=profile_top_n,
            include_pytest_durations=args.include_pytest_durations,
            experiment_label=args.experiment_label,
            patch_records=patch_manager.records,
            patch_errors=patch_manager.errors,
        )
    paths = write_artifacts(run_result, args.output)
    print("Performance artifacts written:")
    for label, path in paths.items():
        print(f"  {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
