"""
Dynamic HTML Report Generator

Single-page app with hash routing. All data stored as JSON.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from benchmark.providers import parse_csv_response, parse_json_response
from benchmark.schemas import GroundTruth, LLMResult
from benchmark.utils import find_matching_column, normalize_header, values_match


def generate_pdf_report(
    ground_truth: GroundTruth,
    llm_results: list[LLMResult],
    output_path: str | Path,
) -> str:
    """
    Generate report_data.json for a single PDF.

    No HTML is generated - just the JSON data file.
    The single-page app loads this JSON via hash routing.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build report data
    report_data = _build_report_data(ground_truth, llm_results)

    # Write the JSON data file
    json_path = output_path.parent / "report_data.json"
    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=2)

    return str(json_path)


def _build_report_data(
    ground_truth: GroundTruth,
    llm_results: list[LLMResult],
) -> dict:
    """Build the complete report data structure."""

    # Get prompt from first result
    prompt_template = llm_results[0].prompt_template if llm_results else ""

    # Build models data
    models = []
    for result in llm_results:
        # Calculate accuracy display
        if result.accuracy_score < 0:
            accuracy = None
            accuracy_display = "N/A"
        else:
            accuracy = result.accuracy_score
            pct = accuracy * 100
            # Use 2 decimal places for >98% to show fine differences
            if pct > 98:
                accuracy_display = f"{pct:.2f}".rstrip("0").rstrip(".") + "%"
            else:
                accuracy_display = f"{round(pct)}%"

        # Calculate total time
        total_ms = sum(r.latency_ms for r in result.responses if r.latency_ms)
        total_time = total_ms / 1000 if total_ms else None

        # Build extracted data with correctness markers
        extracted = _build_extracted_data(result, ground_truth)

        # Count correct/total from extracted data
        correct_count = 0
        total_count = 0
        for f in extracted.get("fields", []):
            total_count += 1
            if f.get("status") == "correct":
                correct_count += 1
        for table in extracted.get("tables", []):
            for row in table.get("rows", []):
                for cell in row.get("cells", []):
                    total_count += 1
                    if cell.get("status") == "correct":
                        correct_count += 1

        # Build trap results
        trap_results = []
        for trap in result.trap_results or []:
            trap_results.append(
                {
                    "name": trap.trap_name,
                    "expected": trap.expected,
                    "actual": trap.actual,
                    "passed": trap.passed,
                    "category": trap.category,
                    "description": getattr(trap, "description", "") or trap.trap_name,
                }
            )

        # Build raw responses
        raw_responses = []
        for resp in result.responses:
            raw_responses.append(
                {
                    "page": resp.page_number,
                    "text": resp.raw_response,
                    "latency_ms": resp.latency_ms,
                }
            )

        models.append(
            {
                "name": result.model,
                "provider": result.provider,
                "accuracy": accuracy,
                "accuracy_display": accuracy_display,
                "correct_count": correct_count,
                "error_count": total_count - correct_count,
                "total_count": total_count,
                "total_time": total_time,
                "extracted": extracted,
                "trap_results": trap_results,
                "raw_responses": raw_responses,
            }
        )

    # Build page image paths (relative to index.html)
    # Normalize pdf_name the same way BenchmarkOutput.pdf_dir does
    clean_name = ground_truth.pdf_name.replace(".pdf", "").replace(" ", "_").lower()
    page_images = []
    for i in range(ground_truth.total_pages):
        page_images.append(f"{clean_name}/pages/page_{i+1:03d}.png")

    return {
        "pdf_name": ground_truth.pdf_name,
        "is_trap": ground_truth.is_trap,
        "total_pages": ground_truth.total_pages,
        "data_format": ground_truth.data_format,
        "prompt_template": prompt_template,
        "models": models,
        "page_images": page_images,
        "generated_at": datetime.now().isoformat(),
    }


def _build_extracted_data(result: LLMResult, ground_truth: GroundTruth) -> dict:
    """Build extracted data structure with correctness markers."""
    fields = []
    tables = []

    is_tabular = ground_truth.data_format == "tabular"

    # With native PDF support, we have a single response for the entire PDF
    if not result.responses:
        return {"fields": fields, "tables": tables}

    pdf_response = result.responses[0]
    raw = pdf_response.raw_response

    # Parse LLM response
    llm_data = parse_json_response(raw)
    if not llm_data:
        llm_data = parse_csv_response(raw)

    if not llm_data:
        tables.append(
            {
                "name": "Extracted Data",
                "headers": [],
                "rows": [],
                "error": "No parseable data",
            }
        )
        return {"fields": fields, "tables": tables}

    if is_tabular:
        # Collect all ground truth rows
        all_gt_rows = []
        for page in ground_truth.pages:
            if page.data:
                all_gt_rows.extend(page.data)

        # Build table
        headers = list(llm_data[0].keys()) if llm_data else []
        table_rows = []

        for i, llm_row in enumerate(llm_data):
            gt_row = all_gt_rows[i] if i < len(all_gt_rows) else {}
            gt_headers = list(gt_row.keys()) if gt_row else []

            cells = []
            row_status = "correct"

            for h in headers:
                llm_val = str(llm_row.get(h, ""))
                matching_gt_col = find_matching_column(h, gt_headers)
                gt_val = str(gt_row.get(matching_gt_col, "")) if matching_gt_col else ""

                cell_wrong = gt_val and not values_match(gt_val, llm_val)
                if cell_wrong:
                    row_status = "wrong"

                cells.append(
                    {
                        "value": llm_val,
                        "expected": gt_val if cell_wrong else None,
                        "status": "wrong" if cell_wrong else "correct",
                    }
                )

            table_rows.append(
                {
                    "cells": cells,
                    "status": row_status,
                }
            )

        # Build missing rows (ground truth rows beyond what LLM extracted)
        missing_rows = []
        if len(all_gt_rows) > len(llm_data):
            # Use ground truth headers for missing rows
            gt_headers = list(all_gt_rows[0].keys()) if all_gt_rows else []
            for i in range(len(llm_data), len(all_gt_rows)):
                gt_row = all_gt_rows[i]
                cells = []
                for h in gt_headers:
                    cells.append(
                        {
                            "value": str(gt_row.get(h, "")),
                            "header": h,
                        }
                    )
                missing_rows.append({"cells": cells})

        tables.append(
            {
                "name": "Extracted Data",
                "headers": headers,
                "rows": table_rows,
                "row_count": {
                    "extracted": len(llm_data),
                    "expected": len(all_gt_rows),
                },
                "missing_rows": missing_rows,
                "missing_headers": (
                    list(all_gt_rows[0].keys()) if missing_rows and all_gt_rows else []
                ),
            }
        )
    else:
        # Structured data (like 01-practice)
        gt_data = {}
        gt_tables = {}
        for page in ground_truth.pages:
            if page.data:
                for item in page.data:
                    for key, value in item.items():
                        if isinstance(value, list):
                            gt_tables[key] = value
                        else:
                            gt_data[normalize_header(key)] = str(value)

        json_data = parse_json_response(raw)
        if json_data:
            for item in json_data:
                if isinstance(item, dict):
                    _process_json_item(item, fields, tables, gt_data, gt_tables)
        else:
            csv_data = parse_csv_response(raw)
            if csv_data:
                if csv_data and "Field" in csv_data[0] and "Value" in csv_data[0]:
                    for row in csv_data:
                        field_name = row.get("Field", "")
                        field_value = row.get("Value", "")
                        if field_name:
                            gt_val = gt_data.get(normalize_header(field_name), "")
                            is_match = values_match(gt_val, str(field_value))
                            fields.append(
                                {
                                    "field": str(field_name),
                                    "value": str(field_value) if field_value else "(empty)",
                                    "expected": gt_val if not is_match else None,
                                    "status": "correct" if is_match else "wrong",
                                }
                            )
                else:
                    if csv_data:
                        headers = list(csv_data[0].keys())
                        table_rows = []
                        for row in csv_data:
                            cells = []
                            for h in headers:
                                val = row.get(h, "")
                                cells.append(
                                    {
                                        "value": str(val),
                                        "expected": None,
                                        "status": "neutral",
                                    }
                                )
                            table_rows.append({"cells": cells, "status": "neutral"})
                        tables.append({"name": "Data", "headers": headers, "rows": table_rows})

    return {"fields": fields, "tables": tables}


def _process_json_item(item: dict, fields: list, tables: list, gt_data: dict, gt_tables: dict):
    """Process a JSON object, extracting fields and tables."""
    for key, value in item.items():
        if isinstance(value, dict):
            for k, v in value.items():
                llm_val = str(v) if v else ""
                gt_val = gt_data.get(normalize_header(k), "")
                is_match = values_match(gt_val, llm_val)
                fields.append(
                    {
                        "field": k,
                        "value": llm_val if llm_val else "(empty)",
                        "expected": gt_val if not is_match else None,
                        "status": "correct" if is_match else "wrong",
                    }
                )
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            headers = list(value[0].keys())
            gt_table = gt_tables.get("violations_table", [])
            table_rows = []

            for i, row in enumerate(value):
                cells = []
                row_status = "correct"
                gt_row = gt_table[i] if i < len(gt_table) else {}
                gt_headers = list(gt_row.keys()) if gt_row else []

                for h in headers:
                    llm_val = str(row.get(h, ""))
                    matching_gt_col = find_matching_column(h, gt_headers)
                    gt_val = str(gt_row.get(matching_gt_col, "")) if matching_gt_col else ""
                    cell_wrong = gt_val and not values_match(gt_val, llm_val)
                    if cell_wrong:
                        row_status = "wrong"
                    cells.append(
                        {
                            "value": llm_val,
                            "expected": gt_val if cell_wrong else None,
                            "status": "wrong" if cell_wrong else "correct",
                        }
                    )

                table_rows.append({"cells": cells, "status": row_status})

            tables.append({"name": key, "headers": headers, "rows": table_rows})
        else:
            llm_val = str(value) if value else ""
            gt_val = gt_data.get(normalize_header(key), "")
            is_match = values_match(gt_val, llm_val)
            fields.append(
                {
                    "field": key,
                    "value": llm_val if llm_val else "(empty)",
                    "expected": gt_val if not is_match else None,
                    "status": "correct" if is_match else "wrong",
                }
            )


def generate_index_page(
    pdf_data: list[dict[str, Any]],
    output_path: str | Path,
) -> str:
    """Generate the single-page app index.html with hash routing."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build benchmark data and embed it directly in HTML
    benchmark_data = _build_benchmark_data(pdf_data)
    html = _generate_spa_html(benchmark_data)

    with open(output_path, "w") as f:
        f.write(html)

    return str(output_path)


def _build_benchmark_data(pdf_data: list[dict[str, Any]]) -> dict:
    """Build benchmark data structure for embedding in HTML."""

    def sort_key(d):
        name = d["name"]
        is_trap = "-trap" in name
        base_name = name.replace("-trap", "")
        return (is_trap, base_name)

    pdf_data = sorted(pdf_data, key=sort_key)

    # Collect models with their providers
    model_providers = {}
    for d in pdf_data:
        for r in d["results"]:
            if r.model not in model_providers:
                model_providers[r.model] = r.provider

    # Sort models by provider, then by name
    all_models = sorted(model_providers.keys(), key=lambda m: (model_providers.get(m, ""), m))

    pdfs = []
    reports = {}  # Full report data for each PDF
    for d in pdf_data:
        results = d["results"]
        is_trap = "-trap" in d["name"]
        result_map = {r.model: r for r in results}

        # Build full report data for this PDF first (need it for error counts)
        report = None
        if results:
            report = _build_report_data(d["ground_truth"], results)
            reports[d["name"]] = report

        model_scores = {}
        model_times = {}
        model_correct = {}
        model_totals = {}
        for model in all_models:
            if model in result_map:
                r = result_map[model]
                model_scores[model] = r.accuracy_score
                total_ms = sum(resp.latency_ms for resp in r.responses if resp.latency_ms)
                model_times[model] = total_ms / 1000 if total_ms else None
                # Get correct counts from the report data
                if report:
                    for m in report.get("models", []):
                        if m["name"] == model:
                            model_correct[model] = m.get("correct_count", 0)
                            model_totals[model] = m.get("total_count", 0)
                            break
                    else:
                        model_correct[model] = None
                        model_totals[model] = None
                else:
                    model_correct[model] = None
                    model_totals[model] = None
            else:
                model_scores[model] = None
                model_times[model] = None
                model_correct[model] = None
                model_totals[model] = None

        pdfs.append(
            {
                "name": d["name"],
                "is_trap": is_trap,
                "total_pages": d["ground_truth"].total_pages,
                "scores": model_scores,
                "times": model_times,
                "correct": model_correct,
                "totals": model_totals,
            }
        )

    return {
        "updated_at": datetime.now().isoformat(),
        "models": all_models,
        "model_providers": model_providers,
        "pdfs": pdfs,
        "reports": reports,  # Embedded report data for each PDF
    }


def _generate_spa_html(benchmark_data: dict) -> str:
    """Generate single-page app HTML with hash routing and embedded data."""
    # Embed the benchmark data directly to avoid file:// fetch issues
    embedded_data = json.dumps(benchmark_data)
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDF Extraction Benchmark</title>
    <style>
        :root {
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --text-primary: #c9d1d9;
            --text-secondary: #8b949e;
            --border-color: #30363d;
            --accent-green: #3fb950;
            --accent-red: #f85149;
            --accent-yellow: #d29922;
            --accent-blue: #58a6ff;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
        }

        /* Dashboard styles */
        .dashboard { padding: 20px; }
        .dashboard .container { max-width: 1200px; margin: 0 auto; }
        .dashboard header { text-align: center; padding: 40px 20px; margin-bottom: 30px; }
        .dashboard header h1 { font-size: 2.2rem; margin-bottom: 10px; }
        .dashboard header .subtitle { color: var(--text-secondary); font-size: 1.1rem; }
        .dashboard header .stats { color: var(--accent-blue); font-size: 0.9rem; margin-top: 10px; }
        .timestamp { color: var(--text-secondary); font-size: 0.85rem; margin-top: 5px; }
        .toggle-container {
            margin-top: 15px;
            display: inline-flex;
            align-items: center;
            gap: 10px;
            color: var(--text-secondary);
            font-size: 0.85rem;
        }
        .toggle-switch {
            position: relative;
            width: 44px;
            height: 24px;
            background: var(--bg-tertiary);
            border-radius: 12px;
            cursor: pointer;
            transition: background 0.2s;
            border: 1px solid var(--border-color);
        }
        .toggle-switch.active {
            background: var(--accent-blue);
            border-color: var(--accent-blue);
        }
        .toggle-switch::after {
            content: '';
            position: absolute;
            top: 2px;
            left: 2px;
            width: 18px;
            height: 18px;
            background: white;
            border-radius: 50%;
            transition: transform 0.2s;
        }
        .toggle-switch.active::after {
            transform: translateX(20px);
        }

        .benchmark-table { width: 100%; border-collapse: collapse; background: var(--bg-secondary); border-radius: 8px; overflow: hidden; }
        .benchmark-table th, .benchmark-table td { padding: 14px 18px; text-align: center; border-bottom: 1px solid var(--border-color); }
        .benchmark-table th { background: var(--bg-tertiary); font-weight: 600; }
        .benchmark-table .pdf-name-col { text-align: left; font-weight: 500; }
        .benchmark-table .pdf-name-col a { color: var(--text-primary); text-decoration: none; cursor: pointer; }
        .benchmark-table .pdf-name-col a:hover { color: var(--accent-blue); text-decoration: underline; }
        .benchmark-table tr:hover { background: var(--bg-tertiary); }
        .benchmark-table td.clickable { cursor: pointer; }
        .benchmark-table td.clickable:hover { background: var(--bg-primary); }
        .benchmark-table .provider-border { border-left: 2px solid var(--border-color); }
        .benchmark-table .benchmark-row { background: rgba(88, 166, 255, 0.05); }

        .score-cell { font-weight: 600; }
        .score-cell.good { color: var(--accent-green); }
        .score-cell.warning { color: var(--accent-yellow); }
        .score-cell.bad { color: var(--accent-red); }
        .score-cell.neutral { color: var(--text-secondary); }
        .score-cell.missing { color: var(--text-secondary); opacity: 0.5; }
        .time-display { display: block; font-size: 0.7rem; font-weight: normal; color: var(--text-secondary); opacity: 0.6; margin-top: 2px; }

        /* Score display text with colors */
        .score-pct { display: block; font-size: 1.1rem; font-weight: 700; }
        .score-perfect .score-pct { color: var(--accent-green); }
        .score-good .score-pct { color: #7ee787; }
        .score-ok .score-pct { color: var(--accent-yellow); }
        .score-bad .score-pct { color: var(--accent-red); }
        .score-missing .score-pct, .score-na .score-pct { color: var(--text-secondary); }
        .fields-col { color: var(--text-secondary); font-size: 0.9rem; }

        .legend { margin-top: 20px; display: flex; gap: 25px; justify-content: center; flex-wrap: wrap; }
        .legend-item { display: flex; align-items: center; gap: 8px; color: var(--text-secondary); font-size: 0.85rem; }
        footer { text-align: center; padding: 40px; color: var(--text-secondary); }

        /* Report view styles */
        .report-view { display: none; }
        .report-view.active { display: block; }

        .report-header {
            padding: 20px 30px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            gap: 30px;
        }
        .back-link { color: var(--accent-blue); text-decoration: none; cursor: pointer; }
        .back-link:hover { text-decoration: underline; }
        .report-header h1 { font-size: 1.5rem; margin: 0; }
        .report-header .subtitle { color: var(--text-secondary); font-size: 0.9rem; }

        .prompt-section {
            padding: 10px 30px;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border-color);
        }
        .prompt-section summary { cursor: pointer; color: var(--accent-blue); font-size: 0.9rem; }
        .prompt-text {
            margin-top: 10px;
            padding: 15px;
            background: var(--bg-tertiary);
            border-radius: 6px;
            font-size: 0.85rem;
            white-space: pre-wrap;
            overflow-x: auto;
        }

        .model-selector {
            display: flex;
            gap: 10px;
            padding: 15px 30px;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border-color);
            overflow-x: auto;
        }
        .model-tab {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 10px 20px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .model-tab:hover { border-color: var(--accent-blue); }
        .model-tab.active { background: var(--accent-blue); border-color: var(--accent-blue); }
        .model-tab.active .model-name { color: white; }
        .model-tab.active .model-accuracy { color: rgba(255,255,255,0.9); }
        .model-tab.active .model-time { color: rgba(255,255,255,0.7); }
        .model-name { font-weight: 600; color: var(--text-primary); font-size: 0.9rem; }
        .model-accuracy { font-size: 1.1rem; font-weight: bold; margin-top: 4px; }
        .model-accuracy.good { color: var(--accent-green); }
        .model-accuracy.warning { color: var(--accent-yellow); }
        .model-accuracy.bad { color: var(--accent-red); }
        .model-accuracy.neutral { color: var(--text-secondary); }
        .model-pct { font-size: 0.75rem; color: var(--text-secondary); margin-top: 1px; }
        .model-time { font-size: 0.75rem; color: var(--text-secondary); margin-top: 2px; }
        .model-tab.active .model-pct { color: rgba(255,255,255,0.7); }

        .split-view { display: flex; height: calc(100vh - 200px); }
        .left-panel { overflow-y: auto; padding: 20px 30px; }
        .resize-handle {
            width: 6px;
            background: var(--border-color);
            cursor: col-resize;
            flex-shrink: 0;
            transition: background 0.2s;
        }
        .resize-handle:hover, .resize-handle.dragging { background: var(--accent-blue); }
        .right-panel { flex: 1; overflow-y: auto; background: var(--bg-tertiary); padding: 20px; }

        .page-image-container { margin-bottom: 20px; background: var(--bg-secondary); border-radius: 8px; overflow: hidden; }
        .page-image-header { padding: 10px 15px; background: var(--bg-primary); font-weight: 500; font-size: 0.9rem; color: var(--text-secondary); border-bottom: 1px solid var(--border-color); }
        .page-image { width: 100%; height: auto; display: block; }

        .error-summary { background: var(--bg-tertiary); border-radius: 6px; padding: 15px 20px; margin-bottom: 20px; }
        .error-summary h4 { margin-bottom: 10px; color: var(--accent-red); }
        .error-summary.success h4 { color: var(--accent-green); }
        .error-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
        .error-table th { text-align: left; padding: 8px; background: var(--bg-primary); border-bottom: 1px solid var(--border-color); color: var(--text-secondary); }
        .error-table td { padding: 8px; border-bottom: 1px solid var(--border-color); }
        .err-name { font-family: monospace; color: var(--accent-blue); }
        .err-expected { font-family: monospace; color: var(--accent-green); }
        .err-actual { font-family: monospace; color: var(--accent-red); }
        .trap-passed .err-actual { color: var(--accent-green); }
        .trap-status { text-align: center; font-weight: bold; }
        .trap-passed .trap-status { color: var(--accent-green); }
        .trap-failed .trap-status { color: var(--accent-red); }

        .data-section h4 { margin-bottom: 15px; color: var(--text-secondary); }
        .data-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; margin-bottom: 20px; }
        .data-table td { padding: 10px 12px; border-bottom: 1px solid var(--border-color); }
        .data-table tr.correct td { background: rgba(63, 185, 80, 0.1); }
        .data-table tr.correct .field-value { color: var(--accent-green); }
        .data-table tr.wrong td { background: rgba(248, 81, 73, 0.1); }
        .data-table tr.wrong .field-value { color: var(--accent-red); }
        .field-key { font-weight: 600; color: var(--text-secondary); width: 30%; vertical-align: top; }
        .field-value { font-family: monospace; word-break: break-word; }
        .expected-value { display: block; font-size: 0.8rem; color: var(--accent-green); margin-top: 4px; }

        .extracted-table { margin-top: 20px; }
        .extracted-table h5 { color: var(--text-secondary); margin-bottom: 10px; text-transform: capitalize; }

        .row-count { font-size: 0.85rem; margin-bottom: 10px; padding: 6px 12px; border-radius: 4px; display: inline-block; }
        .row-count-ok { background: rgba(63, 185, 80, 0.15); color: var(--accent-green); }
        .row-count-missing { background: rgba(248, 81, 73, 0.15); color: var(--accent-red); }

        .missing-rows-section { margin-top: 20px; padding-top: 15px; border-top: 1px dashed var(--border-color); }
        .missing-rows-section h6 { color: var(--accent-red); font-size: 0.9rem; margin-bottom: 10px; font-weight: 500; }
        .missing-table { opacity: 0.6; }
        .missing-table tr.missing-row td { background: rgba(248, 81, 73, 0.05); color: var(--text-secondary); }
        .missing-table s { text-decoration: line-through; }

        .result-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
        .result-table th { text-align: left; padding: 10px; background: var(--bg-tertiary); border: 1px solid var(--border-color); color: var(--text-secondary); font-weight: 500; }
        .result-table td { padding: 10px; border: 1px solid var(--border-color); position: relative; }
        .result-table tr.correct td { background: rgba(63, 185, 80, 0.1); }
        .result-table tr.wrong td { background: rgba(248, 81, 73, 0.1); }

        /* Wrong cell styling - bold with hover tooltip */
        .result-table td.cell-wrong {
            color: var(--accent-red);
            font-weight: bold;
        }
        .result-table td.cell-wrong .cell-content {
            cursor: help;
        }
        .result-table td.cell-wrong .tooltip {
            display: none;
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background: var(--bg-primary);
            border: 1px solid var(--accent-green);
            color: var(--accent-green);
            padding: 6px 10px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: normal;
            white-space: nowrap;
            z-index: 100;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }
        .result-table td.cell-wrong .tooltip::after {
            content: '';
            position: absolute;
            top: 100%;
            left: 50%;
            transform: translateX(-50%);
            border: 6px solid transparent;
            border-top-color: var(--accent-green);
        }
        .result-table td.cell-wrong:hover .tooltip {
            display: block;
        }

        .raw-section { margin-top: 20px; border-top: 1px solid var(--border-color); padding-top: 15px; }
        .raw-section summary { cursor: pointer; color: var(--accent-blue); font-weight: 500; margin-bottom: 10px; }
        .raw-response { margin-bottom: 15px; }
        .raw-header { font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 5px; }
        .raw-text { background: var(--bg-tertiary); padding: 15px; border-radius: 6px; font-size: 0.85rem; white-space: pre-wrap; word-break: break-word; overflow-x: auto; max-height: 400px; overflow-y: auto; }
        .no-data { color: var(--text-secondary); font-style: italic; text-align: center; padding: 20px; }
        .page-image img { max-width: 100%; height: auto; border-radius: 4px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); margin-bottom: 20px; }
        .loading { text-align: center; padding: 60px; color: var(--text-secondary); }
    </style>
</head>
<body>
    <div id="app"></div>

    <script>
        // Embedded benchmark data (avoids file:// fetch issues)
        const BENCHMARK_DATA = __BENCHMARK_DATA_PLACEHOLDER__;

        let currentReportData = null;
        let currentModelIndex = 0;
        let showRawCounts = false;  // Toggle state: false = error %, true = raw count

        // Panel resize state
        const DEFAULT_PANEL_WIDTH = 500;
        function getSavedPanelWidth() {
            return parseInt(localStorage.getItem('panelWidth')) || DEFAULT_PANEL_WIDTH;
        }
        function savePanelWidth(width) {
            localStorage.setItem('panelWidth', width);
        }

        // Utilities
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text || '';
            return div.innerHTML;
        }

        function getAccuracyClass(accuracy) {
            if (accuracy === null) return 'neutral';
            if (accuracy >= 0.9) return 'good';
            if (accuracy >= 0.7) return 'warning';
            return 'bad';
        }

        function formatScore(score) {
            if (score === null) return '-';
            if (score < 0) return 'N/A';
            const pct = score * 100;
            // Use 2 decimal places for >98% to show fine differences
            if (pct > 98) return pct.toFixed(2).replace(/\.?0+$/, '') + '%';
            return Math.round(pct) + '%';
        }

        function formatErrorPct(errorRate) {
            // errorRate is 0-1 (0 = no errors, 1 = all errors)
            const pct = errorRate * 100;
            if (pct === 0) return '0%';
            if (pct <= 1) return pct.toFixed(3).replace(/\.?0+$/, '') + '%';
            if (pct <= 2) return pct.toFixed(2).replace(/\.?0+$/, '') + '%';
            if (pct <= 4) return pct.toFixed(1).replace(/\.?0+$/, '') + '%';
            return Math.round(pct) + '%';
        }

        function getScoreClass(score) {
            if (score === null) return 'score-missing';
            if (score < 0) return 'score-na';
            const pct = score * 100;
            // Based on accuracy (higher = better)
            return pct === 100 ? 'score-perfect' : (pct >= 95 ? 'score-good' : (pct >= 90 ? 'score-ok' : 'score-bad'));
        }

        function formatScoreWithCorrect(score, correct, total) {
            if (score === null) return '-';
            if (score < 0) return 'N/A';
            if (showRawCounts && correct !== null && correct !== undefined && total) {
                return `<span class="score-pct">${correct}/${total}</span>`;
            }
            return `<span class="score-pct">${formatScore(score)}</span>`;
        }

        function getCorrectClass(correct, total) {
            if (correct === null || correct === undefined || !total) return 'neutral';
            const pct = correct / total;
            if (pct >= 0.9) return 'good';
            if (pct >= 0.7) return 'warning';
            return 'bad';
        }

        function getErrorClass(errors) {
            if (errors === null || errors === undefined) return 'neutral';
            if (errors === 0) return 'good';
            if (errors <= 2) return 'warning';
            return 'bad';
        }

        function toggleDisplayMode() {
            showRawCounts = !showRawCounts;
            handleRoute();  // Re-render current view
        }

        function formatTime(seconds) {
            if (seconds === null || seconds === undefined) return '';
            if (seconds < 1) return (seconds * 1000).toFixed(0) + 'ms';
            return seconds.toFixed(1) + 's';
        }

        // Router
        function navigate(path) {
            window.location.hash = path;
        }

        function getRoute() {
            const hash = window.location.hash.slice(1);
            if (hash.startsWith('view/')) {
                const rest = hash.slice(5);
                const slashIdx = rest.indexOf('/');
                if (slashIdx > 0) {
                    return { type: 'report', pdf: rest.slice(0, slashIdx), model: rest.slice(slashIdx + 1) };
                }
                return { type: 'report', pdf: rest, model: null };
            }
            return { type: 'dashboard' };
        }

        async function handleRoute() {
            const route = getRoute();
            if (route.type === 'report') {
                await renderReportView(route.pdf, route.model);
            } else {
                await renderDashboard();
            }
        }

        // Dashboard
        async function renderDashboard() {
            const app = document.getElementById('app');

            try {
                const data = BENCHMARK_DATA;

                const totalPdfs = data.pdfs.length;
                const originalPdfs = data.pdfs.filter(p => !p.is_trap).length;
                const benchmarkPdfs = totalPdfs - originalPdfs;
                const updated = new Date(data.updated_at);

                let tableRows = '';
                for (const pdf of data.pdfs) {
                    const displayName = pdf.is_trap ? pdf.name.replace('-trap', '') + ' (traps)' : pdf.name;
                    const rowClass = pdf.is_trap ? 'benchmark-row' : 'original-row';

                    let cells = `<td class="pdf-name-col"><a onclick="navigate('view/${pdf.name}')">${escapeHtml(displayName)}</a></td>`;
                    let lastProvider = null;
                    for (const model of data.models) {
                        const provider = data.model_providers[model] || '';
                        const needsBorder = lastProvider === null || provider !== lastProvider;
                        const score = pdf.scores[model];
                        const time = pdf.times ? pdf.times[model] : null;
                        const correct = pdf.correct ? pdf.correct[model] : null;
                        const total = pdf.totals ? pdf.totals[model] : null;
                        const scoreClass = getScoreClass(score);
                        const scoreHtml = formatScoreWithCorrect(score, correct, total);
                        const timeDisplay = formatTime(time);
                        const timeHtml = timeDisplay ? `<span class="time-display">${timeDisplay}</span>` : '';
                        cells += `<td class="score-cell clickable ${scoreClass}${needsBorder ? ' provider-border' : ''}" onclick="navigate('view/${pdf.name}/${model}')">${scoreHtml}${timeHtml}</td>`;
                        lastProvider = provider;
                    }
                    tableRows += `<tr class="${rowClass}">${cells}</tr>`;
                }

                let headerCells = '<th class="pdf-name-col">Test</th>';
                let lastProvider = null;
                for (const model of data.models) {
                    const provider = data.model_providers[model] || '';
                    const needsBorder = lastProvider === null || provider !== lastProvider;
                    const shortName = model.replace('anthropic/', '').replace('openai/', '');
                    headerCells += `<th class="model-col${needsBorder ? ' provider-border' : ''}">${escapeHtml(shortName)}</th>`;
                    lastProvider = provider;
                }

                app.innerHTML = `
                    <div class="dashboard">
                        <div class="container">
                            <header>
                                <h1>PDF Extraction Benchmark</h1>
                                <p class="subtitle">Comparing LLM vision extraction accuracy</p>
                                <p class="stats">${originalPdfs} original PDFs | ${benchmarkPdfs} trap PDFs | ${data.models.length} models</p>
                                <p class="timestamp">Updated: ${updated.toLocaleString()}</p>
                                <div class="toggle-container">
                                    <span>Raw counts</span>
                                    <div class="toggle-switch ${showRawCounts ? 'active' : ''}" onclick="toggleDisplayMode()"></div>
                                </div>
                            </header>
                            <main>
                                <table class="benchmark-table">
                                    <thead><tr>${headerCells}</tr></thead>
                                    <tbody>${tableRows}</tbody>
                                </table>
                            </main>
                            <footer><p>Generated by Natural PDF Benchmark Tool</p></footer>
                        </div>
                    </div>
                `;
            } catch (error) {
                console.error('Failed to load benchmark data:', error);
                app.innerHTML = '<div class="dashboard"><div class="loading">Error loading data</div></div>';
            }
        }

        // Report View
        function renderReportView(pdfName, modelName) {
            const app = document.getElementById('app');

            // Look up embedded report data
            const reportData = BENCHMARK_DATA.reports[pdfName];
            if (!reportData) {
                app.innerHTML = '<div class="report-view active"><div class="loading">Report not found for: ' + escapeHtml(pdfName) + '</div></div>';
                return;
            }

            currentReportData = reportData;

            // Find model index if specified
            currentModelIndex = 0;
            if (modelName) {
                const idx = reportData.models.findIndex(m => m.name === modelName);
                if (idx >= 0) currentModelIndex = idx;
            }

            renderReportHTML();
        }

        function renderReportHTML() {
            const data = currentReportData;
            const app = document.getElementById('app');

            // Model tabs
            const modelTabs = data.models.map((m, i) => {
                const hasErrors = m.error_count !== null && m.error_count !== undefined;
                const errorClass = hasErrors ? getErrorClass(m.error_count) : getAccuracyClass(m.accuracy);
                const errorDisplay = hasErrors ? (m.error_count === 0 ? '0 errors' : `${m.error_count} error${m.error_count > 1 ? 's' : ''}`) : '';
                return `
                <button class="model-tab ${i === currentModelIndex ? 'active' : ''}" onclick="selectModel(${i})">
                    <span class="model-name">${escapeHtml(m.name)}</span>
                    ${hasErrors
                        ? `<span class="model-accuracy ${errorClass}">${errorDisplay}</span><span class="model-pct">${m.accuracy_display}</span>`
                        : `<span class="model-accuracy ${errorClass}">${m.accuracy_display}</span>`
                    }
                    ${m.total_time ? `<span class="model-time">${m.total_time.toFixed(1)}s</span>` : ''}
                </button>
            `}).join('');

            // Render page images
            const pageImages = data.page_images && data.page_images.length > 0
                ? data.page_images.map((img, i) => `
                    <div class="page-image-container">
                        <div class="page-image-header">Page ${i + 1}</div>
                        <img src="${img}" alt="Page ${i + 1}" class="page-image" loading="lazy" />
                    </div>
                `).join('')
                : '<p class="no-data">No page images available</p>';

            app.innerHTML = `
                <div class="report-view active">
                    <header class="report-header">
                        <a class="back-link" onclick="navigate('')">&larr; Back to Dashboard</a>
                        <h1>${escapeHtml(data.pdf_name)}</h1>
                        <p class="subtitle">${data.total_pages} page(s) | ${data.is_trap ? 'Trap PDF' : 'Original PDF'}</p>
                    </header>

                    <details class="prompt-section">
                        <summary>View Prompt</summary>
                        <pre class="prompt-text">${escapeHtml(data.prompt_template)}</pre>
                    </details>

                    <div class="model-selector">${modelTabs}</div>

                    <main class="split-view" id="split-view">
                        <div class="left-panel" id="left-panel" style="width: ${getSavedPanelWidth()}px;"></div>
                        <div class="resize-handle" id="resize-handle"></div>
                        <div class="right-panel">${pageImages}</div>
                    </main>
                </div>
            `;

            renderModelContent();
            initResizeHandle();
        }

        function initResizeHandle() {
            const handle = document.getElementById('resize-handle');
            const leftPanel = document.getElementById('left-panel');
            const splitView = document.getElementById('split-view');
            if (!handle || !leftPanel || !splitView) return;

            let isDragging = false;

            handle.addEventListener('mousedown', (e) => {
                isDragging = true;
                handle.classList.add('dragging');
                document.body.style.cursor = 'col-resize';
                document.body.style.userSelect = 'none';
                e.preventDefault();
            });

            document.addEventListener('mousemove', (e) => {
                if (!isDragging) return;
                const rect = splitView.getBoundingClientRect();
                const newWidth = Math.max(200, Math.min(e.clientX - rect.left, rect.width - 200));
                leftPanel.style.width = newWidth + 'px';
            });

            document.addEventListener('mouseup', () => {
                if (isDragging) {
                    isDragging = false;
                    handle.classList.remove('dragging');
                    document.body.style.cursor = '';
                    document.body.style.userSelect = '';
                    savePanelWidth(parseInt(leftPanel.style.width));
                }
            });
        }

        function selectModel(index) {
            currentModelIndex = index;

            // Update tab active state
            document.querySelectorAll('.model-tab').forEach((tab, i) => {
                tab.classList.toggle('active', i === index);
            });

            renderModelContent();
        }

        function renderModelContent() {
            const model = currentReportData.models[currentModelIndex];
            const container = document.getElementById('left-panel');
            let html = '';

            // Trap summary (for trap PDFs)
            if (currentReportData.is_trap && model.trap_results && model.trap_results.length > 0) {
                const passed = model.trap_results.filter(t => t.passed).length;
                const total = model.trap_results.length;
                const allPassed = passed === total;
                html += `<div class="error-summary ${allPassed ? 'success' : ''}">
                    <h4>Traps: ${passed}/${total} passed</h4>
                    <table class="error-table">
                        <thead><tr><th>What</th><th>Expected</th><th>Got</th><th>Status</th></tr></thead>
                        <tbody>${model.trap_results.map(t => `
                            <tr class="${t.passed ? 'trap-passed' : 'trap-failed'}">
                                <td class="err-name">${escapeHtml(t.description || t.name)}</td>
                                <td class="err-expected">${escapeHtml(t.expected)}</td>
                                <td class="err-actual">${escapeHtml(t.actual)}</td>
                                <td class="trap-status">${t.passed ? '✓' : '✗'}</td>
                            </tr>
                        `).join('')}</tbody>
                    </table>
                </div>`;
            }

            // Extracted data section
            html += '<div class="data-section"><h4>Extracted Data</h4>';

            // Fields
            if (model.extracted.fields.length > 0) {
                html += '<table class="data-table"><tbody>';
                model.extracted.fields.forEach(f => {
                    const expectedHtml = f.expected
                        ? `<span class="expected-value">expected: ${escapeHtml(f.expected)}</span>`
                        : '';
                    html += `<tr class="${f.status}">
                        <td class="field-key">${escapeHtml(f.field)}</td>
                        <td class="field-value">${escapeHtml(f.value)}${expectedHtml}</td>
                    </tr>`;
                });
                html += '</tbody></table>';
            }

            // Tables
            model.extracted.tables.forEach(table => {
                html += `<div class="extracted-table"><h5>${escapeHtml(table.name)}</h5>`;

                // Row count summary
                if (table.row_count) {
                    const { extracted, expected } = table.row_count;
                    const countClass = extracted === expected ? 'row-count-ok' : 'row-count-missing';
                    html += `<p class="row-count ${countClass}">${extracted} of ${expected} rows extracted</p>`;
                }

                if (table.error) {
                    html += `<p class="no-data">${escapeHtml(table.error)}</p>`;
                } else if (table.rows.length > 0) {
                    html += `<table class="result-table">
                        <thead><tr>${table.headers.map(h => `<th>${escapeHtml(h)}</th>`).join('')}</tr></thead>
                        <tbody>${table.rows.map(row => `
                            <tr class="${row.status}">${row.cells.map(cell => {
                                if (cell.status === 'wrong' && cell.expected) {
                                    return `<td class="cell-wrong">
                                        <span class="cell-content">${escapeHtml(cell.value)}</span>
                                        <span class="tooltip">Expected: ${escapeHtml(cell.expected)}</span>
                                    </td>`;
                                }
                                return `<td>${escapeHtml(cell.value)}</td>`;
                            }).join('')}</tr>
                        `).join('')}</tbody>
                    </table>`;
                } else {
                    html += '<p class="no-data">No data</p>';
                }

                // Missing rows section
                if (table.missing_rows && table.missing_rows.length > 0) {
                    html += `<div class="missing-rows-section">
                        <h6>Missing Rows (${table.missing_rows.length})</h6>
                        <table class="result-table missing-table">
                            <thead><tr>${table.missing_headers.map(h => `<th>${escapeHtml(h)}</th>`).join('')}</tr></thead>
                            <tbody>${table.missing_rows.map(row => `
                                <tr class="missing-row">${row.cells.map(cell =>
                                    `<td><s>${escapeHtml(cell.value)}</s></td>`
                                ).join('')}</tr>
                            `).join('')}</tbody>
                        </table>
                    </div>`;
                }

                html += '</div>';
            });

            if (model.extracted.fields.length === 0 && model.extracted.tables.length === 0) {
                html += '<p class="no-data">No data extracted</p>';
            }

            html += '</div>';

            // Raw output section
            html += `<details class="raw-section">
                <summary>Raw LLM Output</summary>
                ${model.raw_responses.map(r => `
                    <div class="raw-response">
                        <div class="raw-header">Page ${r.page}${r.latency_ms ? ` (${r.latency_ms}ms)` : ''}</div>
                        <pre class="raw-text">${escapeHtml(r.text)}</pre>
                    </div>
                `).join('')}
            </details>`;

            container.innerHTML = html;
        }

        // Initialize
        window.addEventListener('hashchange', handleRoute);
        handleRoute();
    </script>
</body>
</html>"""
    return html.replace("__BENCHMARK_DATA_PLACEHOLDER__", embedded_data)
