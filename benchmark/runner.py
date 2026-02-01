"""
Benchmark Runner

Core orchestration for the benchmark pipeline:
1. prepare - Generate page images and ground truth
2. evaluate - Run LLM evaluations
3. report - Generate HTML reports
4. run - Full pipeline
"""

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

from benchmark.cache import ResponseCache
from benchmark.config import BenchmarkConfig
from benchmark.providers import (
    LLMProvider,
    get_provider,
    parse_csv_response,
    parse_json_response,
    validate_models,
)
from benchmark.schemas import (
    BenchmarkMeta,
    BenchmarkOutput,
    GroundTruth,
    LLMResponse,
    LLMResult,
    ModelSummary,
    PageGroundTruth,
    PDFSummary,
    TrapDefinition,
    TrapResult,
)
from benchmark.utils import (
    detect_data_format,
    find_matching_column,
    normalize_header,
    normalize_text,
    strip_whitespace,
    values_match,
)


class BenchmarkRunner:
    """
    Main runner for benchmark operations.

    Usage:
        runner = BenchmarkRunner(output_dir="benchmark_output")
        runner.prepare("atlanta_schools")
        runner.evaluate("atlanta_schools", models=["gpt-4o"])
        runner.report("atlanta_schools")
    """

    def __init__(
        self,
        output_dir: str = "benchmark_output",
        config: Optional[BenchmarkConfig] = None,
    ):
        self.config = config or BenchmarkConfig.load()
        self.output = BenchmarkOutput(output_dir or self.config.output_dir)
        self.cache = ResponseCache(self.output.cache_dir()) if self.config.cache_enabled else None

    def prepare(
        self,
        pdf_name: str,
        resolution: int = 150,
        limit_pages: Optional[int] = None,
        use_trap: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> GroundTruth:
        """
        Prepare a PDF for benchmarking.

        1. Extract ground truth with Natural PDF
        2. Generate page images for report visualization
        3. Save to output directory

        Args:
            pdf_name: Name of the PDF config (e.g., "atlanta_schools")
            resolution: DPI for page image rendering (default: 150)
            limit_pages: Max pages to process (None = all)
            progress_callback: Called with (current, total) during processing

        Returns:
            GroundTruth object
        """
        from benchmark.configs import get_config

        config = get_config(pdf_name)
        if not config:
            raise ValueError(f"Unknown PDF config: {pdf_name}")

        # Choose original or benchmark PDF
        if use_trap and hasattr(config, "pdf_path_trap"):
            pdf_path = config.pdf_path_trap
            output_suffix = "-trap"
        else:
            pdf_path = config.pdf_path
            output_suffix = ""

        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Adjust output name for benchmark PDFs
        output_name = pdf_name + output_suffix

        import natural_pdf as npdf

        pdf = npdf.PDF(pdf_path)
        total_pages = len(pdf.pages)
        if limit_pages:
            total_pages = min(total_pages, limit_pages)

        # Determine max pages (global config)
        if self.config.max_pages_per_pdf:
            total_pages = min(total_pages, self.config.max_pages_per_pdf)

        # Per-PDF config may also specify max_pages
        if hasattr(config, "max_pages") and config.max_pages:
            total_pages = min(total_pages, config.max_pages)

        ground_truth = GroundTruth(
            pdf_name=output_name,
            pdf_path=pdf_path,
            total_pages=total_pages,
            extracted_at=datetime.now().isoformat(),
        )

        # Store whether this is a benchmark PDF for trap checking
        ground_truth.is_trap = use_trap

        # Track data format - will be determined from first page with data
        detected_format = None

        # Extract ground truth data and render page images
        for page_num in range(total_pages):
            if progress_callback:
                progress_callback(page_num + 1, total_pages)

            page = pdf.pages[page_num]

            # Render page image
            try:
                image_path = self.output.page_image_path(output_name, page_num + 1)
                page.save_image(
                    str(image_path), resolution=resolution, include_highlights=False, labels=False
                )
            except Exception as e:
                logger.warning(f"Failed to render page {page_num + 1} image: {e}")

            # Extract page data
            start_time = time.time()
            try:
                page_result = config.extract_with_natural_pdf(pdf_path, page_num)
                if isinstance(page_result, pd.DataFrame):
                    page_data = page_result.to_dict("records")
                elif isinstance(page_result, dict):
                    page_data = [page_result]  # Wrap dict in list
                elif isinstance(page_result, list):
                    page_data = page_result
                else:
                    page_data = []
            except Exception as e:
                logger.warning(f"Failed to extract page {page_num + 1}: {e}")
                page_data = []

            extraction_time = int((time.time() - start_time) * 1000)

            # Detect data format from first page with data
            if detected_format is None and page_data:
                detected_format = detect_data_format(page_data)

            # Get traps for this page (only for benchmark PDFs)
            page_traps = self._get_page_traps(config, page_num) if use_trap else []

            ground_truth.pages.append(
                PageGroundTruth(
                    page_number=page_num + 1,
                    data=page_data,
                    traps=page_traps,
                    extraction_time_ms=extraction_time,
                )
            )

        pdf.close()

        # Set detected data format
        if detected_format:
            ground_truth.data_format = detected_format

        # Extract aggregate data (all pages)
        try:
            if hasattr(config, "extract_all_pages"):
                all_df = config.extract_all_pages(pdf_path)
                if isinstance(all_df, pd.DataFrame) and not all_df.empty:
                    ground_truth.aggregate_data = all_df.to_dict("records")
        except Exception as e:
            logger.warning(f"Failed to extract aggregate data: {e}")

        # Calculate total extraction time from page times
        ground_truth.total_extraction_time_ms = sum(
            p.extraction_time_ms for p in ground_truth.pages
        )

        # Save ground truth
        ground_truth.save(self.output.ground_truth_json_path(output_name))

        # Also save as CSV for easy viewing
        if ground_truth.aggregate_data:
            df = pd.DataFrame(ground_truth.aggregate_data)
            df.to_csv(self.output.ground_truth_csv_path(output_name), index=False)

        # Update meta
        self._update_meta_after_prepare(output_name, ground_truth)

        return ground_truth

    def _get_page_traps(self, config, page_num: int) -> list[TrapDefinition]:
        """Extract trap definitions for a specific page."""
        traps = []

        if not hasattr(config, "comparison_fields"):
            return traps

        for cf in config.comparison_fields:
            # For now, assume all traps are on page 1
            # TODO: Add page-specific trap definitions to configs
            if page_num == 0:
                traps.append(
                    TrapDefinition(
                        name=cf["name"],
                        description=cf.get("description", ""),
                        page=1,
                        expected_value=cf.get("expected_trap", cf.get("expected_original", "")),
                        trap_for=cf.get("expected_original", ""),
                        category=cf.get("category", "unknown"),
                    )
                )

        return traps

    def evaluate(
        self,
        pdf_name: str,
        models: Optional[list[str]] = None,
        limit_pages: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> list[LLMResult]:
        """
        Evaluate a PDF with LLM models.

        Args:
            pdf_name: Name of the PDF config
            models: List of models to evaluate (uses config defaults if None)
            limit_pages: Max pages to evaluate
            progress_callback: Called with (current, total, status)

        Returns:
            List of LLMResult objects
        """
        from benchmark.configs import get_config

        # Strip -trap suffix to get the base config name
        base_pdf_name = pdf_name.replace("-trap", "")
        config = get_config(base_pdf_name)
        if not config:
            raise ValueError(f"Unknown PDF config: {base_pdf_name}")

        # Validate API keys before starting (fail fast)
        models = models or self.config.default_models
        validate_models(models)

        # Load ground truth
        gt_path = self.output.ground_truth_json_path(pdf_name)
        if not gt_path.exists():
            raise RuntimeError(f"Ground truth not found. Run 'prepare' first for: {pdf_name}")

        ground_truth = GroundTruth.load(gt_path)

        # Determine pages to evaluate
        pages_to_eval = ground_truth.pages[:limit_pages] if limit_pages else ground_truth.pages

        # Run model evaluations in parallel with live progress display
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from benchmark.progress import SinglePDFProgressDisplay, Status

        progress = SinglePDFProgressDisplay(models, pdf_name=pdf_name)

        def evaluate_and_save(model: str) -> LLMResult:
            progress.update(model, Status.RUNNING)
            start_time = time.time()

            result = self._evaluate_model(
                pdf_name=pdf_name,
                config=config,
                ground_truth=ground_truth,
                pages=pages_to_eval,
                model=model,
                progress_callback=None,
            )
            # Save result immediately
            result.save(self.output.llm_result_path(pdf_name, model))

            elapsed = time.time() - start_time
            progress.update(
                model,
                Status.COMPLETED,
                score=result.accuracy_score,
                time_seconds=elapsed,
            )
            return result

        results = []
        max_workers = min(len(models), self.config.max_concurrent)

        progress.start()
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_model = {
                    executor.submit(evaluate_and_save, model): model for model in models
                }

                for future in as_completed(future_to_model):
                    model = future_to_model[future]
                    try:
                        result = future.result()
                        results.append(result)
                        if progress_callback:
                            progress_callback(len(results), len(models), f"Completed {model}")
                    except Exception as e:
                        progress.update(model, Status.ERROR, error=str(e))
                        logger.error(f"Model {model} failed: {e}")
        finally:
            progress.finish()

        # Update meta
        self._update_meta_after_evaluate(pdf_name, results)

        return results

    def _evaluate_model(
        self,
        pdf_name: str,
        config,
        ground_truth: GroundTruth,
        pages: list[PageGroundTruth],
        model: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> LLMResult:
        """Evaluate a single model on the entire PDF."""
        provider = get_provider(model)
        prompt = config.prompt
        pdf_path = ground_truth.pdf_path

        result = LLMResult(
            pdf_name=pdf_name,
            model=model,
            provider=provider.provider_name,
            evaluated_at=datetime.now().isoformat(),
            prompt_template=prompt,
            total_pages=len(pages),
        )

        total_tokens = 0

        if progress_callback:
            progress_callback(0, 1, f"Evaluating PDF with {model}")

        # Check cache
        cached = None
        if self.cache:
            cached = self.cache.get(pdf_name, pdf_path, prompt, model, provider.provider_name)

        if cached:
            # Create a single response for the entire PDF
            response = LLMResponse(
                page_number=1,  # PDF-level response
                prompt=prompt,
                raw_response=cached.raw_response,
                parsed_data=cached.parsed_data,
                tokens_input=cached.tokens_input,
                tokens_output=cached.tokens_output,
                latency_ms=cached.latency_ms,
                cached=True,
            )
            result.responses.append(response)
            total_tokens = response.tokens_input + response.tokens_output
        else:
            # Make single API call for entire PDF
            try:
                # Truncate PDF to only the pages being evaluated
                api_response = provider.call(
                    prompt, pdf_path, model, max_pages=ground_truth.total_pages
                )

                # Parse response - try JSON first since CSV parser incorrectly parses JSON
                parsed = parse_json_response(api_response.raw_text)
                if not parsed:
                    parsed = parse_csv_response(api_response.raw_text)

                response = LLMResponse(
                    page_number=1,  # PDF-level response
                    prompt=prompt,
                    raw_response=api_response.raw_text,
                    parsed_data=parsed,
                    tokens_input=api_response.tokens_input,
                    tokens_output=api_response.tokens_output,
                    latency_ms=api_response.latency_ms,
                    cached=False,
                )

                # Cache the response
                if self.cache:
                    self.cache.set(
                        pdf_name,
                        pdf_path,
                        prompt,
                        model,
                        provider.provider_name,
                        api_response.raw_text,
                        parsed,
                        api_response.tokens_input,
                        api_response.tokens_output,
                        api_response.latency_ms,
                    )

                result.responses.append(response)
                total_tokens = response.tokens_input + response.tokens_output

            except Exception as e:
                logger.error(f"Error evaluating PDF with {model}: {e}")
                response = LLMResponse(
                    page_number=1,
                    prompt=prompt,
                    raw_response=f"ERROR: {e}",
                    parsed_data=[],
                    cached=False,
                )
                result.responses.append(response)

        if progress_callback:
            progress_callback(1, 1, f"Completed {model}")

        result.total_tokens = total_tokens

        # Calculate accuracy by comparing LLM output to ground truth
        result.accuracy_score = self._calculate_accuracy(ground_truth, result)

        # Check traps (only for benchmark PDFs) - separate from accuracy
        if ground_truth.is_trap:
            result.trap_results = self._check_traps(ground_truth, result)
        else:
            result.trap_results = []

        return result

    def _calculate_accuracy(self, ground_truth: GroundTruth, result: LLMResult) -> float:
        """Calculate accuracy by comparing LLM output to ground truth."""
        from benchmark.providers import parse_csv_response, parse_json_response

        total_fields = 0
        correct_fields = 0
        got_any_llm_data = False  # Track if LLM returned any parseable data

        # Use explicit data_format from ground truth
        is_tabular = ground_truth.data_format == "tabular"

        # With native PDF support, we have a single response for the entire PDF
        # Get LLM data from the first (and only) response
        if not result.responses:
            return -1.0

        pdf_response = result.responses[0]

        # Parse LLM response
        llm_data = parse_json_response(pdf_response.raw_response)
        if not llm_data:
            llm_data = parse_csv_response(pdf_response.raw_response)

        if not llm_data:
            return -1.0  # No parseable data

        got_any_llm_data = True

        # Collect all ground truth data
        all_gt_data = []
        for page in ground_truth.pages:
            if page.data:
                all_gt_data.extend(page.data)

        if not all_gt_data:
            return -1.0

        if is_tabular:
            # Tabular data: compare row by row
            llm_rows = llm_data if isinstance(llm_data, list) else [llm_data]
            llm_headers = list(llm_rows[0].keys()) if llm_rows else []

            for i, gt_row in enumerate(all_gt_data):
                if i < len(llm_rows):
                    llm_row = llm_rows[i]
                    for col, exp_val in gt_row.items():
                        total_fields += 1
                        # Find matching column using normalized header matching
                        matching_col = find_matching_column(col, llm_headers)
                        llm_val = llm_row.get(matching_col, "") if matching_col else ""
                        if values_match(str(exp_val), str(llm_val)):
                            correct_fields += 1
                else:
                    # LLM is missing this row
                    total_fields += len(gt_row)
        else:
            # Structured data (like 01-practice): single dict with nested fields
            gt_data = all_gt_data[0] if all_gt_data else {}
            llm_flat = self._flatten_llm_data(llm_data)

            for key, expected_value in gt_data.items():
                if key == "violations_table":
                    # Compare table rows
                    llm_table = llm_flat.get("violations", [])
                    if isinstance(expected_value, list):
                        for i, expected_row in enumerate(expected_value):
                            if i < len(llm_table):
                                llm_row = llm_table[i]
                                llm_headers = list(llm_row.keys())
                                for col, exp_val in expected_row.items():
                                    total_fields += 1
                                    matching_col = find_matching_column(col, llm_headers)
                                    llm_val = llm_row.get(matching_col, "") if matching_col else ""
                                    if values_match(str(exp_val), str(llm_val)):
                                        correct_fields += 1
                            else:
                                total_fields += len(expected_row)
                elif key == "summary":
                    # Skip summary - too long to compare character by character
                    pass
                else:
                    total_fields += 1
                    llm_val = llm_flat.get(key, "")
                    if values_match(str(expected_value), str(llm_val)):
                        correct_fields += 1

        if total_fields == 0 or not got_any_llm_data:
            return -1.0  # No data to compare - use -1 to indicate N/A

        return correct_fields / total_fields

    def _flatten_llm_data(self, llm_data: list) -> dict:
        """Flatten LLM parsed data into a simple dict for comparison."""
        result = {}
        for item in llm_data:
            if isinstance(item, dict):
                for key, value in item.items():
                    if isinstance(value, dict):
                        # Nested object (like form_fields)
                        for k, v in value.items():
                            result[k] = v
                    elif isinstance(value, list):
                        # Array (like violations)
                        result[key] = value
                    else:
                        result[key] = value
        return result

    def _check_traps(self, ground_truth: GroundTruth, result: LLMResult) -> list[TrapResult]:
        """Check if LLM output matches trap values."""
        trap_results = []

        # With native PDF support, we have a single response for the entire PDF
        if not result.responses:
            return trap_results

        pdf_response = result.responses[0]

        for page in ground_truth.pages:
            for trap in page.traps:
                # Search for trap value in LLM output
                llm_value = self._find_trap_value(trap, pdf_response.parsed_data)

                # Check if trap value appears in output (not exact match)
                # This handles both field traps ("site" = "X") and substring traps ("fertiliser" in summary)
                # Use whitespace-stripped comparison so "cat\ndog" matches "cat dog"
                if llm_value:
                    passed = strip_whitespace(trap.expected_value) in strip_whitespace(llm_value)
                    # For display, show the relevant portion if it's a long string
                    if len(llm_value) > 50 and passed:
                        # Find and show context around the match
                        idx = llm_value.find(trap.expected_value)
                        start = max(0, idx - 10)
                        end = min(len(llm_value), idx + len(trap.expected_value) + 10)
                        display_value = (
                            ("..." if start > 0 else "")
                            + llm_value[start:end]
                            + ("..." if end < len(llm_value) else "")
                        )
                    else:
                        display_value = llm_value
                else:
                    passed = False
                    display_value = "(not found)"

                trap_results.append(
                    TrapResult(
                        trap_name=trap.name,
                        expected=trap.expected_value,
                        actual=display_value,
                        passed=passed,
                        category=trap.category,
                        page=page.page_number,
                        description=trap.description,
                    )
                )

        return trap_results

    def _find_trap_value(self, trap: TrapDefinition, parsed_data: list[dict]) -> Optional[str]:
        """Find the value for a trap in parsed LLM output."""

        def search_value(obj, target_values: list[str]) -> Optional[str]:
            """Recursively search for a value containing any target string."""
            if isinstance(obj, str):
                for target in target_values:
                    if target and target in obj:
                        return obj
            elif isinstance(obj, dict):
                for v in obj.values():
                    result = search_value(v, target_values)
                    if result:
                        return result
            elif isinstance(obj, list):
                for item in obj:
                    result = search_value(item, target_values)
                    if result:
                        return result
            return None

        # Search for expected value or trap_for value
        targets = [trap.expected_value]
        if trap.trap_for:
            targets.append(trap.trap_for)

        for row in parsed_data:
            result = search_value(row, targets)
            if result:
                return result
        return None

    def report(
        self,
        pdf_name: str,
        models: Optional[list[str]] = None,
        open_browser: bool = True,
    ) -> str:
        """
        Generate report data JSON for a PDF.

        Args:
            pdf_name: Name of the PDF
            models: Models to include (None = all available)
            open_browser: Open report in browser after generation

        Returns:
            Path to generated JSON data
        """
        from benchmark.report_generator import generate_pdf_report

        # Load ground truth
        gt_path = self.output.ground_truth_json_path(pdf_name)
        if not gt_path.exists():
            raise RuntimeError(f"Ground truth not found. Run 'prepare' first for: {pdf_name}")

        ground_truth = GroundTruth.load(gt_path)

        # Load LLM results
        available_models = self.output.list_models(pdf_name)
        if models:
            models = [m for m in models if m in available_models]
        else:
            models = available_models

        llm_results = []
        for model in models:
            result_path = self.output.llm_result_path(pdf_name, model)
            if result_path.exists():
                result = LLMResult.load(result_path)
                # Re-parse responses from raw text (fixes corrupted cached parsed_data)
                # Try JSON first since CSV parser incorrectly parses JSON
                from benchmark.providers import parse_csv_response, parse_json_response

                for response in result.responses:
                    parsed = parse_json_response(response.raw_response)
                    if not parsed:
                        parsed = parse_csv_response(response.raw_response)
                    response.parsed_data = parsed
                # Recompute trap_results with current logic
                if ground_truth.is_trap:
                    result.trap_results = self._check_traps(ground_truth, result)
                llm_results.append(result)

        # Generate report JSON data (no HTML file)
        json_path = generate_pdf_report(
            ground_truth=ground_truth,
            llm_results=llm_results,
            output_path=self.output.report_path(pdf_name),  # Used to determine directory
        )

        # Regenerate index.html with updated embedded data
        self.generate_index(open_browser=False)

        # Open single-page app with hash route
        if open_browser and self.config.open_report_after_generation:
            import webbrowser

            abs_path = self.output.index_path().resolve()
            webbrowser.open(f"file://{abs_path}#view/{pdf_name}")

        return str(json_path)

    def generate_index(self, open_browser: bool = True) -> str:
        """
        Generate index.html dashboard for all PDFs.

        Returns:
            Path to generated index
        """
        from benchmark.report_generator import generate_index_page

        pdfs = self.output.list_pdfs()

        # Gather data for each PDF
        pdf_data = []
        for pdf_name in pdfs:
            gt_path = self.output.ground_truth_json_path(pdf_name)
            if gt_path.exists():
                ground_truth = GroundTruth.load(gt_path)

                models = self.output.list_models(pdf_name)
                model_results = []
                for model in models:
                    result_path = self.output.llm_result_path(pdf_name, model)
                    if result_path.exists():
                        result = LLMResult.load(result_path)
                        # Re-parse responses from raw text (fixes corrupted cached parsed_data)
                        # Try JSON first since CSV parser incorrectly parses JSON
                        from benchmark.providers import parse_csv_response, parse_json_response

                        for response in result.responses:
                            parsed = parse_json_response(response.raw_response)
                            if not parsed:
                                parsed = parse_csv_response(response.raw_response)
                            response.parsed_data = parsed
                        # Recompute trap_results with current logic
                        if ground_truth.is_trap:
                            result.trap_results = self._check_traps(ground_truth, result)
                        model_results.append(result)

                pdf_data.append(
                    {
                        "name": pdf_name,
                        "ground_truth": ground_truth,
                        "results": model_results,
                        "report_path": f"{pdf_name}/report.html",
                    }
                )

        # Generate index
        index_path = generate_index_page(
            pdf_data=pdf_data,
            output_path=self.output.index_path(),
        )

        if open_browser and self.config.open_report_after_generation:
            import webbrowser

            abs_path = Path(index_path).resolve()
            webbrowser.open(f"file://{abs_path}")

        return str(index_path)

    def run(
        self,
        pdf_names: Optional[list[str]] = None,
        models: Optional[list[str]] = None,
        limit_pages: Optional[int] = None,
        include_traps: bool = True,
        skip_existing: bool = False,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> str:
        """
        Run complete benchmark pipeline with parallel processing.

        Args:
            pdf_names: PDFs to benchmark (None = all configured)
            models: Models to evaluate
            limit_pages: Max pages per PDF
            include_traps: Whether to include trap PDF variants (default: True)
            skip_existing: Skip PDF+model combinations that already have results
            progress_callback: Called with (pdf_name, current, total)

        Returns:
            Path to index.html
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from benchmark.configs import ALL_CONFIGS, get_config
        from benchmark.progress import ProgressDisplay, Status

        base_pdf_names = pdf_names or list(ALL_CONFIGS.keys())
        models = models or self.config.default_models

        # Expand to include trap variants if requested
        pdf_names_expanded = []
        skipped_traps = []
        for pdf_name in base_pdf_names:
            pdf_names_expanded.append(pdf_name)
            if include_traps:
                config = get_config(pdf_name)
                if config and hasattr(config, "pdf_path_trap") and config.pdf_path_trap:
                    trap_path = Path(config.pdf_path_trap)
                    if trap_path.exists():
                        pdf_names_expanded.append(pdf_name + "-trap")
                    else:
                        skipped_traps.append((pdf_name, config.pdf_path_trap))

        if skipped_traps:
            print(f"\nNote: Skipping {len(skipped_traps)} missing trap PDF(s):")
            for name, path in skipped_traps:
                print(f"  - {name}: {path} not found")

        pdf_names = pdf_names_expanded

        # Validate API keys before starting (fail fast)
        validate_models(models)

        logger.info(f"Benchmarking {len(pdf_names)} PDFs with {len(models)} models")

        # Create progress display for all PDF × model combinations
        # Show slightly more than max_concurrent to give context
        progress = ProgressDisplay(pdf_names, models, max_display=self.config.max_concurrent + 4)
        progress.start()

        # Track prepared PDFs and their ground truths
        prepared_pdfs: dict[str, GroundTruth] = {}
        results_by_pdf: dict[str, list[LLMResult]] = {pdf: [] for pdf in pdf_names}

        cancelled = False
        try:
            # Phase 1: Prepare all PDFs in parallel
            def prepare_pdf(pdf_name: str) -> tuple[str, Optional[GroundTruth]]:
                progress.update_pdf(pdf_name, Status.PREPARING)
                try:
                    # Handle trap variants: "01-practice-trap" -> prepare("01-practice", use_trap=True)
                    if pdf_name.endswith("-trap"):
                        base_name = pdf_name[:-5]  # Remove "-trap" suffix
                        gt = self.prepare(base_name, limit_pages=limit_pages, use_trap=True)
                    else:
                        gt = self.prepare(pdf_name, limit_pages=limit_pages, use_trap=False)
                    return pdf_name, gt
                except Exception as e:
                    # Mark all models for this PDF as error
                    for model in models:
                        progress.update(pdf_name, model, Status.ERROR, error=str(e))
                    return pdf_name, None

            # Use serial processing for preparation to avoid PDF library segfaults
            with ThreadPoolExecutor(max_workers=1) as executor:
                futures = {executor.submit(prepare_pdf, pdf): pdf for pdf in pdf_names}
                try:
                    for future in as_completed(futures):
                        pdf_name, gt = future.result()
                        if gt:
                            prepared_pdfs[pdf_name] = gt
                except KeyboardInterrupt:
                    logger.info("Cancelling... (waiting for in-progress tasks)")
                    for f in futures:
                        f.cancel()
                    cancelled = True

            if cancelled:
                raise KeyboardInterrupt

            # Phase 2: Evaluate all PDF × model combinations in parallel
            def evaluate_task(pdf_name: str, model: str) -> tuple[str, str, Optional[LLMResult]]:
                if pdf_name not in prepared_pdfs:
                    return pdf_name, model, None

                progress.update(pdf_name, model, Status.RUNNING)
                start_time = time.time()

                try:
                    from benchmark.configs import get_config

                    # Handle trap variants: use base name to get config
                    config_name = pdf_name[:-5] if pdf_name.endswith("-trap") else pdf_name
                    config = get_config(config_name)
                    ground_truth = prepared_pdfs[pdf_name]
                    pages = ground_truth.pages[:limit_pages] if limit_pages else ground_truth.pages

                    result = self._evaluate_model(
                        pdf_name=pdf_name,
                        config=config,
                        ground_truth=ground_truth,
                        pages=pages,
                        model=model,
                    )
                    result.save(self.output.llm_result_path(pdf_name, model))

                    elapsed = time.time() - start_time
                    progress.update(
                        pdf_name,
                        model,
                        Status.COMPLETED,
                        score=result.accuracy_score,
                        time_seconds=elapsed,
                    )
                    return pdf_name, model, result

                except Exception as e:
                    progress.update(pdf_name, model, Status.ERROR, error=str(e))
                    return pdf_name, model, None

            # Group tasks by provider, then interleave (round-robin)
            # This ensures concurrent tasks are spread across providers
            from collections import defaultdict

            tasks_by_provider: dict[str, list[tuple[str, str]]] = defaultdict(list)
            skipped_count = 0

            for pdf in pdf_names:
                if pdf not in prepared_pdfs:
                    continue
                for model in models:
                    # Skip if result already exists and skip_existing is True
                    if skip_existing:
                        result_path = self.output.llm_result_path(pdf, model)
                        if result_path.exists():
                            progress.update(pdf, model, Status.COMPLETED, score=None)
                            skipped_count += 1
                            continue
                    provider = get_provider(model).provider_name
                    tasks_by_provider[provider].append((pdf, model))

            if skipped_count > 0:
                logger.info(f"Skipped {skipped_count} existing results")

            # Interleave: take one from each provider in turn
            eval_tasks = []
            provider_queues = list(tasks_by_provider.values())
            while any(provider_queues):
                for queue in provider_queues:
                    if queue:
                        eval_tasks.append(queue.pop(0))

            if not eval_tasks:
                logger.info("All tasks skipped - nothing to evaluate")
            else:
                max_workers = min(len(eval_tasks), self.config.max_concurrent)

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(evaluate_task, pdf, model): (pdf, model)
                        for pdf, model in eval_tasks
                    }
                    try:
                        for future in as_completed(futures):
                            pdf_name, model, result = future.result()
                            if result:
                                results_by_pdf[pdf_name].append(result)
                    except KeyboardInterrupt:
                        logger.info("Cancelling... (waiting for in-progress tasks)")
                        for f in futures:
                            f.cancel()
                        cancelled = True

        except KeyboardInterrupt:
            cancelled = True
        finally:
            progress.finish()

        if cancelled:
            logger.info("Benchmark cancelled.")
            return ""

        # Phase 3: Generate reports (quick, do sequentially)
        logger.info("Generating reports...")
        for pdf_name in pdf_names:
            if pdf_name in prepared_pdfs:
                try:
                    self._update_meta_after_evaluate(pdf_name, results_by_pdf[pdf_name])
                    self.report(pdf_name, models=models, open_browser=False)
                except Exception as e:
                    logger.warning(f"Failed to generate report for {pdf_name}: {e}")

        # Generate index
        logger.info("Generating index dashboard...")
        return self.generate_index(open_browser=True)

    def _update_meta_after_prepare(self, pdf_name: str, ground_truth: GroundTruth) -> None:
        """Update meta.json after preparing a PDF."""
        meta = self.output.get_or_create_meta()

        # Update or add PDF summary
        total_traps = sum(len(p.traps) for p in ground_truth.pages)
        pdf_summary = PDFSummary(
            pdf_name=pdf_name,
            total_pages=ground_truth.total_pages,
            total_traps=total_traps,
            models_evaluated=[],
            best_model="",
            best_accuracy=0.0,
        )

        # Replace existing or add new
        meta.pdfs = [p for p in meta.pdfs if p.pdf_name != pdf_name]
        meta.pdfs.append(pdf_summary)

        self.output.save_meta(meta)

    def _update_meta_after_evaluate(self, pdf_name: str, results: list[LLMResult]) -> None:
        """Update meta.json after evaluation."""
        meta = self.output.get_or_create_meta()

        # Find PDF summary
        pdf_summary = next((p for p in meta.pdfs if p.pdf_name == pdf_name), None)
        if pdf_summary:
            pdf_summary.models_evaluated = [r.model for r in results]
            if results:
                best = max(results, key=lambda r: r.accuracy_score)
                pdf_summary.best_model = best.model
                pdf_summary.best_accuracy = best.accuracy_score

        self.output.save_meta(meta)
