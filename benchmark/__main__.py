#!/usr/bin/env python3
"""
Natural PDF LLM Extraction Benchmark CLI

Usage:
    python -m benchmark prepare <pdf_name>              # Generate ground truth
    python -m benchmark evaluate <pdf_name> --models    # Run LLM evaluations (native PDF)
    python -m benchmark report <pdf_name>               # Generate HTML report
    python -m benchmark run [--models MODEL1,MODEL2]    # Run full pipeline
    python -m benchmark list                            # List available PDF configs
    python -m benchmark init                            # Create sample config file
"""

# Load .env file if present
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import argparse
import sys
from pathlib import Path


def cmd_prepare(args):
    """Generate ground truth for a PDF."""
    from benchmark.runner import BenchmarkRunner

    runner = BenchmarkRunner(output_dir=args.output)

    # Determine what to prepare (both by default, --no-traps to skip traps)
    variants = [("original", False)]
    if not args.no_traps:
        variants.append(("trap", True))

    for variant_name, use_trap in variants:
        suffix = "-trap" if use_trap else ""
        output_name = args.pdf_name + suffix

        print(f"\nPreparing: {args.pdf_name} ({variant_name})")
        print(f"Output: {args.output}")
        if args.limit_pages:
            print(f"Limit: {args.limit_pages} pages")
        print()

        def progress(current, total):
            print(f"  Processing page {current}/{total}...", end="\r")

        try:
            ground_truth = runner.prepare(
                pdf_name=args.pdf_name,
                limit_pages=args.limit_pages,
                use_trap=use_trap,
                progress_callback=progress,
            )
            print()
            print(f"\nPrepared {ground_truth.total_pages} pages")
            print(f"Ground truth saved to: {runner.output.ground_truth_json_path(output_name)}")
        except FileNotFoundError as e:
            if use_trap:
                print(f"\n  Skipping trap variant: {e}")
            else:
                print(f"\nError: {e}")
                sys.exit(1)
        except Exception as e:
            print(f"\nError: {e}")
            sys.exit(1)


def cmd_evaluate(args):
    """Run LLM evaluation on a PDF."""
    from benchmark.runner import BenchmarkRunner

    runner = BenchmarkRunner(output_dir=args.output)

    models = args.models.split(",") if args.models else None

    # Determine what to evaluate (both by default, --no-traps to skip traps)
    variants = [("original", "")]
    if not args.no_traps:
        variants.append(("trap", "-trap"))

    for variant_name, suffix in variants:
        pdf_key = args.pdf_name + suffix

        # Check if ground truth exists
        gt_path = runner.output.ground_truth_json_path(pdf_key)
        if not gt_path.exists():
            if suffix:
                print(f"\nSkipping {variant_name}: not prepared (run 'prepare' first)")
                continue
            else:
                print(f"\nError: Ground truth not found for {pdf_key}. Run 'prepare' first.")
                sys.exit(1)

        print(f"\nEvaluating: {args.pdf_name} ({variant_name})")
        print(f"Models: {models or 'default'}")
        if args.limit_pages:
            print(f"Limit: {args.limit_pages} pages")
        print()

        def progress(current, total, status):
            print(f"  {status} ({current}/{total})", end="\r")

        try:
            results = runner.evaluate(
                pdf_name=pdf_key,
                models=models,
                limit_pages=args.limit_pages,
                progress_callback=progress,
            )

            print()
            print("\nResults:")
            for result in results:
                if result.trap_results:
                    passed = sum(1 for t in result.trap_results if t.passed)
                    total_traps = len(result.trap_results)
                    trap_info = f"{passed}/{total_traps} traps"
                else:
                    trap_info = "no traps (original PDF)"
                print(f"  {result.model}: {trap_info}")
                print(f"    Saved to: {runner.output.llm_result_path(pdf_key, result.model)}")
        except Exception as e:
            print(f"\nError: {e}")
            sys.exit(1)


def cmd_report(args):
    """Generate HTML report for a PDF."""
    from benchmark.runner import BenchmarkRunner

    runner = BenchmarkRunner(output_dir=args.output)

    models = args.models.split(",") if args.models else None

    print(f"\nGenerating report: {args.pdf_name}")
    print()

    try:
        report_path = runner.report(
            pdf_name=args.pdf_name,
            models=models,
            open_browser=not args.no_open,
        )
        print(f"Report saved to: {report_path}")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


def cmd_run(args):
    """Run complete benchmark pipeline."""
    from benchmark.runner import BenchmarkRunner

    runner = BenchmarkRunner(output_dir=args.output)

    pdfs = args.pdfs.split(",") if args.pdfs else None
    models = args.models.split(",") if args.models else None

    print("\n" + "=" * 60)
    print("NATURAL PDF BENCHMARK - FULL RUN")
    print("=" * 60)
    print(f"PDFs: {pdfs or 'all configured'}")
    print(f"Models: {models or 'default'}")
    if args.limit_pages:
        print(f"Page limit: {args.limit_pages}")
    print()

    # Override cache setting if --cache flag is used
    if args.cache:
        runner.config.cache_enabled = True
        from benchmark.cache import ResponseCache

        runner.cache = ResponseCache(runner.output.cache_dir())

    try:
        index_path = runner.run(
            pdf_names=pdfs,
            models=models,
            limit_pages=args.limit_pages,
            include_traps=not args.no_traps,
            skip_existing=args.skip_existing,
        )

        if index_path:
            print("\n" + "=" * 60)
            print("BENCHMARK COMPLETE")
            print("=" * 60)
            print(f"\nDashboard: {index_path}")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


def cmd_index(args):
    """Generate index dashboard."""
    from benchmark.runner import BenchmarkRunner

    runner = BenchmarkRunner(output_dir=args.output)

    print("\nGenerating index dashboard...")

    try:
        index_path = runner.generate_index(open_browser=not args.no_open)
        print(f"Dashboard saved to: {index_path}")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


def cmd_list(args):
    """List available PDF configurations."""
    from benchmark.configs import ALL_CONFIGS

    print("\nAvailable PDF configurations:")
    print("-" * 60)

    for name, config_class in ALL_CONFIGS.items():
        config = config_class()
        print(f"\n  {name}")
        print(f"    Description: {config.description}")
        print(f"    PDF: {config.pdf_path}")
        if hasattr(config, "pdf_path_trap"):
            print(f"    Trap PDF: {config.pdf_path_trap}")
        if hasattr(config, "comparison_fields"):
            print(f"    Traps: {len(config.comparison_fields)}")


def cmd_init(args):
    """Create sample configuration file."""
    from benchmark.config import create_sample_config

    path = args.path or "benchmark.json"
    create_sample_config(path)
    print(f"\nSample config created: {path}")
    print("\nEdit this file to configure API keys and settings.")
    print("Alternatively, set environment variables:")
    print("  - OPENAI_API_KEY")
    print("  - GOOGLE_API_KEY")
    print("  - OPENROUTER_API_KEY")


def cmd_cache(args):
    """Manage the response cache."""
    from benchmark.cache import ResponseCache
    from benchmark.schemas import BenchmarkOutput

    output = BenchmarkOutput(args.output)
    cache = ResponseCache(output.cache_dir())

    if args.action == "stats":
        stats = cache.stats()
        size_mb = cache.size_bytes() / (1024 * 1024)
        print("\nCache Statistics:")
        print(f"  Entries: {stats['total_entries']}")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  Session hits: {stats['hits']}")
        print(f"  Session misses: {stats['misses']}")
        print(f"  Hit rate: {stats['hit_rate']:.0%}")

    elif args.action == "clear":
        count = cache.clear()
        print(f"\nCleared {count} cache entries.")

    elif args.action == "list":
        entries = cache.list_entries(model=args.model)
        print(f"\nCached responses ({len(entries)}):")
        for entry in entries[:20]:
            print(f"  {entry.model}: {entry.pdf_path}")
        if len(entries) > 20:
            print(f"  ... and {len(entries) - 20} more")


def cmd_demo(args):
    """Run a demo with sample data (no API calls)."""
    from benchmark.runner import BenchmarkRunner

    runner = BenchmarkRunner(output_dir=args.output)

    print("\n" + "=" * 60)
    print("BENCHMARK DEMO - No API calls required")
    print("=" * 60)

    # Prepare a PDF
    print("\n[1/2] Preparing ground truth for 01-practice...")
    try:
        runner.prepare("01-practice", limit_pages=1)
    except Exception as e:
        print(f"  Warning: {e}")

    # Generate index with whatever we have
    print("\n[2/2] Generating dashboard...")
    try:
        runner.generate_index(open_browser=not args.no_open)
    except Exception as e:
        print(f"  Warning: {e}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {args.output}")
    print(
        "Run 'python -m benchmark evaluate 01-practice --models gpt-4o' to test with a real model"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Natural PDF LLM Extraction Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m benchmark list                              # List available PDF configs
  python -m benchmark prepare atlanta_schools           # Generate ground truth
  python -m benchmark evaluate atlanta_schools --models gpt-4o
  python -m benchmark report atlanta_schools            # Generate HTML report
  python -m benchmark run --models gpt-4o,gemini-1.5-pro --limit-pages 2
  python -m benchmark demo                              # Run demo (no API calls)
        """,
    )

    # Global options
    parser.add_argument(
        "--output",
        "-o",
        default="benchmark_output",
        help="Output directory (default: benchmark_output)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # prepare command
    prepare_parser = subparsers.add_parser("prepare", help="Generate ground truth for PDF")
    prepare_parser.add_argument("pdf_name", help="Name of PDF config (use 'list' to see available)")
    prepare_parser.add_argument("--limit-pages", "-l", type=int, help="Max pages to process")
    prepare_parser.add_argument(
        "--no-traps",
        "-nt",
        action="store_true",
        help="Skip trap PDF (by default both original and trap are prepared)",
    )
    prepare_parser.set_defaults(func=cmd_prepare)

    # evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Run LLM evaluation")
    evaluate_parser.add_argument("pdf_name", help="Name of PDF config")
    evaluate_parser.add_argument("--models", "-m", help="Comma-separated list of models")
    evaluate_parser.add_argument("--limit-pages", "-l", type=int, help="Max pages to evaluate")
    evaluate_parser.add_argument(
        "--no-traps",
        "-nt",
        action="store_true",
        help="Skip trap PDF (by default both original and trap are evaluated)",
    )
    evaluate_parser.set_defaults(func=cmd_evaluate)

    # report command
    report_parser = subparsers.add_parser("report", help="Generate HTML report")
    report_parser.add_argument("pdf_name", help="Name of PDF config")
    report_parser.add_argument("--models", "-m", help="Comma-separated list of models to include")
    report_parser.add_argument("--no-open", action="store_true", help="Don't open in browser")
    report_parser.set_defaults(func=cmd_report)

    # run command (full pipeline)
    run_parser = subparsers.add_parser("run", help="Run complete benchmark pipeline")
    run_parser.add_argument("--pdfs", "-p", help="Comma-separated list of PDF names (default: all)")
    run_parser.add_argument("--models", "-m", help="Comma-separated list of models")
    run_parser.add_argument("--limit-pages", "-l", type=int, help="Max pages per PDF")
    run_parser.add_argument(
        "--no-traps",
        "-nt",
        action="store_true",
        help="Skip trap PDFs (by default both original and trap are run)",
    )
    run_parser.add_argument(
        "--skip-existing",
        "-s",
        action="store_true",
        help="Skip PDF+model combinations that already have results",
    )
    run_parser.add_argument(
        "--cache",
        "-c",
        action="store_true",
        help="Use cache for LLM responses (same prompt+PDF = cached result)",
    )
    run_parser.set_defaults(func=cmd_run)

    # index command
    index_parser = subparsers.add_parser("index", help="Generate index dashboard")
    index_parser.add_argument("--no-open", action="store_true", help="Don't open in browser")
    index_parser.set_defaults(func=cmd_index)

    # list command
    list_parser = subparsers.add_parser("list", help="List available PDF configurations")
    list_parser.set_defaults(func=cmd_list)

    # init command
    init_parser = subparsers.add_parser("init", help="Create sample configuration file")
    init_parser.add_argument("--path", help="Path for config file (default: benchmark.json)")
    init_parser.set_defaults(func=cmd_init)

    # cache command
    cache_parser = subparsers.add_parser("cache", help="Manage response cache")
    cache_parser.add_argument("action", choices=["stats", "clear", "list"], help="Cache action")
    cache_parser.add_argument("--model", help="Filter by model (for list action)")
    cache_parser.set_defaults(func=cmd_cache)

    # demo command
    demo_parser = subparsers.add_parser("demo", help="Run demo without API calls")
    demo_parser.add_argument("--no-open", action="store_true", help="Don't open in browser")
    demo_parser.set_defaults(func=cmd_demo)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
