"""Enhanced LLM enrichment with automatic retry for low-scoring suggestions.

Usage
-----
python -m tools.bad_pdf_eval.llm_enrich_with_retry --submission ja6EqV1 --model gpt-4o

Environment
-----------
OPENAI_API_KEY must be set or passed via --api-key.
"""

import argparse
import concurrent.futures as _futures
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

from openai import OpenAI
from pydantic import BaseModel, Field

# Import quality evaluation
from tools.bad_pdf_eval.evaluate_quality import analyze_code_quality

# Import existing functions and classes
from tools.bad_pdf_eval.llm_enrich import (
    CHEATSHEET_PATH,
    DECISION_TREE_PATH,
    EVAL_DIR,
    WORKFLOWS_PATH,
    DocOutput,
    build_pdf_prompt,
    build_prompt,
    img_to_b64_jpeg,
    read_md,
)

# Global variable
FORCE = False


class RetryOutput(BaseModel):
    """Improved version after feedback."""

    thought_process: str = Field(
        ..., description="Revised reasoning addressing the specific feedback"
    )
    code_suggestion: str = Field(
        ..., description="Improved Python snippet addressing all feedback points"
    )


def build_retry_prompt(
    original_code: str, quality_analysis: Dict[str, Any], context: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Build a retry prompt with specific feedback."""

    feedback_points = []

    # Build specific feedback based on what's missing
    if not quality_analysis["uses_guides"] and "table" in context.get("goal", "").lower():
        feedback_points.append(
            "• Your code doesn't use the Guides API for table extraction. "
            "Use `Guides.from_content()` with actual column headers from the PDF, "
            "then `snap_to_whitespace()` for better results."
        )

    if not quality_analysis["uses_real_text"]:
        feedback_points.append(
            "• Use actual text from the inspect/describe data as anchors. "
            "Look for real headers, labels, or unique text in the evidence."
        )

    if not quality_analysis["uses_until"]:
        feedback_points.append(
            "• Use the `until=` parameter in `.below()` or `.above()` calls "
            "to define region boundaries based on content, not pixels."
        )

    if quality_analysis["uses_tatr"] and quality_analysis["score"] < 6:
        feedback_points.append(
            "• Consider if TATR is really necessary. Can you use Guides or "
            "direct region extraction instead?"
        )

    if not quality_analysis["uses_snap_to_whitespace"] and quality_analysis["uses_guides"]:
        feedback_points.append(
            "• Add `.snap_to_whitespace()` after creating guides to auto-align "
            "to natural gaps in the content."
        )

    retry_prompt = f"""
Your previous code suggestion scored {quality_analysis['score']}/12 in our quality evaluation.
Here's specific feedback to improve it:

{chr(10).join(feedback_points)}

Original code:
```python
{original_code}
```

Please provide an improved version that addresses all the feedback points.
Focus on using modern Natural PDF features and patterns.
"""

    messages = [
        {
            "role": "system",
            "content": "You are a Natural PDF expert. Improve the code based on specific feedback.",
        },
        {"role": "user", "content": retry_prompt},
    ]

    return messages


def enrich_with_retry(
    summary_path: Path,
    api_key: str,
    model: str = "gpt-4o",
    retry_threshold: int = 6,
    max_retries: int = 2,
):
    """Enrich with automatic retry for low-quality suggestions.

    The function will keep *all* attempts (initial + retries) in `attempts` list so we can
    analyse which feedback helped.  The highest-scoring version becomes the primary
    `thought_process` / `code_suggestion` stored at the root level.
    """

    summary = json.loads(summary_path.read_text())

    # Skip if already enriched (unless forced)
    if (
        not FORCE
        and summary.get("thought_process")
        and summary.get("code_suggestion")
        and summary.get("attempts")
    ):
        print(f"[skip] {summary_path.parent.name}: already enriched with attempts")
        return

    print(f"[send] {summary_path.parent.name}: requesting initial enrichment")

    client = OpenAI(api_key=api_key)
    msgs = build_pdf_prompt(summary)

    attempts: List[Dict[str, Any]] = []  # keep all versions

    # Initial attempt
    completion = client.beta.chat.completions.parse(
        model=model, messages=msgs, response_format=DocOutput
    )

    doc_out = completion.choices[0].message.parsed
    quality = analyze_code_quality(doc_out.code_suggestion)
    print(f"Initial quality score: {quality['score']}/12")
    attempts.append(
        {
            "attempt": 0,
            "score": quality["score"],
            "thought_process": doc_out.thought_process,
            "code_suggestion": doc_out.code_suggestion,
        }
    )

    best_doc = doc_out
    best_score = quality["score"]

    # Retry loop
    retry_count = 0
    while quality["score"] < retry_threshold and retry_count < max_retries:
        retry_count += 1
        print(f"[retry {retry_count}] Score below threshold, requesting improvement...")

        retry_msgs = build_retry_prompt(
            doc_out.code_suggestion, quality, {"goal": summary.get("goal", "")}
        )

        retry_completion = client.beta.chat.completions.parse(
            model=model, messages=retry_msgs, response_format=RetryOutput
        )
        retry_out = retry_completion.choices[0].message.parsed

        # Evaluate new version
        new_quality = analyze_code_quality(retry_out.code_suggestion)
        print(f"Retry {retry_count} quality score: {new_quality['score']}/12")

        # Record attempt details
        attempts.append(
            {
                "attempt": retry_count,
                "score": new_quality["score"],
                "thought_process": retry_out.thought_process,
                "code_suggestion": retry_out.code_suggestion,
            }
        )

        # Update best if improved
        if new_quality["score"] > best_score:
            best_score = new_quality["score"]
            best_doc.thought_process = retry_out.thought_process
            best_doc.code_suggestion = retry_out.code_suggestion

        # Prepare for next iteration
        quality = new_quality
        doc_out.code_suggestion = retry_out.code_suggestion
        doc_out.thought_process = retry_out.thought_process

    # Save results – keep best version at root, all attempts nested
    summary["thought_process"] = best_doc.thought_process
    summary["code_suggestion"] = best_doc.code_suggestion
    summary["difficult_elements"] = getattr(
        best_doc, "difficult_elements", summary.get("difficult_elements")
    )
    summary["test_case"] = getattr(best_doc, "test_case", summary.get("test_case"))
    summary["quality_score"] = best_score
    summary["retry_count"] = retry_count
    summary["attempts"] = attempts  # new key for analysis

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(
        f"[update] Best score: {best_score}/12 after {retry_count} retries (kept all {len(attempts)} attempts)"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", help="Submission ID to enrich")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--retry-threshold", type=int, default=6, help="Minimum quality score before retry"
    )
    ap.add_argument("--max-retries", type=int, default=2, help="Maximum number of retry attempts")
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (use 1 to disable parallelism)",
    )
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("OPENAI_API_KEY not provided")

    global FORCE
    FORCE = args.force

    def _get_paths() -> Iterable[Path]:
        if args.submission:
            p = EVAL_DIR / args.submission / "summary.json"
            if not p.exists():
                raise SystemExit("summary.json not found")
            return [p]
        else:
            return list(EVAL_DIR.glob("*/summary.json"))

    paths = _get_paths()

    if args.workers <= 1:
        # Sequential processing
        for p in paths:
            try:
                enrich_with_retry(
                    p, args.api_key, args.model, args.retry_threshold, args.max_retries
                )
            except Exception as e:
                print(f"[error] {p.parent.name}: {e}")
    else:
        # Parallel processing with thread pool (IO-bound)
        print(f"Running with {args.workers} parallel workers…")

        def _safe_process(p: Path):
            try:
                enrich_with_retry(
                    p, args.api_key, args.model, args.retry_threshold, args.max_retries
                )
            except Exception as exc:
                print(f"[error] {p.parent.name}: {exc}")

        with _futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(_safe_process, paths))


if __name__ == "__main__":
    main()
