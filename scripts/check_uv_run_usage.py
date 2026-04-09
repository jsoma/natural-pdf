#!/usr/bin/env python3
"""Enforce uv-based command examples in repo workflow and docs files."""

from __future__ import annotations

import re
import sys
from pathlib import Path

DIRECT_ENV_PATTERN = re.compile(r"\.venv/bin/(?:python|pytest|black|ruff|mypy|pyright|isort|nox)\b")
ACTIVATE_PATTERN = re.compile(r"^\s*source\s+\.venv/bin/activate\b")
PYTHON_MODULE_PATTERN = re.compile(
    r"\bpython(?:3)?\s+-m\s+(pytest|black|ruff|mypy|pyright|isort|nox)\b"
)
RAW_COMMAND_PATTERN = re.compile(
    r"^\s*(?:-\s+)?(?:`)?(?P<cmd>python(?:3)?|pytest|black|ruff|mypy|pyright|isort|nox)\b"
)

COMMAND_RECOMMENDATIONS = {
    "python": "uv run python ...",
    "python3": "uv run python ...",
    "pytest": "uv run pytest ...",
    "black": "uv run black ...",
    "ruff": "uv run ruff ...",
    "mypy": "uv run mypy ...",
    "pyright": "uv run pyright ...",
    "isort": "uv run isort ...",
    "nox": "uv run nox ...",
}


def iter_violations(path: Path):
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return

    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.lstrip()
        command_like = bool(
            re.match(
                r"^(?:-\s+)?(?:`)?(?:source\s+\.venv/bin/activate|\.venv/bin/|python(?:3)?\s+-m\s+|pytest\b|black\b|ruff\b|mypy\b|pyright\b|isort\b|nox\b)",
                stripped,
            )
        )
        if not command_like:
            continue
        if DIRECT_ENV_PATTERN.search(line):
            yield lineno, "Do not use `.venv/bin/...`; use `uv run ...` instead."
            continue
        if ACTIVATE_PATTERN.search(line):
            yield lineno, "Do not `source .venv/bin/activate`; use `uv run ...` or `uv sync`."
            continue
        module_match = PYTHON_MODULE_PATTERN.search(line)
        if module_match:
            cmd = module_match.group(1)
            yield lineno, f"Use `uv run {cmd} ...` instead of `python -m {cmd}`."
            continue
        raw_match = RAW_COMMAND_PATTERN.search(line)
        if not raw_match:
            continue
        cmd = raw_match.group("cmd")
        if line.lstrip().startswith("uv "):
            continue
        suggestion = COMMAND_RECOMMENDATIONS.get(cmd)
        if suggestion:
            yield lineno, f"Use `{suggestion}` instead of raw `{cmd}`."


def main(argv: list[str]) -> int:
    violations = []
    for raw_path in argv[1:]:
        path = Path(raw_path)
        for lineno, message in iter_violations(path):
            violations.append(f"{path}:{lineno}: {message}")

    if not violations:
        return 0

    print("uv command policy violations:\n", file=sys.stderr)
    for violation in violations:
        print(violation, file=sys.stderr)
    print(
        "\nUse `uv run ...` for repo commands, and avoid direct `.venv` activation in docs/workflows.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
