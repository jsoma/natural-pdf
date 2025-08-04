import argparse
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, distribution
from importlib.metadata import version as get_version
from pathlib import Path
from typing import Dict

from packaging.requirements import Requirement

# ---------------------------------------------------------------------------
# Mapping: sub-command name -> list of pip requirement specifiers to install
# ---------------------------------------------------------------------------
INSTALL_RECIPES: Dict[str, list[str]] = {
    # heavyweight stacks
    "paddle": ["paddlepaddle>=3.0.0", "paddleocr>=3.0.1", "paddlex>=3.0.2", "pandas>=2.2.0"],
    "numpy-high": ["numpy>=2.0"],
    "numpy-low": ["numpy<1.27"],
    "surya": ["surya-ocr<0.15"],
    "yolo": ["doclayout_yolo", "huggingface_hub>=0.29.3"],
    "docling": ["docling"],
    # light helpers
    "deskew": [f"{__package__.split('.')[0]}[deskew]"],
    "search": [f"{__package__.split('.')[0]}[search]"],
    "easyocr": ["easyocr"],
    "ai": [f"{__package__.split('.')[0]}[ai]"],
}


def _build_pip_install_args(requirements: list[str], upgrade: bool = True):
    """Return the pip command list to install/upgrade the given requirement strings."""
    cmd = [sys.executable, "-m", "pip", "--quiet", "install"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.extend(requirements)
    return cmd


def _run(cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def cmd_install(args):
    for extra in args.extras:
        group_key = extra.lower()
        if group_key not in INSTALL_RECIPES:
            print(
                f"❌ Unknown extra '{group_key}'. Known extras: {', '.join(sorted(INSTALL_RECIPES))}",
                file=sys.stderr,
            )
            continue

        requirements = INSTALL_RECIPES[group_key]

        # Special handling for paddle stack: install paddlepaddle & paddleocr first
        # each in its own resolver run, then paddlex.
        base_reqs = [r for r in requirements]
        for req in base_reqs:
            pip_cmd = _build_pip_install_args([req])
            _run(pip_cmd)
        print("✔ Finished installing extra dependencies for", group_key)


def main():
    parser = argparse.ArgumentParser(
        prog="npdf",
        description="Utility CLI for the natural-pdf library",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # install subcommand
    install_p = subparsers.add_parser(
        "install", help="Install optional dependency groups (e.g. paddle, surya)"
    )
    install_p.add_argument(
        "extras", nargs="+", help="One or more extras to install (e.g. paddle surya)"
    )
    install_p.set_defaults(func=cmd_install)

    # list subcommand -------------------------------------------------------
    list_p = subparsers.add_parser("list", help="Show status of optional dependency groups")
    list_p.set_defaults(func=cmd_list)

    args = parser.parse_args()
    args.func(args)


# ---------------------------------------------------------------------------
# List command implementation
# ---------------------------------------------------------------------------


def _pkg_version(pkg_name: str):
    try:
        return get_version(pkg_name)
    except PackageNotFoundError:
        return None


def cmd_list(args):
    print("Optional dependency groups status:\n")
    for extra, reqs in INSTALL_RECIPES.items():
        installed_all = True
        pieces = []
        for req_str in reqs:
            pkg_name = Requirement(req_str).name  # strip version specifiers
            ver = _pkg_version(pkg_name)
            if ver is None:
                installed_all = False
                pieces.append(f"{pkg_name} (missing)")
            else:
                pieces.append(f"{pkg_name} {ver}")
        status = "✓" if installed_all else "✗"
        print(f"{status} {extra:<8} -> " + ", ".join(pieces))
    print("\nLegend: ✓ group fully installed, ✗ some packages missing\n")


if __name__ == "__main__":
    main()
