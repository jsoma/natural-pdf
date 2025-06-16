import argparse
import subprocess
import sys
from importlib.metadata import distribution, PackageNotFoundError
from pathlib import Path
from typing import Dict

# ---------------------------------------------------------------------------
# Mapping: sub-command name -> list of pip requirement specifiers to install
# ---------------------------------------------------------------------------
INSTALL_RECIPES: Dict[str, list[str]] = {
    # heavyweight stacks
    "paddle": ["paddlepaddle>=3.0.0", "paddleocr>=3.0.1", "paddlex>=3.0.2"],
    "surya": ["surya-ocr>=0.13.0"],
    "yolo": ["doclayout_yolo", "huggingface_hub>=0.29.3"],
    "docling": ["docling"],
    # light helpers
    "deskew": [f"{__package__.split('.')[0]}[deskew]"],
    "search": [f"{__package__.split('.')[0]}[search]"],
    "easyocr": ["easyocr"],
}


def _build_pip_install_args(requirements: list[str], upgrade: bool = True):
    """Return the pip command list to install/upgrade the given requirement strings."""
    cmd = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.extend(requirements)
    return cmd


def _run(cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def cmd_install(args):
    group_key = args.extra.lower()
    if group_key not in INSTALL_RECIPES:
        print(
            f"❌ Unknown extra '{group_key}'. Known extras: {', '.join(sorted(INSTALL_RECIPES))}",
            file=sys.stderr,
        )
        sys.exit(1)

    requirements = INSTALL_RECIPES[group_key]

    # Try trivial skip for paddlex specific check
    try:
        dist = distribution("paddlex") if group_key == "paddle" else None
        if dist and group_key == "paddle":
            from packaging.version import parse as V

            if V(dist.version) >= V("3.0.2"):
                print("✓ paddlex already ≥ 3.0.2 – nothing to do.")
                return
    except PackageNotFoundError:
        pass

    pip_cmd = _build_pip_install_args(requirements)
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
    install_p.add_argument("extra", help="Name of the extra to install")
    install_p.set_defaults(func=cmd_install)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main() 