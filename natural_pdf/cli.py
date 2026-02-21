import argparse
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version
from typing import Dict

from natural_pdf.utils.optional_imports import list_optional_dependencies


def main():
    parser = argparse.ArgumentParser(
        prog="npdf",
        description="Utility CLI for the natural-pdf library",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list subcommand
    list_p = subparsers.add_parser("list", help="Show status of optional dependencies")
    list_p.set_defaults(func=cmd_list)

    args = parser.parse_args()
    args.func(args)


# ---------------------------------------------------------------------------
# List command implementation
# ---------------------------------------------------------------------------

EXTRA_GROUPS: Dict[str, list[str]] = {
    "export": ["pikepdf", "img2pdf", "jupytext", "nbformat"],
    "paddle": ["paddlepaddle", "paddleocr", "paddlex"],
}


def _pkg_version(pkg_name: str):
    try:
        return get_version(pkg_name)
    except PackageNotFoundError:
        return None


def cmd_list(args):
    print("Optional dependency groups:\n")
    for group, pkgs in EXTRA_GROUPS.items():
        installed_all = True
        pieces = []
        for pkg in pkgs:
            ver = _pkg_version(pkg)
            if ver is None:
                installed_all = False
                pieces.append(f"{pkg} (missing)")
            else:
                pieces.append(f"{pkg} {ver}")
        status = "\u2713" if installed_all else "\u2717"
        install_cmd = f'pip install "natural-pdf[{group}]"'
        print(f"{status} {group:<8} -> " + ", ".join(pieces))
        if not installed_all:
            print(f"   install: {install_cmd}")
    print()

    print("Optional dependency modules:\n")
    dep_info = list_optional_dependencies()
    for name, payload in sorted(dep_info.items()):
        status = "\u2713" if payload["available"] else "\u2717"
        hints = " or ".join(payload["install_hints"]) or "pip install"
        desc = payload.get("description") or ""
        print(f"{status} {name:<20} -> {desc}")
        if not payload["available"]:
            print(f"   install: {hints}")
    print()


if __name__ == "__main__":
    main()
