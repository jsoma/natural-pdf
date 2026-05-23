import argparse

from natural_pdf.utils.optional_imports import list_dependency_groups, list_optional_dependencies


def main():
    parser = argparse.ArgumentParser(
        prog="npdf",
        description="Utility CLI for the natural-pdf library",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_p = subparsers.add_parser(
        "doctor",
        help="Show passive diagnostics for optional dependencies and engines",
    )
    doctor_p.set_defaults(func=cmd_doctor)

    list_p = subparsers.add_parser("list", help="Alias for 'doctor'")
    list_p.set_defaults(func=cmd_doctor)

    args = parser.parse_args()
    args.func(args)


def cmd_doctor(args):
    dep_info = list_optional_dependencies()
    groups = list_dependency_groups()

    print("Natural PDF doctor\n")
    print("Optional dependency groups:\n")
    for group, dependency_names in groups.items():
        pieces = []
        installed_all = True
        for dep_name in dependency_names:
            payload = dep_info[dep_name]
            versions = payload["versions"]
            label = ", ".join(f"{pkg} {ver}" for pkg, ver in sorted(versions.items()))
            if not payload["available"]:
                installed_all = False
                label = f"{payload['module_name']} missing"
            elif not label:
                label = payload["module_name"]
            pieces.append(label)
        status = "OK" if installed_all else "MISS"
        print(f"{status:<4} natural-pdf[{group}] -> " + "; ".join(pieces))
        if not installed_all:
            print(f'     install: pip install "natural-pdf[{group}]"')
    print()

    print("Optional dependency modules:\n")
    for name, payload in sorted(dep_info.items()):
        status = "OK" if payload["available"] else "MISS"
        if not payload.get("applicable", True):
            status = "N/A"
        versions = payload["versions"]
        version_text = ", ".join(f"{pkg} {ver}" for pkg, ver in sorted(versions.items()))
        desc = payload.get("description") or ""
        suffix = f" ({version_text})" if version_text else ""
        print(f"{status:<4} {name:<22} -> {desc}{suffix}")
        if not payload["available"] and payload.get("applicable", True):
            hints = " or ".join(payload["install_hints"]) or "pip install"
            print(f"     install: {hints}")
    print()

    print("OCR engines:\n")
    from natural_pdf import options as npdf_options
    from natural_pdf.engine_provider import get_provider
    from natural_pdf.ocr.unified_dispatch import list_engines

    default_engine = getattr(getattr(npdf_options, "ocr", None), "engine", None)
    provider = get_provider()
    provider_engines = set()
    for capability in ("ocr", "ocr.apply", "ocr.extract"):
        provider_engines.update(provider.list(capability).get(capability, ()))

    for name, entry in sorted(list_engines().items()):
        marker = " default" if name == default_engine else ""
        hint = f" install: {entry.install_hint}" if entry.install_hint else ""
        family = f" family={entry.vlm_family}" if entry.vlm_family else ""
        print(f"INFO {name:<16} -> {entry.engine_type}{family}{marker}{hint}")

    for name in sorted(provider_engines - set(list_engines().keys())):
        print(f"INFO {name:<16} -> provider-registered classic")
    print()


if __name__ == "__main__":
    main()
