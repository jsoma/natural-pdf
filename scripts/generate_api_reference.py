#!/usr/bin/env python
"""Generate Markdown API-reference fragments from natural_pdf using Griffe.

Replaces the nine mkdocstrings ``::: natural_pdf...`` directives that used to
live in ``docs/api/index.md``. Each directive maps to one fragment id; the
staging script substitutes ``<!-- npdf-api:include id=<fragment-id> -->``
markers in the authored pages with the fragment file contents.

Design notes (all best-effort per docs/specs/starlight_migration_spec.md §6):

- Static analysis only: ``griffe.load(..., allow_inspection=False)`` with
  Google-style docstring parsing. The package is never imported at runtime.
- Ordering is deterministic: the package-root fragment follows ``__all__``
  order; class members are rendered alphabetically. No timestamps, no
  absolute paths — running twice produces byte-identical output.
- Private members (leading ``_``) are skipped. Unresolvable aliases or
  otherwise broken objects are skipped with a note on stderr, never a crash.
- Headings: fragment-level objects are H3, their members H4 (the authored
  pages place fragments under H2 sections).

Usage:
    uv run python scripts/generate_api_reference.py [--out temp/api-fragments]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import griffe

PACKAGE = "natural_pdf"

# Maximum one-line signature length before switching to one-parameter-per-line.
MAX_SIGNATURE_WIDTH = 88


@dataclass(frozen=True)
class FragmentSpec:
    """One fragment: what mkdocstrings directive it replaces and how to render it."""

    fragment_id: str
    target: str  # dotted griffe path, or the bare package for the root fragment
    include_inherited: bool = False
    module_exports: bool = False  # render the package root's ``__all__`` surface


# One spec per retired ``:::`` directive, in the order they appeared on the page.
FRAGMENTS: tuple[FragmentSpec, ...] = (
    FragmentSpec("natural-pdf", PACKAGE, include_inherited=True, module_exports=True),
    FragmentSpec("text-layout-options", "natural_pdf.text.contracts.TextLayoutOptions"),
    FragmentSpec("extracted-text", "natural_pdf.text.contracts.ExtractedText"),
    FragmentSpec("source-text-segment", "natural_pdf.text.contracts.SourceTextSegment"),
    FragmentSpec(
        "rectangle-element",
        "natural_pdf.elements.rect.RectangleElement",
        include_inherited=True,
    ),
    FragmentSpec(
        "element-collection",
        "natural_pdf.elements.element_collection.ElementCollection",
        include_inherited=True,
    ),
    FragmentSpec(
        "flow-element-collection",
        "natural_pdf.flows.collections.FlowElementCollection",
        include_inherited=True,
    ),
    FragmentSpec(
        "flow-region-collection",
        "natural_pdf.flows.collections.FlowRegionCollection",
        include_inherited=True,
    ),
    FragmentSpec(
        "flow-element",
        "natural_pdf.flows.element.FlowElement",
        include_inherited=True,
    ),
)


def note(message: str) -> None:
    print(f"note: {message}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Signature rendering
# ---------------------------------------------------------------------------


def _format_parameter(param) -> str:
    name = param.name
    kind = param.kind
    variadic = kind in (
        griffe.ParameterKind.var_positional,
        griffe.ParameterKind.var_keyword,
    )
    if kind is griffe.ParameterKind.var_positional:
        name = f"*{name}"
    elif kind is griffe.ParameterKind.var_keyword:
        name = f"**{name}"
    text = name
    if param.annotation is not None:
        text += f": {param.annotation}"
        if param.default is not None and not variadic:
            text += f" = {param.default}"
    elif param.default is not None and not variadic:
        text += f"={param.default}"
    return text


def _parameter_pieces(func, *, skip_self: bool) -> list[str]:
    """Parameter strings including ``/`` and ``*`` markers."""
    params = [
        p
        for p in func.parameters
        if not (skip_self and p.name in ("self", "cls") and not p.annotation)
    ]
    pieces: list[str] = []
    saw_positional_only = False
    needs_star = True
    for param in params:
        if param.kind is griffe.ParameterKind.positional_only:
            saw_positional_only = True
        elif saw_positional_only:
            pieces.append("/")
            saw_positional_only = False
        if param.kind is griffe.ParameterKind.var_positional:
            needs_star = False
        elif param.kind is griffe.ParameterKind.keyword_only and needs_star:
            pieces.append("*")
            needs_star = False
        pieces.append(_format_parameter(param))
    if saw_positional_only:
        pieces.append("/")
    return pieces


def format_function_signature(
    func, display_name: str, *, skip_self: bool = False, include_return: bool = True
) -> str:
    pieces = _parameter_pieces(func, skip_self=skip_self)
    returns = f" -> {func.returns}" if include_return and func.returns is not None else ""
    one_line = f"{display_name}({', '.join(pieces)}){returns}"
    if len(one_line) <= MAX_SIGNATURE_WIDTH:
        return one_line
    body = "".join(f"    {piece},\n" for piece in pieces)
    return f"{display_name}(\n{body}){returns}"


def format_attribute_signature(attr, display_name: str) -> str:
    text = display_name
    if getattr(attr, "annotation", None) is not None:
        text += f": {attr.annotation}"
    value = getattr(attr, "value", None)
    if value is not None:
        rendered = str(value)
        if len(text) + len(rendered) + 3 <= MAX_SIGNATURE_WIDTH:
            text += f" = {rendered}"
        else:
            text += " = ..."
    return text


# ---------------------------------------------------------------------------
# Docstring rendering
# ---------------------------------------------------------------------------


def _render_named_elements(title: str, elements) -> list[str]:
    lines = [f"**{title}:**", ""]
    for element in elements:
        name = getattr(element, "name", "") or ""
        annotation = getattr(element, "annotation", None)
        description = (getattr(element, "description", "") or "").strip()
        description = " ".join(description.split())
        parts = []
        if name:
            # Escape leading * / ** on variadic parameter names so they don't
            # collide with the bold markers.
            escaped_name = name.replace("*", "\\*")
            parts.append(f"**{escaped_name}**")
        if annotation is not None:
            parts.append(f"(`{annotation}`)")
        head = " ".join(parts)
        if head and description:
            lines.append(f"- {head} – {description}")
        elif head:
            lines.append(f"- {head}")
        elif description:
            lines.append(f"- {description}")
    lines.append("")
    return lines


def render_docstring(obj) -> list[str]:
    docstring = obj.docstring
    if docstring is None:
        return []
    try:
        sections = docstring.parsed
    except Exception as exc:  # pragma: no cover - defensive
        note(f"could not parse docstring for {obj.path}: {exc}")
        return [docstring.value.strip(), ""] if docstring.value.strip() else []

    lines: list[str] = []
    kinds = griffe.DocstringSectionKind
    for section in sections:
        kind = section.kind
        if kind is kinds.text:
            text = section.value.strip()
            if text:
                lines.extend([text, ""])
        elif kind is kinds.parameters:
            lines.extend(_render_named_elements("Parameters", section.value))
        elif kind is kinds.other_parameters:
            lines.extend(_render_named_elements("Other parameters", section.value))
        elif kind is kinds.attributes:
            lines.extend(_render_named_elements("Attributes", section.value))
        elif kind is kinds.returns:
            lines.extend(_render_named_elements("Returns", section.value))
        elif kind is kinds.yields:
            lines.extend(_render_named_elements("Yields", section.value))
        elif kind is kinds.raises:
            lines.extend(_render_named_elements("Raises", section.value))
        elif kind is kinds.warns:
            lines.extend(_render_named_elements("Warns", section.value))
        elif kind is kinds.examples:
            # This codebase writes Examples as plain (non-doctest) code
            # blocks, which griffe classifies as "text" items. Emitting them
            # bare lets their '# comment' lines parse as Markdown headings,
            # so fence the whole section as one Python block.
            chunks = [text.strip() for _, text in section.value if text.strip()]
            if chunks:
                lines.extend(["**Examples:**", "", "```python", "\n\n".join(chunks), "```", ""])
        elif kind is kinds.admonition:
            admonition = section.value
            raw_kind = str(admonition.kind or section.title or "note").lower()
            title = (section.title or admonition.kind or "Note").replace("-", " ").title()
            contents = (admonition.contents or "").strip()
            if raw_kind.startswith("example"):
                # Example admonitions hold code; blockquoting it raw lets
                # Python comments parse as Markdown headings (h1s on the
                # built page). Fence the body instead.
                if contents:
                    lines.extend([f"**{title}:**", "", "```python", contents, "```", ""])
                else:
                    lines.extend([f"**{title}**", ""])
            else:
                body = "\n".join(f"> {line}".rstrip() for line in contents.splitlines())
                lines.extend([f"> **{title}:**", body, ""] if body else [f"> **{title}**", ""])
        elif kind is kinds.deprecated:
            element = section.value
            description = (getattr(element, "description", "") or str(element)).strip()
            version = getattr(element, "version", "") or ""
            label = f"Deprecated since {version}" if version else "Deprecated"
            lines.extend([f"> **{label}:** {description}", ""])
        else:
            text = str(getattr(section, "value", "")).strip()
            if text:
                lines.extend([text, ""])
    while lines and lines[-1] == "":
        lines.pop()
    if lines:
        lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Object rendering
# ---------------------------------------------------------------------------


def resolve(member, context: str):
    """Resolve an alias to its final target; return None (with a note) on failure."""
    if not getattr(member, "is_alias", False):
        return member
    try:
        return member.final_target
    except Exception as exc:
        note(f"skipped {context}: unresolvable alias ({exc.__class__.__name__}: {exc})")
        return None


def _is_property(obj) -> bool:
    return "property" in getattr(obj, "labels", set())


def _function_qualifier(obj) -> str:
    labels = getattr(obj, "labels", set())
    for label in ("classmethod", "staticmethod"):
        if label in labels:
            return f" *({label})*"
    return ""


def render_function(func, display_name: str, anchor: str, level: int) -> list[str]:
    heading = "#" * level
    lines = [f'<a id="{anchor}"></a>', ""]
    if _is_property(func):
        lines.append(f"{heading} `{display_name}` *(property)*")
        lines.append("")
        annotation = f": {func.returns}" if func.returns is not None else ""
        lines.extend(["```python", f"{display_name}{annotation}", "```", ""])
    else:
        lines.append(f"{heading} `{display_name}`{_function_qualifier(func)}")
        lines.append("")
        signature = format_function_signature(func, display_name, skip_self=True)
        lines.extend(["```python", signature, "```", ""])
    lines.extend(render_docstring(func))
    return lines


def render_attribute(attr, display_name: str, anchor: str, level: int) -> list[str]:
    heading = "#" * level
    lines = [f'<a id="{anchor}"></a>', ""]
    lines.append(f"{heading} `{display_name}` *(attribute)*")
    lines.append("")
    lines.extend(["```python", format_attribute_signature(attr, display_name), "```", ""])
    lines.extend(render_docstring(attr))
    return lines


def _class_members(cls, include_inherited: bool):
    members = dict(cls.members)
    if include_inherited:
        try:
            for name, member in cls.inherited_members.items():
                members.setdefault(name, member)
        except Exception as exc:
            note(f"could not collect inherited members of {cls.path}: {exc}")
    return members


def render_class(
    cls,
    display_name: str,
    anchor: str,
    level: int,
    *,
    include_inherited: bool,
    show_bases: bool = True,
) -> list[str]:
    heading = "#" * level
    lines = [f'<a id="{anchor}"></a>', "", f"{heading} `{display_name}`", ""]
    bases = [str(base) for base in getattr(cls, "bases", [])]
    if show_bases and bases:
        lines.extend(["Bases: " + ", ".join(f"`{base}`" for base in bases), ""])

    init = cls.members.get("__init__")
    init_target = resolve(init, f"{cls.path}.__init__") if init is not None else None
    if init_target is not None and init_target.kind is griffe.Kind.FUNCTION:
        signature = format_function_signature(
            init_target, display_name, skip_self=True, include_return=False
        )
    else:
        signature = f"class {display_name}"
    lines.extend(["```python", signature, "```", ""])
    lines.extend(render_docstring(cls))
    if init_target is not None and init_target.docstring is not None:
        init_doc = render_docstring(init_target)
        if init_doc:
            lines.extend(init_doc)

    members = _class_members(cls, include_inherited)
    for name in sorted(members):
        if name.startswith("_"):
            continue
        member = resolve(members[name], f"{cls.path}.{name}")
        if member is None:
            continue
        member_anchor = f"{anchor}.{name}"
        if member.kind is griffe.Kind.FUNCTION:
            lines.extend(render_function(member, name, member_anchor, level + 1))
        elif member.kind is griffe.Kind.ATTRIBUTE:
            lines.extend(render_attribute(member, name, member_anchor, level + 1))
        elif member.kind is griffe.Kind.CLASS:
            lines.extend(
                render_class(
                    member,
                    name,
                    member_anchor,
                    level + 1,
                    include_inherited=include_inherited,
                )
            )
        else:
            note(f"skipped {cls.path}.{name}: unhandled kind {member.kind.value}")
    return lines


def render_object(
    obj,
    display_name: str,
    anchor: str,
    level: int,
    *,
    include_inherited: bool,
) -> list[str]:
    if obj.kind is griffe.Kind.CLASS:
        return render_class(obj, display_name, anchor, level, include_inherited=include_inherited)
    if obj.kind is griffe.Kind.FUNCTION:
        return render_function(obj, display_name, anchor, level)
    if obj.kind is griffe.Kind.ATTRIBUTE:
        return render_attribute(obj, display_name, anchor, level)
    note(f"skipped {anchor}: unhandled kind {obj.kind.value}")
    return []


# ---------------------------------------------------------------------------
# Fragment generation
# ---------------------------------------------------------------------------


def generate_module_fragment(module, spec: FragmentSpec) -> str:
    lines: list[str] = []
    exports = list(module.exports or [])
    if not exports:
        exports = sorted(name for name in module.members if not name.startswith("_"))
    for name in exports:
        if name.startswith("_"):
            continue
        member = module.members.get(name)
        if member is None:
            note(f"skipped {PACKAGE}.{name}: listed in __all__ but not found")
            continue
        target = resolve(member, f"{PACKAGE}.{name}")
        if target is None:
            continue
        lines.extend(
            render_object(
                target,
                name,
                f"{PACKAGE}.{name}",
                3,
                include_inherited=spec.include_inherited,
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def generate_object_fragment(module, spec: FragmentSpec) -> str | None:
    try:
        obj = module.modules_collection[spec.target]
    except KeyError:
        note(f"skipped fragment {spec.fragment_id}: {spec.target} not found")
        return None
    obj = resolve(obj, spec.target)
    if obj is None:
        return None
    display_name = spec.target.rsplit(".", 1)[-1]
    lines = render_object(
        obj,
        display_name,
        spec.target,
        3,
        include_inherited=spec.include_inherited,
    )
    if not lines:
        return None
    return "\n".join(lines).rstrip() + "\n"


def generate_fragments(out_dir: Path) -> list[str]:
    module = griffe.load(
        PACKAGE,
        docstring_parser="google",
        allow_inspection=False,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for spec in FRAGMENTS:
        if spec.module_exports:
            content = generate_module_fragment(module, spec)
        else:
            content = generate_object_fragment(module, spec)
        if content is None:
            continue
        path = out_dir / f"{spec.fragment_id}.md"
        path.write_text(content, encoding="utf-8")
        written.append(spec.fragment_id)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        default="temp/api-fragments",
        help="Output directory for fragment files (default: temp/api-fragments)",
    )
    args = parser.parse_args(argv)
    out_dir = Path(args.out)
    written = generate_fragments(out_dir)
    print(f"wrote {len(written)} fragments to {out_dir}")
    for fragment_id in written:
        print(f"  {fragment_id}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
