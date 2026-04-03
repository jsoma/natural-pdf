"""
Individual section renderers for .to_llm() output.

Each function takes a page (or region) and returns a string for one section.
All computations are O(n) over pre-parsed elements. No ML, no image rendering.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Tuple

if TYPE_CHECKING:
    from natural_pdf.core.page import Page
    from natural_pdf.elements.base import Element


def _cluster_values(values: List[float], tolerance: float = 3.0) -> List[Tuple[float, List[int]]]:
    """Cluster sorted numeric values by tolerance gap.

    Returns list of (center, [indices]) tuples sorted by center.
    """
    if not values:
        return []

    indexed = sorted(enumerate(values), key=lambda x: x[1])
    clusters: List[Tuple[List[int], List[float]]] = []
    current_indices = [indexed[0][0]]
    current_values = [indexed[0][1]]

    for i in range(1, len(indexed)):
        idx, val = indexed[i]
        if val - current_values[-1] > tolerance:
            clusters.append((current_indices, current_values))
            current_indices = [idx]
            current_values = [val]
        else:
            current_indices.append(idx)
            current_values.append(val)

    clusters.append((current_indices, current_values))

    result = []
    for indices, vals in clusters:
        center = sum(vals) / len(vals)
        result.append((center, indices))

    return sorted(result, key=lambda x: x[0])


def render_text_layer(page: "Page") -> str:
    """Render the TEXT LAYER section.

    Computes: word count by source, image inventory, page coverage.
    """
    text_elements = page.find_all("text")
    images = page.images

    # Count by source
    source_counts: dict[str, int] = {}
    ocr_confidences: list[float] = []
    ocr_engines: set[str] = set()

    for el in text_elements:
        src = getattr(el, "source", "native")
        source_counts[src] = source_counts.get(src, 0) + 1
        if src == "ocr":
            conf = getattr(el, "confidence", None)
            if conf is not None:
                ocr_confidences.append(conf)
            engine = getattr(el, "ocr_engine", None)
            if engine:
                ocr_engines.add(engine)

    total = len(text_elements)
    native_count = source_counts.get("native", 0) + source_counts.get("pdf", 0)
    ocr_count = source_counts.get("ocr", 0)

    # Build status line
    lines = ["TEXT LAYER"]

    if total == 0:
        lines.append("  0 words")
    elif ocr_count == 0:
        lines.append(f"  {total} words, all native")
    elif native_count == 0:
        engine_str = ", ".join(sorted(ocr_engines)) if ocr_engines else "unknown engine"
        conf_str = ""
        if ocr_confidences:
            median_conf = sorted(ocr_confidences)[len(ocr_confidences) // 2]
            conf_str = f", median confidence {median_conf:.2f}"
        lines.append(f"  {total} words, all OCR ({engine_str}{conf_str})")
    else:
        lines.append(f"  {total} words ({native_count} native + {ocr_count} OCR)")

    # Vector elements summary
    h_lines = sum(1 for l in page.lines if l.is_horizontal)
    v_lines = sum(1 for l in page.lines if l.is_vertical)
    rects = page.find_all("rect")
    if h_lines or v_lines or len(rects):
        vec_parts = []
        if h_lines or v_lines:
            vec_parts.append(f"{h_lines}h + {v_lines}v lines")
        if len(rects):
            vec_parts.append(f"{len(rects)} rects")
        lines.append(f"  {', '.join(vec_parts)}")

    # Image inventory
    if not images:
        lines.append("  No images")
    else:
        page_area = page.width * page.height
        total_img_area = sum(img.width * img.height for img in images)
        coverage = (total_img_area / page_area * 100) if page_area > 0 else 0

        if len(images) == 1:
            img = images[0]
            lines.append(
                f"  1 image ({img.width:.0f}x{img.height:.0f} pts at "
                f"x={img.x0:.0f}, y={img.top:.0f}) — {coverage:.0f}% coverage"
            )
        else:
            lines.append(f"  {len(images)} images — {coverage:.0f}% total coverage")

    return "\n".join(lines)


def render_styles(
    page: "Page",
    include_text: bool = True,
    max_tiers: int = 8,
    max_clusters: int = 3,
    max_samples: int = 5,
) -> str:
    """Render the STYLES & CONTENT section.

    Groups text elements by (size, bold, italic), shows counts, x-clusters,
    text samples, and working selectors.
    """
    text_elements = page.find_all("text")
    if not text_elements:
        return "STYLES & CONTENT\n  (no text elements)"

    # Group by style: round size to nearest 0.5pt for Word-drift tolerance
    tiers: Dict[Tuple, list] = defaultdict(list)
    for el in text_elements:
        size = round(el.size * 2) / 2  # nearest 0.5pt
        bold = getattr(el, "bold", False)
        italic = getattr(el, "italic", False)
        key = (size, bold, italic)
        tiers[key].append(el)

    # Sort: largest size first, then by count descending
    sorted_tiers = sorted(tiers.items(), key=lambda x: (-x[0][0], -len(x[1])))
    sorted_tiers = sorted_tiers[:max_tiers]

    lines = ["STYLES & CONTENT"]

    for (size, bold, italic), elements in sorted_tiers:
        # Style label
        style_parts = [f"{size:g}pt"]
        if bold:
            style_parts.append("bold")
        if italic:
            style_parts.append("italic")
        style_str = " ".join(style_parts)

        count = len(elements)
        lines.append(f"  {style_str} — {count} element{'s' if count != 1 else ''}")

        # Generate selector
        size_min = size - 0.5
        size_max = size + 0.5
        sel_parts = ["text"]
        if bold:
            sel_parts.append(":bold")
        if italic:
            sel_parts.append(":italic")
        # Only add size filter if there are other tiers at nearby sizes
        sel_parts.append(f"[size>={size_min:g}][size<={size_max:g}]")
        if not bold and any(
            k[1] and abs(k[0] - size) < 1.0 for k in tiers if k != (size, bold, italic)
        ):
            sel_parts.append(":not(:bold)")
        selector = "".join(sel_parts)
        lines.append(f"      selector: {selector}")

        if include_text:
            # X-cluster the elements
            x_positions = [el.x0 for el in elements]
            clusters = _cluster_values(x_positions, tolerance=3.0)

            # Cap clusters
            clusters = clusters[:max_clusters]
            remaining = len(elements) - sum(len(c[1]) for c in clusters)

            if len(clusters) == 1:
                # Single cluster — just show samples
                center, indices = clusters[0]
                sorted_els = sorted([elements[i] for i in indices], key=lambda e: (e.top, e.x0))
                samples = sorted_els[:max_samples]
                sample_strs = [f'"{el.text}"' for el in samples]
                extra = len(indices) - len(samples)
                if extra > 0:
                    sample_strs.append(f"+{extra} more")
                lines.append(f"      {', '.join(sample_strs)}")
            else:
                for center, indices in clusters:
                    sorted_els = sorted([elements[i] for i in indices], key=lambda e: (e.top, e.x0))
                    samples = sorted_els[:max_samples]
                    sample_strs = [f'"{el.text}"' for el in samples]
                    extra = len(indices) - len(samples)
                    if extra > 0:
                        sample_strs.append(f"+{extra} more")
                    lines.append(f"      at x≈{center:.0f}: {', '.join(sample_strs)}")

                if remaining > 0:
                    lines.append(f"      (+{remaining} more at other positions)")

    return "\n".join(lines)


def render_lines(
    page: "Page",
    max_h_positions: int = 20,
    max_v_positions: int = 10,
    counts_only: bool = False,
) -> str:
    """Render the LINES section.

    Clusters horizontal lines by y-position, vertical lines by x-position.
    Detects regular spacing. Reports spans.
    """
    all_lines = page.lines
    h_lines = [l for l in all_lines if l.is_horizontal]
    v_lines = [l for l in all_lines if l.is_vertical]

    if not h_lines and not v_lines:
        return "LINES\n  None"

    if counts_only:
        parts = ["LINES"]
        if h_lines:
            parts.append(f"  {len(h_lines)} horizontal")
        if v_lines:
            parts.append(f"  {len(v_lines)} vertical")
        return "\n".join(parts)

    lines = ["LINES"]

    # Horizontal lines — cluster by y-position
    if h_lines:
        y_values = [l.top for l in h_lines]
        h_clusters = _cluster_values(y_values, tolerance=2.0)

        all_x0 = min(l.x0 for l in h_lines)
        all_x1 = max(l.x1 for l in h_lines)

        positions = [c[0] for c in h_clusters]
        positions = positions[:max_h_positions]

        lines.append(f"  Horizontal ({len(h_clusters)}):")

        if len(positions) <= 10:
            pos_str = ", ".join(f"{p:.0f}" for p in positions)
        else:
            first5 = ", ".join(f"{p:.0f}" for p in positions[:5])
            last2 = ", ".join(f"{p:.0f}" for p in positions[-2:])
            pos_str = f"{first5}, ... {last2}"
        lines.append(f"    y = {pos_str}")

        if len(positions) >= 3:
            gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
            avg_gap = sum(gaps) / len(gaps)
            if avg_gap > 0:
                cv = (max(gaps) - min(gaps)) / avg_gap
                if cv < 0.3:
                    lines.append(f"    regular spacing ~{avg_gap:.0f}pt")

        lines.append(f"    spanning x={all_x0:.0f} to x={all_x1:.0f}")

    # Vertical lines — cluster by x-position
    if v_lines:
        x_values = [l.x0 for l in v_lines]
        v_clusters = _cluster_values(x_values, tolerance=2.0)

        all_top = min(l.top for l in v_lines)
        all_bottom = max(l.bottom for l in v_lines)

        positions = [c[0] for c in v_clusters]
        positions = positions[:max_v_positions]

        lines.append(f"  Vertical ({len(v_clusters)}):")
        pos_str = ", ".join(f"{p:.0f}" for p in positions)
        lines.append(f"    x = {pos_str}")
        lines.append(f"    spanning y={all_top:.0f} to y={all_bottom:.0f}")

    return "\n".join(lines)


def render_rectangles(page: "Page", detail: str = "standard") -> str:
    """Render the RECTANGLES section.

    detail: "brief" — count line + small squares only
            "standard" — individual rects with ~100 char text previews
            "full" — individual rects with ~200 char text previews

    Indexes match page.find_all('rect') order so agents can reference them.
    Small rects (max_dim < 20) are summarized as potential checkboxes, not listed individually.
    """
    rects = page.find_all("rect")

    if not rects:
        return "RECTANGLES\n  None"

    total = len(rects)

    # Classify by size, skip degenerate rects (zero width or height — really lines)
    small = []
    small_indices = set()
    degenerate_indices = set()
    for i, r in enumerate(rects):
        if r.width < 1 or r.height < 1:
            degenerate_indices.add(i)
        elif max(r.width, r.height) < 20:
            small.append(r)
            small_indices.add(i)

    skip = small_indices | degenerate_indices
    # Non-small, non-degenerate rects for individual listing
    display_rects = [(i, r) for i, r in enumerate(rects) if i not in skip]

    lines = ["RECTANGLES"]
    lines.append(f"  {total} total (find_all('rect') order)")
    omitted = 0

    if detail != "brief" and display_rects:
        preview_len = 200 if detail == "full" else 100
        max_rects = 30 if detail == "full" else 15
        tolerance = 2.0  # bbox containment tolerance in pts

        # Detect nesting: for each display rect, find if it's inside another
        parent_map: Dict[int, int] = {}  # child_index -> parent_index
        for i, child in display_rects:
            for j, parent in display_rects:
                if i == j:
                    continue
                # Check if child is fully contained within parent (with tolerance)
                if (
                    child.x0 >= parent.x0 - tolerance
                    and child.top >= parent.top - tolerance
                    and child.x1 <= parent.x1 + tolerance
                    and child.bottom <= parent.bottom + tolerance
                ):
                    # Pick the smallest containing parent (tightest fit)
                    if i not in parent_map:
                        parent_map[i] = j
                    else:
                        # Compare areas — prefer smaller parent
                        prev_parent = next(r for idx, r in display_rects if idx == parent_map[i])
                        prev_area = prev_parent.width * prev_parent.height
                        this_area = parent.width * parent.height
                        if this_area < prev_area:
                            parent_map[i] = j

        # Render each non-small rect (capped)
        shown = display_rects[:max_rects]
        omitted = len(display_rects) - len(shown)
        for i, r in shown:
            # Extract text content
            try:
                text = r.extract_text()
            except Exception:
                text = ""
            text = (text or "").strip()

            # Format position and size
            pos = f"({r.x0:.0f},{r.top:.0f})-({r.x1:.0f},{r.bottom:.0f})"
            dims = f"{r.width:.0f}x{r.height:.0f}pt"

            # Format text preview
            if text:
                char_count = len(text)
                # Escape newlines for single-line display
                preview = text.replace("\n", "\\n")
                if len(preview) > preview_len:
                    preview = preview[:preview_len] + "..."
                text_part = f'{char_count} chars: "{preview}"'
            else:
                text_part = "empty"

            # Indentation and nesting suffix
            indent = "    " if i in parent_map else "  "
            suffix = f"  [inside #{parent_map[i]}]" if i in parent_map else ""

            lines.append(f"{indent}#{i} {pos} {dims} — {text_part}{suffix}")

    # Omitted/degenerate summaries
    if detail != "brief":
        if display_rects and omitted > 0:
            lines.append(f"  ... +{omitted} more (use detail='full' to see all)")
        if degenerate_indices:
            lines.append(
                f"  ({len(degenerate_indices)} degenerate rects with zero width/height omitted)"
            )

    # Small squares summary (all detail levels)
    squares = [r for r in small if abs(r.width - r.height) < 3.0]
    if squares:
        avg_size = sum(max(r.width, r.height) for r in squares) / len(squares)
        y_range = f"y={min(r.top for r in squares):.0f}-{max(r.bottom for r in squares):.0f}"
        lines.append(
            f"  {len(squares)} small squares ~{avg_size:.0f}x{avg_size:.0f}pt at {y_range}"
        )

    return "\n".join(lines)


def render_alignment(page: "Page", min_pct: float = 3.0) -> str:
    """Render the ALIGNMENT section.

    Shows x-position clusters (columns) and y-position clusters (rows).
    """
    text_elements = page.find_all("text")
    if not text_elements:
        return "ALIGNMENT\n  (no text elements)"

    total = len(text_elements)
    min_count = max(2, int(total * min_pct / 100))

    out = ["ALIGNMENT"]

    # X-clusters (columns)
    x_positions = [el.x0 for el in text_elements]
    x_clusters = _cluster_values(x_positions, tolerance=3.0)
    x_significant = [(c, idx) for c, idx in x_clusters if len(idx) >= min_count]

    if x_significant:
        out.append("  Columns (x-clusters):")
        for center, indices in x_significant:
            elements = [text_elements[i] for i in indices]
            y_min = min(el.top for el in elements)
            y_max = max(el.bottom for el in elements)
            out.append(f"    x≈{center:.0f} ({len(indices)} elements, y={y_min:.0f}-{y_max:.0f})")

    # Y-clusters (rows) — only useful if there are enough distinct rows
    y_positions = [el.top for el in text_elements]
    y_clusters = _cluster_values(y_positions, tolerance=3.0)
    y_significant = [(c, idx) for c, idx in y_clusters if len(idx) >= 2]

    if len(y_significant) >= 3:
        # Detect regular row spacing
        centers = [c for c, _ in y_significant]
        if len(centers) >= 3:
            gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
            avg_gap = sum(gaps) / len(gaps)
            if avg_gap > 0:
                cv = (max(gaps) - min(gaps)) / avg_gap
                if cv < 0.3:
                    out.append(
                        f"  Rows: {len(y_significant)} rows, regular spacing ~{avg_gap:.0f}pt"
                    )
                else:
                    out.append(
                        f"  Rows: {len(y_significant)} rows, irregular spacing ({min(gaps):.0f}-{max(gaps):.0f}pt)"
                    )

    if len(out) == 1:
        out.append("  No dominant alignment patterns")

    return "\n".join(out)


def render_layout_preview(page: "Page", max_lines: int = 15) -> str:
    """Render the LAYOUT PREVIEW section.

    Bounded extract_text(layout=True) output.
    """
    text_elements = page.find_all("text")
    if not text_elements:
        return "LAYOUT PREVIEW\n  (no text on this page)"

    text = page.extract_text(layout=True)
    if not text or not text.strip():
        return "LAYOUT PREVIEW\n  (no text on this page)"

    all_text_lines = text.split("\n")
    while all_text_lines and not all_text_lines[-1].strip():
        all_text_lines.pop()
    # Also strip leading empty lines
    while all_text_lines and not all_text_lines[0].strip():
        all_text_lines.pop(0)

    total_count = len(all_text_lines)
    shown = all_text_lines[:max_lines]

    out = []
    if total_count > max_lines:
        out.append(f"LAYOUT PREVIEW (first {max_lines} of {total_count} lines)")
    else:
        out.append("LAYOUT PREVIEW")

    for line in shown:
        out.append(f"  {line}")

    if total_count > max_lines:
        out.append("  ...")

    return "\n".join(out)


def render_hints(page: "Page", style: str = "api") -> str:
    """Render the SUGGESTED METHODS section (feature-flagged).

    style: "descriptive" (natural language) or "api" (method names)
    """
    text_elements = page.find_all("text")
    h_lines = sum(1 for l in page.lines if l.is_horizontal)
    v_lines = sum(1 for l in page.lines if l.is_vertical)
    rects = page.find_all("rect")
    images = page.images
    word_count = len(text_elements)
    img_coverage = 0
    if images:
        page_area = page.width * page.height
        img_coverage = sum(i.width * i.height for i in images) / page_area * 100 if page_area else 0

    hints = []

    if h_lines >= 4 and v_lines >= 2:
        if style == "api":
            hints.append(
                f"{h_lines}h+{v_lines}v lines form grid -> .detect_form_cells() or .extract_table()"
            )
        else:
            hints.append(
                "Grid of ruling lines detected — cell-based extraction likely more reliable than spatial navigation"
            )

    if word_count == 0 and img_coverage > 50:
        if style == "api":
            hints.append("No text, high image coverage -> .apply_ocr()")
        else:
            hints.append("Page appears to be scanned — OCR required before text extraction")

    small_squares = [
        r for r in rects if max(r.width, r.height) < 20 and abs(r.width - r.height) < 3
    ]
    if len(small_squares) >= 3:
        if style == "api":
            hints.append(f"{len(small_squares)} small squares -> .detect_checkboxes()")
        else:
            hints.append(f"{len(small_squares)} small square shapes may be checkboxes")

    if h_lines == 0 and v_lines == 0 and word_count > 20:
        x_clusters = _cluster_values([el.x0 for el in text_elements], tolerance=3.0)
        significant = [c for c in x_clusters if len(c[1]) >= max(2, int(word_count * 0.03))]
        if len(significant) >= 3:
            if style == "api":
                hints.append(
                    "Multiple aligned columns without lines -> .extract_table(method='text') or .guides().from_whitespace()"
                )
            else:
                hints.append(
                    "Text appears column-aligned without ruling lines — may be a borderless table"
                )

    if not hints:
        return ""

    header = "SUGGESTED METHODS" if style == "api" else "OBSERVATIONS"
    result = [header]
    for hint in hints[:5]:
        result.append(f"  {hint}")
    return "\n".join(result)
