"""Form cell detection via morphological line isolation + contour analysis.

Detects rectangular cells in government forms, arrest reports, tax forms, etc.
from rendered page images. Optionally merges cells where OCR text spans
multiple adjacent cells on the same row.

Pipeline:
1. Render page to image at target resolution
2. Adaptive threshold -> binary
3. Multi-pass morphological opening -> isolate H and V lines
4. Extract line contours, extend bounding boxes along primary axis
5. Hybrid V-line context filter (short lines need both H-line endpoints)
6. Redraw extended lines -> watertight grid -> invert -> contour detection
7. Filter contours by area/dimensions/aspect/rectangularity
8. Optional: OCR-based merge (cells with same y0/y1 overlapping same text)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class FormCellAnalyzer:
    """Detects form cells from a rendered page image."""

    # Default parameters (calibrated for 2200px target width)
    DEFAULTS = {
        "resolution": 2200,
        "merge": True,
        "adaptive_block": 15,
        "adaptive_c": 5,
        "min_line_area_h": 100,
        "min_line_area_v": 100,
        "line_extend_px": 15,
        "v_context_tolerance": 12,
        "v_height_both_threshold": 40,
        "min_cell_area": 350,
        "max_cell_area_ratio": 0.40,
        "min_cell_w": 45,
        "min_cell_h": 18,
        "max_aspect": 50.0,
        "min_rectangularity": 0.35,
        "border_margin": 6,
        "dedup_iou": 0.85,
    }

    def __init__(self, page):
        self._page = page

    def detect_form_cells(
        self,
        merge: bool = True,
        resolution: int = 2200,
        existing: str = "replace",
        debug_dir: str | None = None,
        cell_labels: list | None = None,
        **kwargs,
    ) -> list:
        """Detect form cells and create Region elements.

        Args:
            merge: If True, merge cells where OCR text spans multiple cells
                on the same row.
            resolution: Target image width in pixels for detection.
            existing: How to handle existing form_cell regions.
                "replace" removes old ones first, "keep" adds alongside.
            debug_dir: If set, save step-by-step debug images to this directory.
            cell_labels: List of selectors or elements that identify real cell
                labels. When two adjacent cells both contain a label element
                (by center point), they won't be merged. Strings are resolved
                via page.find(). Elements are used directly.
            **kwargs: Override any default parameter (e.g. min_cell_w=30).

        Returns:
            List of Region objects for detected form cells.
        """
        try:
            import cv2
            import numpy as np
        except ImportError:
            raise ImportError(
                "Form cell detection requires opencv-python and numpy. "
                "Install with: pip install opencv-python-headless numpy"
            )

        if debug_dir:
            import os

            os.makedirs(debug_dir, exist_ok=True)

        # Build params from defaults + overrides
        params = dict(self.DEFAULTS)
        params["resolution"] = resolution
        params["merge"] = merge
        params.update(kwargs)

        # Render page to image
        img, scale_x, scale_y = self._render_page(params["resolution"])
        if img is None:
            logger.error("Failed to render page for form cell detection")
            return []

        img_h, img_w = img.shape[:2]
        k = img_w / 2200.0

        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, "00_rendered.png"), img)

        # Run detection pipeline
        cells = self._detect_cells(img, params, k, debug_dir=debug_dir)
        logger.info("Detected %d raw cells", len(cells))

        # Debug: pre-merge cells with text bboxes
        if debug_dir:
            inv_sx = 1.0 / scale_x
            inv_sy = 1.0 / scale_y
            txt_img = img.copy()
            # Draw cells
            for i, (x1, y1, x2, y2) in enumerate(cells):
                cv2.rectangle(txt_img, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(
                    txt_img, str(i), (x1 + 4, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
                )
                cv2.putText(
                    txt_img,
                    str(i),
                    (x1 + 4, y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
            # Draw text element bboxes
            text_elements = self._page.find_all("text")
            if text_elements:
                for el in text_elements:
                    tx1 = int(el.x0 * inv_sx)
                    ty1 = int(el.top * inv_sy)
                    tx2 = int(el.x1 * inv_sx)
                    ty2 = int(el.bottom * inv_sy)
                    cv2.rectangle(txt_img, (tx1, ty1), (tx2, ty2), (0, 0, 255), 1)
            cv2.imwrite(os.path.join(debug_dir, "08b_cells_with_text.png"), txt_img)

        # Resolve cell_labels into elements
        label_elements = []
        if cell_labels:
            for label in cell_labels:
                if isinstance(label, str):
                    el = self._page.find(label)
                    if el is not None:
                        label_elements.append(el)
                    else:
                        logger.debug("cell_labels selector matched nothing: %s", label)
                else:
                    label_elements.append(label)

        # OCR-based merge
        if params["merge"]:
            cells = self._merge_with_ocr(
                cells,
                img_w,
                img_h,
                scale_x,
                scale_y,
                label_elements=label_elements,
            )
            logger.info("After OCR merge: %d cells", len(cells))

        if debug_dir:
            cell_img = img.copy()
            for i, (x1, y1, x2, y2) in enumerate(cells):
                h_val = int(i * 180 / max(len(cells), 1)) % 180
                hsv = np.uint8([[[h_val, 200, 230]]])
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
                color = tuple(int(c) for c in bgr)
                overlay = cell_img.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.addWeighted(overlay, 0.3, cell_img, 0.7, 0, cell_img)
                cv2.rectangle(cell_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    cell_img, str(i), (x1 + 4, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
                )
                cv2.putText(
                    cell_img,
                    str(i),
                    (x1 + 4, y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
            cv2.imwrite(os.path.join(debug_dir, "09_cells_final.png"), cell_img)

        # Remove existing regions AFTER successful detection
        if existing == "replace":
            self._remove_existing_regions()

        # Convert image-pixel cells to PDF-coordinate regions
        regions = self._create_regions(cells, scale_x, scale_y)
        logger.info("Created %d form_cell regions", len(regions))

        return regions

    def _render_page(self, target_width: int):
        """Render page to an OpenCV image at the target pixel width.

        Renders at a high DPI then resizes to the target width using
        cv2.INTER_CUBIC, which preserves thin lines better than PDF
        renderer upscaling.
        """
        import cv2
        import numpy as np

        page_width_pts = float(getattr(self._page, "width", 0) or 0)
        if page_width_pts <= 1:
            logger.error("Invalid page width (%s); cannot render", page_width_pts)
            return None, 1.0, 1.0

        # Render at native/moderate DPI then let OpenCV resize to target
        # with INTER_CUBIC. This avoids soft upscaling by the PDF renderer
        # when the embedded image is lower-res than the target width.
        target_dpi = int(target_width / page_width_pts * 72)
        # Cap at 300 DPI to get native-ish resolution; OpenCV upscales the rest
        render_dpi = max(150, min(300, target_dpi))
        image = self._page.render(resolution=render_dpi)
        if image is None:
            return None, 1.0, 1.0

        img = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

        # Resize to target width using INTER_CUBIC (matches standalone behavior)
        h_img, w_img = img.shape[:2]
        if abs(w_img - target_width) > 10:
            scale = target_width / w_img
            new_h = int(round(h_img * scale))
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
            img = cv2.resize(img, (target_width, new_h), interpolation=interp)

        scale_x = self._page.width / img.shape[1]
        scale_y = self._page.height / img.shape[0]
        return img, scale_x, scale_y

    def _detect_cells(
        self, img, params: dict, k: float, debug_dir: str | None = None
    ) -> List[Tuple[int, int, int, int]]:
        """Core detection pipeline: lines -> grid -> contours -> cells."""
        import cv2
        import numpy as np

        def clamp_int(v, lo, hi):
            return int(max(lo, min(hi, round(v))))

        def dbg(name, image):
            if debug_dir:
                import os

                cv2.imwrite(os.path.join(debug_dir, f"{name}.png"), image)

        # Step 1: Binary
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        block = params["adaptive_block"]
        if block < 3:
            block = 3
        if block % 2 == 0:
            block += 1
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block,
            params["adaptive_c"],
        )
        dbg("01_gray", gray)
        dbg("02_binary", binary)

        # Step 2: Horizontal lines (two-pass + gap close)
        h_long = clamp_int(40 * k, 35, 80)
        h_short = clamp_int(25 * k, 18, 45)
        hk_long = cv2.getStructuringElement(cv2.MORPH_RECT, (h_long, 1))
        hk_short = cv2.getStructuringElement(cv2.MORPH_RECT, (h_short, 1))
        h_open_long = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk_long)
        h_open_short = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk_short)
        h_lines = cv2.bitwise_or(h_open_long, h_open_short)
        dbg("03a_h_open_long", h_open_long)
        dbg("03b_h_open_short", h_open_short)
        dbg("03c_h_combined", h_lines)
        h_close_w = clamp_int(15 * k, 10, 30)
        h_close = cv2.getStructuringElement(cv2.MORPH_RECT, (h_close_w, 1))
        h_lines = cv2.morphologyEx(h_lines, cv2.MORPH_CLOSE, h_close)
        dbg("03d_h_gap_closed", h_lines)
        h_lines = self._remove_small_components(h_lines, params["min_line_area_h"])
        dbg("03e_h_cleaned", h_lines)

        # Step 3: Vertical lines (two-pass)
        v_long = clamp_int(30 * k, 18, 55)
        v_short = clamp_int(18 * k, 12, 35)
        vk_long = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_long))
        vk_short = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_short))
        v_open_long = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vk_long)
        v_open_short = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vk_short)
        v_lines = cv2.bitwise_or(v_open_long, v_open_short)
        dbg("04a_v_open_long", v_open_long)
        dbg("04b_v_open_short", v_open_short)
        dbg("04c_v_combined", v_lines)
        v_close_h = clamp_int(10 * k, 6, 18)
        v_close = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_close_h))
        v_lines = cv2.morphologyEx(v_lines, cv2.MORPH_CLOSE, v_close)
        dbg("04d_v_gap_closed", v_lines)
        v_lines = self._remove_small_components(v_lines, params["min_line_area_v"])
        dbg("04e_v_cleaned", v_lines)

        # Step 4: Extract and extend line bounding boxes
        extend = clamp_int(params["line_extend_px"] * k, 6, 20)
        min_h_len = clamp_int(20 * k, 15, 60)
        min_v_len = clamp_int(20 * k, 12, 40)
        context_tol = clamp_int(params["v_context_tolerance"] * k, 6, 16)
        both_thresh = clamp_int(params["v_height_both_threshold"] * k, 25, 60)

        h_boxes = self._extract_and_extend_lines(h_lines, "h", extend, min_h_len)
        v_boxes = self._extract_and_extend_lines(
            v_lines,
            "v",
            extend,
            min_v_len,
            h_line_mask=h_lines,
            context_tol=context_tol,
            both_threshold=both_thresh,
        )
        logger.debug("Extended lines: %d H, %d V", len(h_boxes), len(v_boxes))

        if debug_dir:
            bbox_img = img.copy()
            for x, y, w, h in h_boxes:
                cv2.rectangle(bbox_img, (x, y), (x + w, y + h), (0, 0, 255), 1)
            for x, y, w, h in v_boxes:
                cv2.rectangle(bbox_img, (x, y), (x + w, y + h), (255, 0, 0), 1)
            dbg("05_line_bboxes", bbox_img)

        # Step 5: Redraw -> watertight grid
        img_h, img_w = img.shape[:2]
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        for x, y, w, h in h_boxes:
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        for x, y, w, h in v_boxes:
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        dbg("06_grid_mask", mask)

        # Step 6: Invert -> cell regions
        inverse = cv2.bitwise_not(mask)
        m = clamp_int(params["border_margin"] * k, 2, 12)
        inverse[:m, :] = 0
        inverse[-m:, :] = 0
        inverse[:, :m] = 0
        inverse[:, -m:] = 0
        dbg("07_inverse", inverse)

        # Step 7: Find cells from contours
        img_area = img_h * img_w
        contours, _ = cv2.findContours(inverse, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cells = []
        max_area = img_area * params["max_cell_area_ratio"]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < params["min_cell_area"] or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w < params["min_cell_w"] or h < params["min_cell_h"]:
                continue
            aspect = max(w, h) / max(min(w, h), 1)
            if aspect > params["max_aspect"]:
                continue
            rectangularity = float(area) / float(max(w * h, 1))
            if rectangularity < params["min_rectangularity"]:
                continue
            cells.append((x, y, x + w, y + h))

        # Dedup
        cells = self._dedup_boxes(cells, params["dedup_iou"])
        cells.sort(key=lambda c: (c[1], c[0]))

        if debug_dir:
            cell_img = img.copy()
            for i, (x1, y1, x2, y2) in enumerate(cells):
                h_val = int(i * 180 / max(len(cells), 1)) % 180
                hsv = np.uint8([[[h_val, 200, 230]]])
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
                color = tuple(int(c) for c in bgr)
                overlay = cell_img.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.addWeighted(overlay, 0.3, cell_img, 0.7, 0, cell_img)
                cv2.rectangle(cell_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    cell_img, str(i), (x1 + 4, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
                )
                cv2.putText(
                    cell_img,
                    str(i),
                    (x1 + 4, y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
            dbg("08_cells_pre_merge", cell_img)

        return cells

    def _merge_with_ocr(
        self,
        cells: List[Tuple[int, int, int, int]],
        img_w: int,
        img_h: int,
        scale_x: float,
        scale_y: float,
        label_elements: list | None = None,
    ) -> List[Tuple[int, int, int, int]]:
        """Merge cells where OCR text spans multiple cells on the same row."""
        # Get text elements from the page
        text_elements = self._page.find_all("text")
        if not text_elements or len(text_elements) == 0:
            logger.debug("No text elements found for OCR merge")
            return cells

        # Scale text element PDF coords to image pixel coords
        inv_sx = 1.0 / scale_x  # PDF -> image
        inv_sy = 1.0 / scale_y
        ocr_boxes = []
        for el in text_elements:
            text = el.extract_text().strip() if hasattr(el, "extract_text") else ""
            if not text:
                continue
            ocr_boxes.append(
                (
                    el.x0 * inv_sx,
                    el.top * inv_sy,
                    el.x1 * inv_sx,
                    el.bottom * inv_sy,
                )
            )

        if not ocr_boxes:
            return cells

        # Precompute which cells are protected by label elements.
        # A cell is protected if any label element's center falls inside it.
        # Assign to the smallest containing cell (avoids matching a giant parent).
        n = len(cells)
        protected = set()
        if label_elements:
            for el in label_elements:
                cx = (el.x0 + el.x1) / 2 * inv_sx
                cy = (el.top + el.bottom) / 2 * inv_sy
                best_i = -1
                best_area = float("inf")
                for i in range(n):
                    x1, y1, x2, y2 = cells[i]
                    if x1 <= cx <= x2 and y1 <= cy <= y2:
                        area = (x2 - x1) * (y2 - y1)
                        if area < best_area:
                            best_area = area
                            best_i = i
                if best_i >= 0:
                    protected.add(best_i)
            if protected:
                logger.debug("Protected cells (label elements): %s", protected)

        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        row_tol = max(3, round(5 * img_w / 2200))
        merge_count = 0
        for tx1, ty1, tx2, ty2 in ocr_boxes:
            # Find all cells this text overlaps
            overlapping = []
            for i in range(n):
                cx1, cy1, cx2, cy2 = cells[i]
                ox = min(tx2, cx2) - max(tx1, cx1)
                oy = min(ty2, cy2) - max(ty1, cy1)
                if ox >= 2 and oy >= 2:
                    overlapping.append(i)

            if len(overlapping) < 2:
                continue

            # Merge pairs with same y0/y1 (same row = same horizontal lines)
            for a_pos in range(len(overlapping)):
                for b_pos in range(a_pos + 1, len(overlapping)):
                    ai = overlapping[a_pos]
                    bi = overlapping[b_pos]
                    # Skip if both cells are protected by labels
                    if ai in protected and bi in protected:
                        continue
                    # Same row: matching top and bottom within tolerance
                    if (
                        abs(cells[ai][1] - cells[bi][1]) <= row_tol
                        and abs(cells[ai][3] - cells[bi][3]) <= row_tol
                    ):
                        if find(ai) != find(bi):
                            union(ai, bi)
                            merge_count += 1

        if merge_count == 0:
            return cells

        # Build merged cells from union-find groups
        from collections import defaultdict

        groups = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(i)

        merged = []
        for group_indices in groups.values():
            xs1 = [cells[i][0] for i in group_indices]
            ys1 = [cells[i][1] for i in group_indices]
            xs2 = [cells[i][2] for i in group_indices]
            ys2 = [cells[i][3] for i in group_indices]
            merged.append((min(xs1), min(ys1), max(xs2), max(ys2)))

        merged.sort(key=lambda c: (c[1], c[0]))
        logger.debug("OCR merge: %d pair merges, %d -> %d cells", merge_count, n, len(merged))
        return merged

    def _create_regions(self, cells, scale_x, scale_y) -> list:
        """Convert image-pixel cells to PDF Region objects."""
        regions = []
        for x1, y1, x2, y2 in cells:
            # Convert to PDF coordinates
            pdf_x0 = x1 * scale_x
            pdf_y0 = y1 * scale_y
            pdf_x1 = x2 * scale_x
            pdf_y1 = y2 * scale_y

            # Clamp to page bounds
            pdf_x0 = max(0, pdf_x0)
            pdf_y0 = max(0, pdf_y0)
            pdf_x1 = min(self._page.width, pdf_x1)
            pdf_y1 = min(self._page.height, pdf_y1)

            region = self._page.create_region(pdf_x0, pdf_y0, pdf_x1, pdf_y1)
            region.region_type = "form_cell"
            region.normalized_type = "form_cell"
            region.source = "form_cell"
            self._page.add_region(region, source="form_cell")
            regions.append(region)

        return regions

    def _remove_existing_regions(self):
        """Remove previously detected form_cell regions."""
        existing = self._page.find_all("region[type=form_cell]")
        if existing:
            for r in existing:
                try:
                    self._page._regions.remove(r)
                except (ValueError, AttributeError):
                    pass

    # --- CV helper methods ---

    @staticmethod
    def _remove_small_components(mask, min_area):
        import cv2
        import numpy as np

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                out[labels == i] = 255
        return out

    @staticmethod
    def _extract_and_extend_lines(
        line_mask,
        orientation,
        extend_px,
        min_length,
        max_thickness=20,
        h_line_mask=None,
        context_tol=8,
        both_threshold=40,
    ):
        """Extract line contours, extend along primary axis, with hybrid context filter."""
        import cv2
        import numpy as np

        contours, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        extended = []
        img_h, img_w = line_mask.shape[:2]
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if orientation == "h":
                if w < min_length or h > max_thickness:
                    continue
                x_new = max(0, x - extend_px)
                x_end = min(img_w, x + w + extend_px)
                extended.append((x_new, y, x_end - x_new, h))
            else:
                if h < min_length or w > max_thickness:
                    continue
                if h_line_mask is not None:
                    cx = x + w // 2
                    top_y = y
                    bot_y = min(y + h, img_h - 1)
                    x_lo = max(0, cx - context_tol)
                    x_hi = min(img_w, cx + context_tol + 1)
                    top_window = h_line_mask[
                        max(0, top_y - context_tol) : top_y + context_tol + 1, x_lo:x_hi
                    ]
                    bot_window = h_line_mask[
                        max(0, bot_y - context_tol) : min(img_h, bot_y + context_tol + 1), x_lo:x_hi
                    ]
                    near_top = np.any(top_window > 0) if top_window.size > 0 else False
                    near_bot = np.any(bot_window > 0) if bot_window.size > 0 else False
                    if h < both_threshold:
                        if not (near_top and near_bot):
                            continue
                    else:
                        if not (near_top or near_bot):
                            continue
                y_new = max(0, y - extend_px)
                y_end = min(img_h, y + h + extend_px)
                extended.append((x, y_new, w, y_end - y_new))
        return extended

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ox = max(0, min(ax2, bx2) - max(ax1, bx1))
        oy = max(0, min(ay2, by2) - max(ay1, by1))
        inter = ox * oy
        if inter <= 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return float(inter) / float(max(area_a + area_b - inter, 1))

    @classmethod
    def _dedup_boxes(cls, boxes, thresh):
        if not boxes:
            return []
        boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        keep = []
        for b in boxes:
            if all(cls._iou(b, k) <= thresh for k in keep):
                keep.append(b)
        return keep
