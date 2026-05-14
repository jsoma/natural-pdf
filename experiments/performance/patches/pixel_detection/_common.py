"""Shared pixel line-detection experiment helpers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Optional, Tuple

import numpy as np


def metric_count(label: str, amount: int | float = 1) -> None:
    try:
        from experiments.performance import vector_metrics

        vector_metrics.count(label, amount)
    except Exception:
        return


def _otsu_threshold(image: np.ndarray) -> int:
    hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(float)
    total_pixels = image.size
    current_max = 0.0
    threshold = 0
    sum_total = np.sum(np.arange(256) * hist)
    sum_background = 0.0
    weight_background = 0.0

    for i in range(256):
        weight_background += hist[i]
        if weight_background == 0:
            continue
        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break
        sum_background += i * hist[i]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance_between = (
            weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        )
        if variance_between > current_max:
            current_max = variance_between
            threshold = i
    return threshold


def _gray_numpy(cv_image: np.ndarray) -> np.ndarray:
    if len(cv_image.shape) == 3:
        # Integer approximation of standard luminance weights.
        rgb = cv_image[..., :3].astype(np.uint16, copy=False)
        return ((77 * rgb[..., 0] + 150 * rgb[..., 1] + 29 * rgb[..., 2]) >> 8).astype(np.uint8)
    return cv_image


def _binarize_numpy(
    gray_image: np.ndarray,
    *,
    binarization_method: str,
    adaptive_thresh_block_size: int,
    adaptive_thresh_C_val: int,
) -> np.ndarray:
    if binarization_method == "adaptive":
        from scipy.ndimage import gaussian_filter

        sigma = adaptive_thresh_block_size / 6.0
        local_mean = gaussian_filter(gray_image.astype(float), sigma=sigma)
        return np.where(gray_image > (local_mean - adaptive_thresh_C_val), 0, 255).astype(np.uint8)
    if binarization_method == "otsu":
        threshold = _otsu_threshold(gray_image)
        return (gray_image <= threshold).astype(np.uint8) * 255

    threshold = _otsu_threshold(gray_image)
    return (gray_image <= threshold).astype(np.uint8) * 255


def _binarize_cv2(
    gray_image: np.ndarray,
    *,
    binarization_method: str,
    adaptive_thresh_block_size: int,
    adaptive_thresh_C_val: int,
) -> np.ndarray:
    import cv2

    if binarization_method == "adaptive":
        block = max(3, int(adaptive_thresh_block_size))
        if block % 2 == 0:
            block += 1
        return cv2.adaptiveThreshold(
            gray_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block,
            adaptive_thresh_C_val,
        )
    if binarization_method == "otsu":
        _, binary = cv2.threshold(
            gray_image,
            0,
            255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
        )
        return binary
    _, binary = cv2.threshold(
        gray_image,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    return binary


def _morph_numpy(image: np.ndarray, operation: str, kernel_size: Tuple[int, int]) -> np.ndarray:
    if operation == "none":
        return image
    from scipy.ndimage import binary_closing, binary_opening

    cols, rows = kernel_size
    structure = np.ones((rows, cols), dtype=bool)
    binary = image > 0
    if operation == "open":
        return binary_opening(binary, structure=structure)
    if operation == "close":
        return binary_closing(binary, structure=structure)
    return binary


def _morph_cv2(image: np.ndarray, operation: str, kernel_size: Tuple[int, int]) -> np.ndarray:
    if operation == "none":
        return image
    import cv2

    cols, rows = kernel_size
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols, rows))
    if operation == "open":
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    if operation == "close":
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return image


def find_lines_uint8(
    original,
    self,
    cv_image,
    pil_image_rgb,
    horizontal=True,
    vertical=True,
    peak_threshold_h=0.5,
    min_gap_h=5,
    peak_threshold_v=0.5,
    min_gap_v=5,
    max_lines_h: Optional[int] = None,
    max_lines_v: Optional[int] = None,
    binarization_method="adaptive",
    adaptive_thresh_block_size=21,
    adaptive_thresh_C_val=5,
    morph_op_h="none",
    morph_kernel_h=(1, 2),
    morph_op_v="none",
    morph_kernel_v=(2, 1),
    smoothing_sigma_h=0.6,
    smoothing_sigma_v=0.6,
    peak_width_rel_height=0.5,
):
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    if cv_image is None:
        return [], None, None

    metric_count("pixel_detection.uint8_pipeline")
    gray_image = _gray_numpy(cv_image)
    binarized = _binarize_numpy(
        gray_image,
        binarization_method=binarization_method,
        adaptive_thresh_block_size=adaptive_thresh_block_size,
        adaptive_thresh_C_val=adaptive_thresh_C_val,
    )

    detected_lines_data = []
    profile_h_smoothed_for_viz = None
    profile_v_smoothed_for_viz = None

    def get_lines_from_profile(profile_data, max_dimension_for_ratio, is_horizontal_detection):
        sigma = smoothing_sigma_h if is_horizontal_detection else smoothing_sigma_v
        profile_smoothed = gaussian_filter1d(profile_data.astype(float), sigma=sigma)
        peak_threshold = peak_threshold_h if is_horizontal_detection else peak_threshold_v
        min_gap = min_gap_h if is_horizontal_detection else min_gap_v
        max_lines = max_lines_h if is_horizontal_detection else max_lines_v
        current_peak_height_threshold = peak_threshold * max_dimension_for_ratio
        find_peaks_distance = min_gap

        if max_lines is not None:
            current_peak_height_threshold = 1.0
            find_peaks_distance = 1

        candidate_peaks_indices, candidate_properties = find_peaks(
            profile_smoothed,
            height=current_peak_height_threshold,
            distance=find_peaks_distance,
            width=1,
            prominence=1,
            rel_height=peak_width_rel_height,
        )
        final_peaks_indices = candidate_peaks_indices
        final_properties = candidate_properties

        if max_lines is not None:
            if len(candidate_peaks_indices) > 0 and "prominences" in candidate_properties:
                prominences = candidate_properties["prominences"]
                sorted_candidate_indices_by_prominence = np.argsort(prominences)[::-1]
                selected_peaks_original_indices = []
                suppressed_profile_indices = np.zeros(len(profile_smoothed), dtype=bool)
                for original_idx_in_candidate_list in sorted_candidate_indices_by_prominence:
                    actual_profile_idx = candidate_peaks_indices[original_idx_in_candidate_list]
                    if suppressed_profile_indices[actual_profile_idx]:
                        continue
                    selected_peaks_original_indices.append(original_idx_in_candidate_list)
                    lower_bound = max(0, actual_profile_idx - min_gap)
                    upper_bound = min(len(profile_smoothed), actual_profile_idx + min_gap + 1)
                    suppressed_profile_indices[lower_bound:upper_bound] = True
                    if len(selected_peaks_original_indices) >= max_lines:
                        break
                final_peaks_indices = candidate_peaks_indices[selected_peaks_original_indices]
                final_properties = {
                    key: val_array[selected_peaks_original_indices]
                    for key, val_array in candidate_properties.items()
                }
            else:
                final_peaks_indices = np.array([])
                final_properties = {}
        elif not final_peaks_indices.size:
            final_properties = {}

        if final_peaks_indices.size > 0:
            sort_order = np.argsort(final_peaks_indices)
            final_peaks_indices = final_peaks_indices[sort_order]
            for key in final_properties:
                final_properties[key] = final_properties[key][sort_order]

        lines_info = []
        for i, peak_idx in enumerate(final_peaks_indices):
            center_coord = int(peak_idx)
            profile_thickness = (
                final_properties.get("widths", [])[i]
                if "widths" in final_properties and i < len(final_properties["widths"])
                else 1.0
            )
            profile_thickness = max(1, int(round(profile_thickness)))
            if is_horizontal_detection:
                lines_info.append(
                    {
                        "x1": 0,
                        "y1": center_coord,
                        "x2": pil_image_rgb.width - 1,
                        "y2": center_coord,
                        "width": profile_thickness,
                        "length": pil_image_rgb.width,
                        "line_thickness_px": profile_thickness,
                        "line_position_px": center_coord,
                    }
                )
            else:
                lines_info.append(
                    {
                        "x1": center_coord,
                        "y1": 0,
                        "x2": center_coord,
                        "y2": pil_image_rgb.height - 1,
                        "width": profile_thickness,
                        "length": pil_image_rgb.height,
                        "line_thickness_px": profile_thickness,
                        "line_position_px": center_coord,
                    }
                )
        return lines_info, profile_smoothed

    if horizontal:
        processed = _morph_numpy(binarized, morph_op_h, morph_kernel_h)
        profile = np.count_nonzero(processed, axis=1)
        horizontal_lines, profile_h_smoothed_for_viz = get_lines_from_profile(
            profile, pil_image_rgb.width, True
        )
        detected_lines_data.extend(horizontal_lines)

    if vertical:
        processed = _morph_numpy(binarized, morph_op_v, morph_kernel_v)
        profile = np.count_nonzero(processed, axis=0)
        vertical_lines, profile_v_smoothed_for_viz = get_lines_from_profile(
            profile, pil_image_rgb.height, False
        )
        detected_lines_data.extend(vertical_lines)

    return detected_lines_data, profile_h_smoothed_for_viz, profile_v_smoothed_for_viz


def find_lines_cv2(
    original,
    self,
    cv_image,
    pil_image_rgb,
    horizontal=True,
    vertical=True,
    **kwargs,
):
    try:
        import cv2
    except Exception:
        metric_count("pixel_detection.cv2_fallback")
        return original(
            self, cv_image, pil_image_rgb, horizontal=horizontal, vertical=vertical, **kwargs
        )

    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    if cv_image is None:
        return [], None, None

    metric_count("pixel_detection.cv2_pipeline")
    if len(cv_image.shape) == 3:
        gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = cv_image

    binarized = _binarize_cv2(
        gray,
        binarization_method=kwargs.get("binarization_method", "adaptive"),
        adaptive_thresh_block_size=kwargs.get("adaptive_thresh_block_size", 21),
        adaptive_thresh_C_val=kwargs.get("adaptive_thresh_C_val", 5),
    )

    peak_threshold_h = kwargs.get("peak_threshold_h", 0.5)
    min_gap_h = kwargs.get("min_gap_h", 5)
    peak_threshold_v = kwargs.get("peak_threshold_v", 0.5)
    min_gap_v = kwargs.get("min_gap_v", 5)
    max_lines_h = kwargs.get("max_lines_h")
    max_lines_v = kwargs.get("max_lines_v")
    smoothing_sigma_h = kwargs.get("smoothing_sigma_h", 0.6)
    smoothing_sigma_v = kwargs.get("smoothing_sigma_v", 0.6)
    peak_width_rel_height = kwargs.get("peak_width_rel_height", 0.5)

    def get_lines(profile_data, max_dimension_for_ratio, is_horizontal_detection):
        sigma = smoothing_sigma_h if is_horizontal_detection else smoothing_sigma_v
        profile_smoothed = gaussian_filter1d(profile_data.astype(float), sigma=sigma)
        peak_threshold = peak_threshold_h if is_horizontal_detection else peak_threshold_v
        min_gap = min_gap_h if is_horizontal_detection else min_gap_v
        max_lines = max_lines_h if is_horizontal_detection else max_lines_v
        height = peak_threshold * max_dimension_for_ratio
        distance = min_gap
        if max_lines is not None:
            height = 1.0
            distance = 1
        peaks, props = find_peaks(
            profile_smoothed,
            height=height,
            distance=distance,
            width=1,
            prominence=1,
            rel_height=peak_width_rel_height,
        )
        if max_lines is not None and len(peaks) and "prominences" in props:
            order = np.argsort(props["prominences"])[::-1]
            selected = []
            suppressed = np.zeros(len(profile_smoothed), dtype=bool)
            for candidate_idx in order:
                peak_idx = peaks[candidate_idx]
                if suppressed[peak_idx]:
                    continue
                selected.append(candidate_idx)
                suppressed[
                    max(0, peak_idx - min_gap) : min(len(suppressed), peak_idx + min_gap + 1)
                ] = True
                if len(selected) >= max_lines:
                    break
            peaks = peaks[selected]
            props = {key: val[selected] for key, val in props.items()}
        elif max_lines is not None:
            peaks = np.array([])
            props = {}

        if len(peaks):
            order = np.argsort(peaks)
            peaks = peaks[order]
            for key in props:
                props[key] = props[key][order]

        lines = []
        for i, peak_idx in enumerate(peaks):
            thickness = (
                props.get("widths", [])[i]
                if "widths" in props and i < len(props["widths"])
                else 1.0
            )
            thickness = max(1, int(round(thickness)))
            center = int(peak_idx)
            if is_horizontal_detection:
                lines.append(
                    {
                        "x1": 0,
                        "y1": center,
                        "x2": pil_image_rgb.width - 1,
                        "y2": center,
                        "width": thickness,
                        "length": pil_image_rgb.width,
                        "line_thickness_px": thickness,
                        "line_position_px": center,
                    }
                )
            else:
                lines.append(
                    {
                        "x1": center,
                        "y1": 0,
                        "x2": center,
                        "y2": pil_image_rgb.height - 1,
                        "width": thickness,
                        "length": pil_image_rgb.height,
                        "line_thickness_px": thickness,
                        "line_position_px": center,
                    }
                )
        return lines, profile_smoothed

    detected = []
    profile_h = None
    profile_v = None
    if horizontal:
        processed = _morph_cv2(
            binarized,
            kwargs.get("morph_op_h", "none"),
            kwargs.get("morph_kernel_h", (1, 2)),
        )
        h_lines, profile_h = get_lines(
            np.count_nonzero(processed, axis=1), pil_image_rgb.width, True
        )
        detected.extend(h_lines)
    if vertical:
        processed = _morph_cv2(
            binarized,
            kwargs.get("morph_op_v", "none"),
            kwargs.get("morph_kernel_v", (2, 1)),
        )
        v_lines, profile_v = get_lines(
            np.count_nonzero(processed, axis=0), pil_image_rgb.height, False
        )
        detected.extend(v_lines)
    return detected, profile_h, profile_v


def find_lines_fast_topk(
    original,
    self,
    cv_image,
    pil_image_rgb,
    horizontal=True,
    vertical=True,
    **kwargs,
):
    from scipy.ndimage import gaussian_filter1d

    if cv_image is None:
        return [], None, None
    if kwargs.get("max_lines_h") is None and kwargs.get("max_lines_v") is None:
        return find_lines_uint8(
            original,
            self,
            cv_image,
            pil_image_rgb,
            horizontal=horizontal,
            vertical=vertical,
            **kwargs,
        )

    metric_count("pixel_detection.fast_topk")
    gray = _gray_numpy(cv_image)
    binarized = _binarize_numpy(
        gray,
        binarization_method=kwargs.get("binarization_method", "adaptive"),
        adaptive_thresh_block_size=kwargs.get("adaptive_thresh_block_size", 21),
        adaptive_thresh_C_val=kwargs.get("adaptive_thresh_C_val", 5),
    )

    def topk_lines(profile_data, max_dimension, is_horizontal):
        sigma = kwargs.get("smoothing_sigma_h" if is_horizontal else "smoothing_sigma_v", 0.6)
        profile = gaussian_filter1d(profile_data.astype(float), sigma=sigma)
        max_lines = kwargs.get("max_lines_h" if is_horizontal else "max_lines_v")
        min_gap = kwargs.get("min_gap_h" if is_horizontal else "min_gap_v", 5)
        threshold = kwargs.get("peak_threshold_h" if is_horizontal else "peak_threshold_v", 0.5)
        height = 1.0 if max_lines is not None else threshold * max_dimension

        if len(profile) < 3:
            return [], profile
        candidates = np.flatnonzero(
            (profile >= height)
            & (profile >= np.r_[profile[0], profile[:-1]])
            & (profile >= np.r_[profile[1:], profile[-1]])
        )
        if not len(candidates):
            return [], profile
        order = candidates[np.argsort(profile[candidates])[::-1]]
        selected = []
        suppressed = np.zeros(len(profile), dtype=bool)
        limit = max_lines or len(order)
        for idx in order:
            if suppressed[idx]:
                continue
            selected.append(int(idx))
            suppressed[max(0, idx - min_gap) : min(len(profile), idx + min_gap + 1)] = True
            if len(selected) >= limit:
                break
        selected.sort()

        lines = []
        for center in selected:
            if is_horizontal:
                lines.append(
                    {
                        "x1": 0,
                        "y1": center,
                        "x2": pil_image_rgb.width - 1,
                        "y2": center,
                        "width": 1,
                        "length": pil_image_rgb.width,
                        "line_thickness_px": 1,
                        "line_position_px": center,
                    }
                )
            else:
                lines.append(
                    {
                        "x1": center,
                        "y1": 0,
                        "x2": center,
                        "y2": pil_image_rgb.height - 1,
                        "width": 1,
                        "length": pil_image_rgb.height,
                        "line_thickness_px": 1,
                        "line_position_px": center,
                    }
                )
        return lines, profile

    detected = []
    profile_h = None
    profile_v = None
    if horizontal:
        processed = _morph_numpy(
            binarized,
            kwargs.get("morph_op_h", "none"),
            kwargs.get("morph_kernel_h", (1, 2)),
        )
        h_lines, profile_h = topk_lines(
            np.count_nonzero(processed, axis=1), pil_image_rgb.width, True
        )
        detected.extend(h_lines)
    if vertical:
        processed = _morph_numpy(
            binarized,
            kwargs.get("morph_op_v", "none"),
            kwargs.get("morph_kernel_v", (2, 1)),
        )
        v_lines, profile_v = topk_lines(
            np.count_nonzero(processed, axis=0), pil_image_rgb.height, False
        )
        detected.extend(v_lines)
    return detected, profile_h, profile_v


def install_find_lines_patch(replacement):
    @contextmanager
    def manager():
        from natural_pdf.analyzers.shape_detection_mixin import ShapeDetectionMixin

        original = ShapeDetectionMixin._find_lines_on_image_data

        def patched(self, *args, **kwargs):
            return replacement(original, self, *args, **kwargs)

        ShapeDetectionMixin._find_lines_on_image_data = patched
        try:
            yield
        finally:
            ShapeDetectionMixin._find_lines_on_image_data = original

    return manager()


def install_resolution_patch(resolution: int):
    @contextmanager
    def manager():
        from natural_pdf.guides.engines.lines import LinesGuidesEngine

        original = LinesGuidesEngine._detect_coordinates

        def patched(self, *, context, options, axis_label):
            patched_options = dict(options)
            if patched_options.get("detection_method", "auto") in ("auto", "pixels"):
                if patched_options.get("resolution", 192) == 192:
                    metric_count(f"pixel_detection.resolution_{resolution}")
                    patched_options["resolution"] = resolution
            return original(self, context=context, options=patched_options, axis_label=axis_label)

        LinesGuidesEngine._detect_coordinates = patched
        try:
            yield
        finally:
            LinesGuidesEngine._detect_coordinates = original

    return manager()
