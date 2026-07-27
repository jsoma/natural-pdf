from __future__ import annotations

from typing import Any


class ShapeDetectionService:
    """Service wrapper around the shape detection implementation."""

    def __init__(self, context):
        self._context = context

    def detect_lines(self, host: Any, **kwargs) -> Any:
        pdfs = getattr(host, "pdfs", None)
        if pdfs is not None:
            for pdf in pdfs:
                pages = getattr(pdf, "pages", None)
                if pages is None:
                    continue
                for page in pages:
                    detector = getattr(page, "detect_lines", None)
                    if callable(detector):
                        detector(**kwargs)
            return host

        pages = getattr(host, "pages", None)
        if pages is not None and not hasattr(host, "_page"):
            for page in pages:
                detector = getattr(page, "detect_lines", None)
                if callable(detector):
                    detector(**kwargs)
            return host

        from natural_pdf.services import _shape_detection_impl as _impl

        _impl.detect_lines(host, **kwargs)
        return host

    def detect_line_element_data(self, host: Any, **kwargs) -> list[dict[str, Any]]:
        """Return detected line dictionaries without adding LineElements to the host page."""
        from natural_pdf.services import _shape_detection_impl as _impl
        from natural_pdf.services._shape_detection_impl import LINE_DETECTION_PARAM_DEFAULTS

        params = {
            "resolution": 192,
            "source_label": "detected",
            "method": "projection",
            "horizontal": True,
            "vertical": True,
            "peak_threshold_h": 0.5,
            "min_gap_h": 5,
            "peak_threshold_v": 0.5,
            "min_gap_v": 5,
            "max_lines_h": None,
            "max_lines_v": None,
            "binarization_method": LINE_DETECTION_PARAM_DEFAULTS["binarization_method"],
            "adaptive_thresh_block_size": LINE_DETECTION_PARAM_DEFAULTS[
                "adaptive_thresh_block_size"
            ],
            "adaptive_thresh_C_val": LINE_DETECTION_PARAM_DEFAULTS["adaptive_thresh_C_val"],
            "morph_op_h": LINE_DETECTION_PARAM_DEFAULTS["morph_op_h"],
            "morph_kernel_h": LINE_DETECTION_PARAM_DEFAULTS["morph_kernel_h"],
            "morph_op_v": LINE_DETECTION_PARAM_DEFAULTS["morph_op_v"],
            "morph_kernel_v": LINE_DETECTION_PARAM_DEFAULTS["morph_kernel_v"],
            "smoothing_sigma_h": LINE_DETECTION_PARAM_DEFAULTS["smoothing_sigma_h"],
            "smoothing_sigma_v": LINE_DETECTION_PARAM_DEFAULTS["smoothing_sigma_v"],
            "peak_width_rel_height": LINE_DETECTION_PARAM_DEFAULTS["peak_width_rel_height"],
            "off_angle": 5,
            "min_line_length": 30,
            "merge_angle_tolerance": 5,
            "merge_distance_tolerance": 3,
            "merge_endpoint_tolerance": 10,
            "initial_min_line_length": 10,
            "min_nfa_score_horizontal": -10.0,
            "min_nfa_score_vertical": -10.0,
        }
        kwargs.pop("replace", None)
        params.update(kwargs)

        pdfs = getattr(host, "pdfs", None)
        if pdfs is not None:
            results: list[dict[str, Any]] = []
            for pdf in pdfs:
                pages = getattr(pdf, "pages", None)
                if pages is None:
                    continue
                for page in pages:
                    results.extend(self.detect_line_element_data(page, **params))
            return results

        pages = getattr(host, "pages", None)
        if pages is not None and not hasattr(host, "_page"):
            results = []
            for page in pages:
                results.extend(self.detect_line_element_data(page, **params))
            return results

        return _impl.detect_line_element_data(host, **params)
