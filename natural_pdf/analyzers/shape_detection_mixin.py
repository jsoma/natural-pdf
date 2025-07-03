import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_closing, binary_opening, find_objects, gaussian_filter1d
from scipy.ndimage import label as nd_label
from scipy.signal import find_peaks
from sklearn.cluster import MiniBatchKMeans

if TYPE_CHECKING:
    from natural_pdf.core.page import Page
    from natural_pdf.core.page_collection import PageCollection
    from natural_pdf.core.pdf import PDF
    from natural_pdf.elements.element_collection import ElementCollection
    from natural_pdf.elements.line import LineElement

    # from natural_pdf.elements.rect import RectangleElement # Removed
    from natural_pdf.elements.region import Region

logger = logging.getLogger(__name__)

# Constants for default values of less commonly adjusted line detection parameters
LINE_DETECTION_PARAM_DEFAULTS = {
    "binarization_method": "adaptive",
    "adaptive_thresh_block_size": 21,
    "adaptive_thresh_C_val": 5,
    "morph_op_h": "none",
    "morph_kernel_h": (1, 2),  # Kernel as (columns, rows)
    "morph_op_v": "none",
    "morph_kernel_v": (2, 1),  # Kernel as (columns, rows)
    "smoothing_sigma_h": 0.6,
    "smoothing_sigma_v": 0.6,
    "peak_width_rel_height": 0.5,
}


class ShapeDetectionMixin:
    """
    Mixin class to provide shape detection capabilities (lines)
    for Page, Region, PDFCollection, and PageCollection objects.
    """

    def _get_image_for_detection(
        self, resolution: int
    ) -> Tuple[Optional[np.ndarray], float, Tuple[float, float], Optional["Page"]]:
        """
        Gets the image for detection, scale factor, PDF origin offset, and the relevant page object.

        Returns:
            Tuple containing:
                - cv_image (np.ndarray, optional): The OpenCV image array.
                - scale_factor (float): Factor to convert image pixels to PDF points.
                - origin_offset_pdf (Tuple[float, float]): (x0, top) offset in PDF points.
                - page_obj (Page, optional): The page object this detection pertains to.
        """
        pil_image = None
        page_obj = None
        origin_offset_pdf = (0.0, 0.0)

        # Determine the type of self and get the appropriate image and page context
        if (
            hasattr(self, "render") and hasattr(self, "width") and hasattr(self, "height")
        ):  # Page or Region
            if hasattr(self, "x0") and hasattr(self, "top") and hasattr(self, "_page"):  # Region
                logger.debug(f"Shape detection on Region: {self}")
                page_obj = self._page
                # Use render() for clean image without highlights, with cropping
                pil_image = self.render(resolution=resolution, crop=True)
                if pil_image:  # Ensure pil_image is not None before accessing attributes
                    origin_offset_pdf = (self.x0, self.top)
                    logger.debug(
                        f"Region image rendered successfully: {pil_image.width}x{pil_image.height}, origin_offset: {origin_offset_pdf}"
                    )
            else:  # Page
                logger.debug(f"Shape detection on Page: {self}")
                page_obj = self
                # Use render() for clean image without highlights
                pil_image = self.render(resolution=resolution)
                logger.debug(
                    f"Page image rendered successfully: {pil_image.width}x{pil_image.height}"
                )
        else:
            logger.error(f"Instance of type {type(self)} does not support to_image for detection.")
            return None, 1.0, (0.0, 0.0), None

        if not pil_image:
            logger.warning("Failed to render image for shape detection.")
            return None, 1.0, (0.0, 0.0), page_obj

        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        cv_image = np.array(pil_image)

        # Calculate scale_factor: points_per_pixel
        # For a Page, self.width/height are PDF points. pil_image.width/height are pixels.
        # For a Region, self.width/height are PDF points of the region. pil_image.width/height are pixels of the cropped image.
        # The scale factor should always relate the dimensions of the *processed image* to the *PDF dimensions* of that same area.

        if page_obj and pil_image.width > 0 and pil_image.height > 0:
            # If it's a region, its self.width/height are its dimensions in PDF points.
            # pil_image.width/height are the pixel dimensions of the cropped image of that region.
            # So, the scale factor remains consistent.
            # We need to convert pixel distances on the image back to PDF point distances.
            # If 100 PDF points span 200 pixels, then 1 pixel = 0.5 PDF points. scale_factor = points/pixels
            # Example: Page width 500pt, image width 1000px. Scale = 500/1000 = 0.5 pt/px
            # Region width 50pt, cropped image width 100px. Scale = 50/100 = 0.5 pt/px

            # Use self.width/height for scale factor calculation because these correspond to the PDF dimensions of the area imaged.
            # This ensures that if self is a Region, its specific dimensions are used for scaling its own cropped image.

            # We need two scale factors if aspect ratio is not preserved by to_image,
            # but to_image generally aims to preserve it when only resolution is changed.
            # Assuming uniform scaling for now.
            # A robust way: scale_x = self.width / pil_image.width; scale_y = self.height / pil_image.height
            # For simplicity, let's assume uniform scaling or average it.
            # Average scale factor:
            scale_factor = ((self.width / pil_image.width) + (self.height / pil_image.height)) / 2.0
            logger.debug(
                f"Calculated scale_factor: {scale_factor:.4f} (PDF dimensions: {self.width:.1f}x{self.height:.1f}, Image: {pil_image.width}x{pil_image.height})"
            )

        else:
            logger.warning("Could not determine page object or image dimensions for scaling.")
            scale_factor = 1.0  # Default to no scaling if info is missing

        return cv_image, scale_factor, origin_offset_pdf, page_obj

    def _convert_line_to_element_data(
        self,
        line_data_img: Dict,
        scale_factor: float,
        origin_offset_pdf: Tuple[float, float],
        page_obj: "Page",
        source_label: str,
    ) -> Dict:
        """Converts line data from image coordinates to PDF element data."""
        # Ensure scale_factor is not zero to prevent division by zero or incorrect scaling
        if scale_factor == 0:
            logger.warning("Scale factor is zero, cannot convert line coordinates correctly.")
            # Return something or raise error, for now, try to proceed with unscaled if possible (won't be right)
            # This situation ideally shouldn't happen if _get_image_for_detection is robust.
            effective_scale = 1.0
        else:
            effective_scale = scale_factor

        x0 = origin_offset_pdf[0] + line_data_img["x1"] * effective_scale
        top = origin_offset_pdf[1] + line_data_img["y1"] * effective_scale
        x1 = origin_offset_pdf[0] + line_data_img["x2"] * effective_scale
        bottom = (
            origin_offset_pdf[1] + line_data_img["y2"] * effective_scale
        )  # y2 is the second y-coord

        # Clamp coords to image dimensions
        x0 = max(0, min(x0, page_obj.width))
        top = max(0, min(top, page_obj.height))
        x1 = max(0, min(x1, page_obj.width))
        bottom = max(0, min(bottom, page_obj.height))

        # For lines, width attribute in PDF points
        line_width_pdf = line_data_img["width"] * effective_scale

        # initial_doctop might not be loaded if page object is minimal
        initial_doctop = (
            getattr(page_obj._page, "initial_doctop", 0) if hasattr(page_obj, "_page") else 0
        )

        attrs = {
            "x0": x0,
            "top": top,
            "x1": x1,
            "bottom": bottom,  # bottom here is y2_pdf
            "width": abs(x1 - x0),  # This is bounding box width
            "height": abs(bottom - top),  # This is bounding box height
            "linewidth": line_width_pdf,  # Actual stroke width of the line
            "object_type": "line",
            "page_number": page_obj.page_number,
            "doctop": top + initial_doctop,
            "source": source_label,
            "stroking_color": (0, 0, 0),  # Default, can be enhanced
            "non_stroking_color": (0, 0, 0),  # Default
            # Add other raw data if useful
            "raw_line_thickness_px": line_data_img.get(
                "line_thickness_px"
            ),  # Renamed from raw_nfa_score
            "raw_line_position_px": line_data_img.get("line_position_px"),  # Added for clarity
        }

        return attrs

    def _find_lines_on_image_data(
        self,
        cv_image: np.ndarray,
        pil_image_rgb: Image.Image,  # For original dimensions
        horizontal: bool = True,
        vertical: bool = True,
        peak_threshold_h: float = 0.5,
        min_gap_h: int = 5,
        peak_threshold_v: float = 0.5,
        min_gap_v: int = 5,
        max_lines_h: Optional[int] = None,
        max_lines_v: Optional[int] = None,
        binarization_method: str = LINE_DETECTION_PARAM_DEFAULTS["binarization_method"],
        adaptive_thresh_block_size: int = LINE_DETECTION_PARAM_DEFAULTS[
            "adaptive_thresh_block_size"
        ],
        adaptive_thresh_C_val: int = LINE_DETECTION_PARAM_DEFAULTS["adaptive_thresh_C_val"],
        morph_op_h: str = LINE_DETECTION_PARAM_DEFAULTS["morph_op_h"],
        morph_kernel_h: Tuple[int, int] = LINE_DETECTION_PARAM_DEFAULTS["morph_kernel_h"],
        morph_op_v: str = LINE_DETECTION_PARAM_DEFAULTS["morph_op_v"],
        morph_kernel_v: Tuple[int, int] = LINE_DETECTION_PARAM_DEFAULTS["morph_kernel_v"],
        smoothing_sigma_h: float = LINE_DETECTION_PARAM_DEFAULTS["smoothing_sigma_h"],
        smoothing_sigma_v: float = LINE_DETECTION_PARAM_DEFAULTS["smoothing_sigma_v"],
        peak_width_rel_height: float = LINE_DETECTION_PARAM_DEFAULTS["peak_width_rel_height"],
    ) -> Tuple[List[Dict], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Core image processing logic to detect lines using projection profiling.
        Returns raw line data (image coordinates) and smoothed profiles.
        """
        if cv_image is None:
            return [], None, None

        # Convert RGB to grayscale using numpy (faster than PIL)
        # Using standard luminance weights: 0.299*R + 0.587*G + 0.114*B
        if len(cv_image.shape) == 3:
            gray_image = np.dot(cv_image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        else:
            gray_image = cv_image  # Already grayscale

        img_height, img_width = gray_image.shape
        logger.debug(f"Line detection - Image dimensions: {img_width}x{img_height}")

        def otsu_threshold(image):
            """Simple Otsu's thresholding implementation using numpy."""
            # Calculate histogram
            hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
            hist = hist.astype(float)

            # Calculate probabilities
            total_pixels = image.size
            current_max = 0
            threshold = 0
            sum_total = np.sum(np.arange(256) * hist)
            sum_background = 0
            weight_background = 0

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

                # Calculate between-class variance
                variance_between = (
                    weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
                )

                if variance_between > current_max:
                    current_max = variance_between
                    threshold = i

            return threshold

        def adaptive_threshold(image, block_size, C):
            """Simple adaptive thresholding implementation."""
            # Use scipy for gaussian filtering
            from scipy.ndimage import gaussian_filter

            # Calculate local means using gaussian filter
            sigma = block_size / 6.0  # Approximate relationship
            local_mean = gaussian_filter(image.astype(float), sigma=sigma)

            # Apply threshold
            binary = (image > (local_mean - C)).astype(np.uint8) * 255
            return 255 - binary  # Invert to match binary inverse thresholding

        if binarization_method == "adaptive":
            binarized_image = adaptive_threshold(
                gray_image, adaptive_thresh_block_size, adaptive_thresh_C_val
            )
        elif binarization_method == "otsu":
            otsu_thresh_val = otsu_threshold(gray_image)
            binarized_image = (gray_image <= otsu_thresh_val).astype(
                np.uint8
            ) * 255  # Inverted binary
            logger.debug(f"Otsu's threshold applied. Value: {otsu_thresh_val}")
        else:
            logger.error(
                f"Invalid binarization_method: {binarization_method}. Supported: 'otsu', 'adaptive'. Defaulting to 'otsu'."
            )
            otsu_thresh_val = otsu_threshold(gray_image)
            binarized_image = (gray_image <= otsu_thresh_val).astype(
                np.uint8
            ) * 255  # Inverted binary

        binarized_norm = binarized_image.astype(float) / 255.0

        detected_lines_data = []
        profile_h_smoothed_for_viz: Optional[np.ndarray] = None
        profile_v_smoothed_for_viz: Optional[np.ndarray] = None

        def get_lines_from_profile(
            profile_data: np.ndarray,
            max_dimension_for_ratio: int,
            params_key_suffix: str,
            is_horizontal_detection: bool,
        ) -> Tuple[List[Dict], np.ndarray]:  # Ensure it always returns profile_smoothed
            lines_info = []
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
                    num_selected = 0
                    for original_idx_in_candidate_list in sorted_candidate_indices_by_prominence:
                        actual_profile_idx = candidate_peaks_indices[original_idx_in_candidate_list]
                        if not suppressed_profile_indices[actual_profile_idx]:
                            selected_peaks_original_indices.append(original_idx_in_candidate_list)
                            num_selected += 1
                            lower_bound = max(0, actual_profile_idx - min_gap)
                            upper_bound = min(
                                len(profile_smoothed), actual_profile_idx + min_gap + 1
                            )
                            suppressed_profile_indices[lower_bound:upper_bound] = True
                            if num_selected >= max_lines:
                                break
                    final_peaks_indices = candidate_peaks_indices[selected_peaks_original_indices]
                    final_properties = {
                        key: val_array[selected_peaks_original_indices]
                        for key, val_array in candidate_properties.items()
                    }
                    logger.debug(
                        f"Selected {len(final_peaks_indices)} {params_key_suffix.upper()}-lines for max_lines={max_lines}."
                    )
                else:
                    final_peaks_indices = np.array([])
                    final_properties = {}
                    logger.debug(f"No {params_key_suffix.upper()}-peaks for max_lines selection.")
            elif not final_peaks_indices.size:
                final_properties = {}
                logger.debug(f"No {params_key_suffix.upper()}-lines found using threshold.")
            else:
                logger.debug(
                    f"Found {len(final_peaks_indices)} {params_key_suffix.upper()}-lines using threshold."
                )

            if final_peaks_indices.size > 0:
                sort_order = np.argsort(final_peaks_indices)
                final_peaks_indices = final_peaks_indices[sort_order]
                for key in final_properties:
                    final_properties[key] = final_properties[key][sort_order]

            for i, peak_idx in enumerate(final_peaks_indices):
                center_coord = int(peak_idx)
                profile_thickness = (
                    final_properties.get("widths", [])[i]
                    if "widths" in final_properties and i < len(final_properties["widths"])
                    else 1.0
                )
                profile_thickness = max(1, int(round(profile_thickness)))

                current_img_width = pil_image_rgb.width  # Use actual passed image dimensions
                current_img_height = pil_image_rgb.height

                if is_horizontal_detection:
                    lines_info.append(
                        {
                            "x1": 0,
                            "y1": center_coord,
                            "x2": current_img_width - 1,
                            "y2": center_coord,
                            "width": profile_thickness,
                            "length": current_img_width,
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
                            "y2": current_img_height - 1,
                            "width": profile_thickness,
                            "length": current_img_height,
                            "line_thickness_px": profile_thickness,
                            "line_position_px": center_coord,
                        }
                    )
            return lines_info, profile_smoothed

        def apply_morphology(image, operation, kernel_size):
            """Apply morphological operations using scipy.ndimage."""
            if operation == "none":
                return image

            # Create rectangular structuring element
            # kernel_size is (width, height) = (cols, rows)
            cols, rows = kernel_size
            structure = np.ones((rows, cols))  # Note: numpy uses (rows, cols) order

            # Convert to binary for morphological operations
            binary_img = (image > 0.5).astype(bool)

            if operation == "open":
                result = binary_opening(binary_img, structure=structure)
            elif operation == "close":
                result = binary_closing(binary_img, structure=structure)
            else:
                logger.warning(
                    f"Unknown morphological operation: {operation}. Supported: 'open', 'close', 'none'."
                )
                result = binary_img

            # Convert back to float
            return result.astype(float)

        if horizontal:
            processed_image_h = binarized_norm.copy()
            if morph_op_h != "none":
                processed_image_h = apply_morphology(processed_image_h, morph_op_h, morph_kernel_h)
            profile_h_raw = np.sum(processed_image_h, axis=1)
            horizontal_lines, smoothed_h = get_lines_from_profile(
                profile_h_raw, pil_image_rgb.width, "h", True
            )
            profile_h_smoothed_for_viz = smoothed_h
            detected_lines_data.extend(horizontal_lines)
            logger.info(f"Detected {len(horizontal_lines)} horizontal lines.")

        if vertical:
            processed_image_v = binarized_norm.copy()
            if morph_op_v != "none":
                processed_image_v = apply_morphology(processed_image_v, morph_op_v, morph_kernel_v)
            profile_v_raw = np.sum(processed_image_v, axis=0)
            vertical_lines, smoothed_v = get_lines_from_profile(
                profile_v_raw, pil_image_rgb.height, "v", False
            )
            profile_v_smoothed_for_viz = smoothed_v
            detected_lines_data.extend(vertical_lines)
            logger.info(f"Detected {len(vertical_lines)} vertical lines.")

        return detected_lines_data, profile_h_smoothed_for_viz, profile_v_smoothed_for_viz

    def detect_lines(
        self,
        resolution: int = 192,
        source_label: str = "detected",
        method: str = "projection",
        horizontal: bool = True,
        vertical: bool = True,
        peak_threshold_h: float = 0.5,
        min_gap_h: int = 5,
        peak_threshold_v: float = 0.5,
        min_gap_v: int = 5,
        max_lines_h: Optional[int] = None,
        max_lines_v: Optional[int] = None,
        replace: bool = True,
        binarization_method: str = LINE_DETECTION_PARAM_DEFAULTS["binarization_method"],
        adaptive_thresh_block_size: int = LINE_DETECTION_PARAM_DEFAULTS[
            "adaptive_thresh_block_size"
        ],
        adaptive_thresh_C_val: int = LINE_DETECTION_PARAM_DEFAULTS["adaptive_thresh_C_val"],
        morph_op_h: str = LINE_DETECTION_PARAM_DEFAULTS["morph_op_h"],
        morph_kernel_h: Tuple[int, int] = LINE_DETECTION_PARAM_DEFAULTS["morph_kernel_h"],
        morph_op_v: str = LINE_DETECTION_PARAM_DEFAULTS["morph_op_v"],
        morph_kernel_v: Tuple[int, int] = LINE_DETECTION_PARAM_DEFAULTS["morph_kernel_v"],
        smoothing_sigma_h: float = LINE_DETECTION_PARAM_DEFAULTS["smoothing_sigma_h"],
        smoothing_sigma_v: float = LINE_DETECTION_PARAM_DEFAULTS["smoothing_sigma_v"],
        peak_width_rel_height: float = LINE_DETECTION_PARAM_DEFAULTS["peak_width_rel_height"],
        # LSD-specific parameters
        off_angle: int = 5,
        min_line_length: int = 30,
        merge_angle_tolerance: int = 5,
        merge_distance_tolerance: int = 3,
        merge_endpoint_tolerance: int = 10,
        initial_min_line_length: int = 10,
        min_nfa_score_horizontal: float = -10.0,
        min_nfa_score_vertical: float = -10.0,
    ) -> "ShapeDetectionMixin":  # Return type changed back to self
        """
        Detects lines on the Page or Region, or on all pages within a Collection.
        Adds detected lines as LineElement objects to the ElementManager.

        Args:
            resolution: DPI for image rendering before detection.
            source_label: Label assigned to the 'source' attribute of created LineElements.
            method: Detection method - "projection" (default, no cv2 required) or "lsd" (requires opencv-python).
            horizontal: If True, detect horizontal lines.
            vertical: If True, detect vertical lines.

            # Projection profiling parameters:
            peak_threshold_h: Threshold for peak detection in horizontal profile (ratio of image width).
            min_gap_h: Minimum gap between horizontal lines (pixels).
            peak_threshold_v: Threshold for peak detection in vertical profile (ratio of image height).
            min_gap_v: Minimum gap between vertical lines (pixels).
            max_lines_h: If set, limits the number of horizontal lines to the top N by prominence.
            max_lines_v: If set, limits the number of vertical lines to the top N by prominence.
            replace: If True, remove existing detected lines with the same source_label.
            binarization_method: "adaptive" or "otsu".
            adaptive_thresh_block_size: Block size for adaptive thresholding (if method is "adaptive").
            adaptive_thresh_C_val: Constant subtracted from the mean for adaptive thresholding (if method is "adaptive").
            morph_op_h: Morphological operation for horizontal lines ("open", "close", "none").
            morph_kernel_h: Kernel tuple (cols, rows) for horizontal morphology. Example: (1, 2).
            morph_op_v: Morphological operation for vertical lines ("open", "close", "none").
            morph_kernel_v: Kernel tuple (cols, rows) for vertical morphology. Example: (2, 1).
            smoothing_sigma_h: Gaussian smoothing sigma for horizontal profile.
            smoothing_sigma_v: Gaussian smoothing sigma for vertical profile.
            peak_width_rel_height: Relative height for `scipy.find_peaks` 'width' parameter.

            # LSD-specific parameters (only used when method="lsd"):
            off_angle: Maximum angle deviation from horizontal/vertical for line classification.
            min_line_length: Minimum length for final detected lines.
            merge_angle_tolerance: Maximum angle difference for merging parallel lines.
            merge_distance_tolerance: Maximum perpendicular distance for merging lines.
            merge_endpoint_tolerance: Maximum gap at endpoints for merging lines.
            initial_min_line_length: Initial minimum length filter before merging.
            min_nfa_score_horizontal: Minimum NFA score for horizontal lines.
            min_nfa_score_vertical: Minimum NFA score for vertical lines.

        Returns:
            Self for method chaining.

        Raises:
            ImportError: If method="lsd" but opencv-python is not installed.
            ValueError: If method is not "projection" or "lsd".
        """
        if not horizontal and not vertical:
            logger.info("Line detection skipped as both horizontal and vertical are False.")
            return self

        # Validate method parameter
        if method not in ["projection", "lsd"]:
            raise ValueError(f"Invalid method '{method}'. Supported methods: 'projection', 'lsd'")

        collection_params = {
            "resolution": resolution,
            "source_label": source_label,
            "method": method,
            "horizontal": horizontal,
            "vertical": vertical,
            "peak_threshold_h": peak_threshold_h,
            "min_gap_h": min_gap_h,
            "peak_threshold_v": peak_threshold_v,
            "min_gap_v": min_gap_v,
            "max_lines_h": max_lines_h,
            "max_lines_v": max_lines_v,
            "replace": replace,
            "binarization_method": binarization_method,
            "adaptive_thresh_block_size": adaptive_thresh_block_size,
            "adaptive_thresh_C_val": adaptive_thresh_C_val,
            "morph_op_h": morph_op_h,
            "morph_kernel_h": morph_kernel_h,
            "morph_op_v": morph_op_v,
            "morph_kernel_v": morph_kernel_v,
            "smoothing_sigma_h": smoothing_sigma_h,
            "smoothing_sigma_v": smoothing_sigma_v,
            "peak_width_rel_height": peak_width_rel_height,
            # LSD parameters
            "off_angle": off_angle,
            "min_line_length": min_line_length,
            "merge_angle_tolerance": merge_angle_tolerance,
            "merge_distance_tolerance": merge_distance_tolerance,
            "merge_endpoint_tolerance": merge_endpoint_tolerance,
            "initial_min_line_length": initial_min_line_length,
            "min_nfa_score_horizontal": min_nfa_score_horizontal,
            "min_nfa_score_vertical": min_nfa_score_vertical,
        }

        if hasattr(self, "pdfs"):
            for pdf_doc in self.pdfs:
                for page_obj in pdf_doc.pages:
                    page_obj.detect_lines(**collection_params)
            return self
        elif hasattr(self, "pages") and not hasattr(self, "_page"):
            for page_obj in self.pages:
                page_obj.detect_lines(**collection_params)
            return self

        # Dispatch to appropriate detection method
        if method == "projection":
            return self._detect_lines_projection(
                resolution=resolution,
                source_label=source_label,
                horizontal=horizontal,
                vertical=vertical,
                peak_threshold_h=peak_threshold_h,
                min_gap_h=min_gap_h,
                peak_threshold_v=peak_threshold_v,
                min_gap_v=min_gap_v,
                max_lines_h=max_lines_h,
                max_lines_v=max_lines_v,
                replace=replace,
                binarization_method=binarization_method,
                adaptive_thresh_block_size=adaptive_thresh_block_size,
                adaptive_thresh_C_val=adaptive_thresh_C_val,
                morph_op_h=morph_op_h,
                morph_kernel_h=morph_kernel_h,
                morph_op_v=morph_op_v,
                morph_kernel_v=morph_kernel_v,
                smoothing_sigma_h=smoothing_sigma_h,
                smoothing_sigma_v=smoothing_sigma_v,
                peak_width_rel_height=peak_width_rel_height,
            )
        elif method == "lsd":
            return self._detect_lines_lsd(
                resolution=resolution,
                source_label=source_label,
                horizontal=horizontal,
                vertical=vertical,
                off_angle=off_angle,
                min_line_length=min_line_length,
                merge_angle_tolerance=merge_angle_tolerance,
                merge_distance_tolerance=merge_distance_tolerance,
                merge_endpoint_tolerance=merge_endpoint_tolerance,
                initial_min_line_length=initial_min_line_length,
                min_nfa_score_horizontal=min_nfa_score_horizontal,
                min_nfa_score_vertical=min_nfa_score_vertical,
                replace=replace,
            )
        else:
            # This should never happen due to validation above, but just in case
            raise ValueError(f"Unsupported method: {method}")

    def _detect_lines_projection(
        self,
        resolution: int,
        source_label: str,
        horizontal: bool,
        vertical: bool,
        peak_threshold_h: float,
        min_gap_h: int,
        peak_threshold_v: float,
        min_gap_v: int,
        max_lines_h: Optional[int],
        max_lines_v: Optional[int],
        replace: bool,
        binarization_method: str,
        adaptive_thresh_block_size: int,
        adaptive_thresh_C_val: int,
        morph_op_h: str,
        morph_kernel_h: Tuple[int, int],
        morph_op_v: str,
        morph_kernel_v: Tuple[int, int],
        smoothing_sigma_h: float,
        smoothing_sigma_v: float,
        peak_width_rel_height: float,
    ) -> "ShapeDetectionMixin":
        """Internal method for projection profiling line detection."""
        cv_image, scale_factor, origin_offset_pdf, page_object_ctx = self._get_image_for_detection(
            resolution
        )
        if cv_image is None or page_object_ctx is None:
            logger.warning(f"Skipping line detection for {self} due to image error.")
            return self

        pil_image_for_dims = None
        if hasattr(self, "render") and hasattr(self, "width") and hasattr(self, "height"):
            if hasattr(self, "x0") and hasattr(self, "top") and hasattr(self, "_page"):
                pil_image_for_dims = self.render(resolution=resolution, crop=True)
            else:
                # Use render() for clean image without highlights
                pil_image_for_dims = self.render(resolution=resolution)
        if pil_image_for_dims is None:
            logger.warning(f"Could not re-render PIL image for dimensions for {self}.")
            pil_image_for_dims = Image.fromarray(cv_image)  # Ensure it's not None

        if pil_image_for_dims.mode != "RGB":
            pil_image_for_dims = pil_image_for_dims.convert("RGB")

        if replace:
            from natural_pdf.elements.line import LineElement

            element_manager = page_object_ctx._element_mgr
            if hasattr(element_manager, "_elements") and "lines" in element_manager._elements:
                original_count = len(element_manager._elements["lines"])
                element_manager._elements["lines"] = [
                    line
                    for line in element_manager._elements["lines"]
                    if getattr(line, "source", None) != source_label
                ]
                removed_count = original_count - len(element_manager._elements["lines"])
                if removed_count > 0:
                    logger.info(
                        f"Removed {removed_count} existing lines with source '{source_label}' from {page_object_ctx}"
                    )
        lines_data_img, profile_h_smoothed, profile_v_smoothed = self._find_lines_on_image_data(
            cv_image=cv_image,
            pil_image_rgb=pil_image_for_dims,
            horizontal=horizontal,
            vertical=vertical,
            peak_threshold_h=peak_threshold_h,
            min_gap_h=min_gap_h,
            peak_threshold_v=peak_threshold_v,
            min_gap_v=min_gap_v,
            max_lines_h=max_lines_h,
            max_lines_v=max_lines_v,
            binarization_method=binarization_method,
            adaptive_thresh_block_size=adaptive_thresh_block_size,
            adaptive_thresh_C_val=adaptive_thresh_C_val,
            morph_op_h=morph_op_h,
            morph_kernel_h=morph_kernel_h,
            morph_op_v=morph_op_v,
            morph_kernel_v=morph_kernel_v,
            smoothing_sigma_h=smoothing_sigma_h,
            smoothing_sigma_v=smoothing_sigma_v,
            peak_width_rel_height=peak_width_rel_height,
        )
        from natural_pdf.elements.line import LineElement

        element_manager = page_object_ctx._element_mgr

        for line_data_item_img in lines_data_img:
            element_constructor_data = self._convert_line_to_element_data(
                line_data_item_img, scale_factor, origin_offset_pdf, page_object_ctx, source_label
            )
            line_element = LineElement(element_constructor_data, page_object_ctx)
            element_manager.add_element(line_element, element_type="lines")

        logger.info(
            f"Detected and added {len(lines_data_img)} lines to {page_object_ctx} with source '{source_label}' using projection profiling."
        )
        return self

    def _detect_lines_lsd(
        self,
        resolution: int,
        source_label: str,
        horizontal: bool,
        vertical: bool,
        off_angle: int,
        min_line_length: int,
        merge_angle_tolerance: int,
        merge_distance_tolerance: int,
        merge_endpoint_tolerance: int,
        initial_min_line_length: int,
        min_nfa_score_horizontal: float,
        min_nfa_score_vertical: float,
        replace: bool,
    ) -> "ShapeDetectionMixin":
        """Internal method for LSD line detection."""
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "OpenCV (cv2) is required for LSD line detection. "
                "Install it with: pip install opencv-python\n"
                "Alternatively, use method='projection' which requires no additional dependencies."
            )

        cv_image, scale_factor, origin_offset_pdf, page_object_ctx = self._get_image_for_detection(
            resolution
        )
        if cv_image is None or page_object_ctx is None:
            logger.warning(f"Skipping LSD line detection for {self} due to image error.")
            return self

        if replace:
            from natural_pdf.elements.line import LineElement

            element_manager = page_object_ctx._element_mgr
            if hasattr(element_manager, "_elements") and "lines" in element_manager._elements:
                original_count = len(element_manager._elements["lines"])
                element_manager._elements["lines"] = [
                    line
                    for line in element_manager._elements["lines"]
                    if getattr(line, "source", None) != source_label
                ]
                removed_count = original_count - len(element_manager._elements["lines"])
                if removed_count > 0:
                    logger.info(
                        f"Removed {removed_count} existing lines with source '{source_label}' from {page_object_ctx}"
                    )

        lines_data_img = self._process_image_for_lines_lsd(
            cv_image,
            off_angle,
            min_line_length,
            merge_angle_tolerance,
            merge_distance_tolerance,
            merge_endpoint_tolerance,
            initial_min_line_length,
            min_nfa_score_horizontal,
            min_nfa_score_vertical,
        )

        from natural_pdf.elements.line import LineElement

        element_manager = page_object_ctx._element_mgr

        for line_data_item_img in lines_data_img:
            element_constructor_data = self._convert_line_to_element_data(
                line_data_item_img, scale_factor, origin_offset_pdf, page_object_ctx, source_label
            )
            line_element = LineElement(element_constructor_data, page_object_ctx)
            element_manager.add_element(line_element, element_type="lines")

        logger.info(
            f"Detected and added {len(lines_data_img)} lines to {page_object_ctx} with source '{source_label}' using LSD."
        )
        return self

    def _process_image_for_lines_lsd(
        self,
        cv_image: np.ndarray,
        off_angle: int,
        min_line_length: int,
        merge_angle_tolerance: int,
        merge_distance_tolerance: int,
        merge_endpoint_tolerance: int,
        initial_min_line_length: int,
        min_nfa_score_horizontal: float,
        min_nfa_score_vertical: float,
    ) -> List[Dict]:
        """Processes an image to detect lines using OpenCV LSD and merging logic."""
        import cv2  # Import is already validated in calling method

        if cv_image is None:
            return []

        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
        coords_arr, widths_arr, precs_arr, nfa_scores_arr = lsd.detect(gray_image)

        lines_raw = []
        if coords_arr is not None:  # nfa_scores_arr can be None if no lines are found
            nfa_scores_list = (
                nfa_scores_arr.flatten() if nfa_scores_arr is not None else [0.0] * len(coords_arr)
            )
            widths_list = (
                widths_arr.flatten() if widths_arr is not None else [1.0] * len(coords_arr)
            )
            precs_list = precs_arr.flatten() if precs_arr is not None else [0.0] * len(coords_arr)

            for i in range(len(coords_arr)):
                lines_raw.append(
                    (
                        coords_arr[i][0],
                        widths_list[i] if i < len(widths_list) else 1.0,
                        precs_list[i] if i < len(precs_list) else 0.0,
                        nfa_scores_list[i] if i < len(nfa_scores_list) else 0.0,
                    )
                )

        def get_line_properties(line_data_item):
            l_coords, l_width, l_prec, l_nfa_score = line_data_item
            x1, y1, x2, y2 = l_coords
            angle_rad = np.arctan2(y2 - y1, x2 - x1)
            angle_deg = np.degrees(angle_rad)
            normalized_angle_deg = angle_deg % 180
            if normalized_angle_deg < 0:
                normalized_angle_deg += 180

            is_h = (
                abs(normalized_angle_deg) <= off_angle
                or abs(normalized_angle_deg - 180) <= off_angle
            )
            is_v = abs(normalized_angle_deg - 90) <= off_angle

            if is_h and x1 > x2:
                x1, x2, y1, y2 = x2, x1, y2, y1
            elif is_v and y1 > y2:
                y1, y2, x1, x2 = y2, y1, x2, x1

            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            return {
                "coords": (x1, y1, x2, y2),
                "width": l_width,
                "prec": l_prec,
                "angle_deg": normalized_angle_deg,
                "is_horizontal": is_h,
                "is_vertical": is_v,
                "length": length,
                "nfa_score": l_nfa_score,
            }

        processed_lines = [get_line_properties(ld) for ld in lines_raw]

        filtered_lines = []
        for p in processed_lines:
            if p["length"] <= initial_min_line_length:
                continue
            if p["is_horizontal"] and p["nfa_score"] >= min_nfa_score_horizontal:
                filtered_lines.append(p)
            elif p["is_vertical"] and p["nfa_score"] >= min_nfa_score_vertical:
                filtered_lines.append(p)

        horizontal_lines = [p for p in filtered_lines if p["is_horizontal"]]
        vertical_lines = [p for p in filtered_lines if p["is_vertical"]]

        def merge_lines_list(lines_list, is_horizontal_merge):
            if not lines_list:
                return []
            key_sort = (
                (lambda p: (p["coords"][1], p["coords"][0]))
                if is_horizontal_merge
                else (lambda p: (p["coords"][0], p["coords"][1]))
            )
            lines_list.sort(key=key_sort)

            merged_results = []
            merged_flags = [False] * len(lines_list)

            for i, current_line_props in enumerate(lines_list):
                if merged_flags[i]:
                    continue
                group = [current_line_props]
                merged_flags[i] = True

                # Keep trying to expand the group until no more lines can be added
                # Use multiple passes to ensure transitive merging works properly
                for merge_pass in range(10):  # Up to 10 passes to catch complex merging scenarios
                    group_changed = False

                    # Calculate current group boundaries
                    group_x1, group_y1 = min(p["coords"][0] for p in group), min(
                        p["coords"][1] for p in group
                    )
                    group_x2, group_y2 = max(p["coords"][2] for p in group), max(
                        p["coords"][3] for p in group
                    )
                    total_len_in_group = sum(p["length"] for p in group)
                    if total_len_in_group == 0:
                        continue  # Should not happen

                    # Calculate weighted averages for the group
                    group_avg_angle = (
                        sum(p["angle_deg"] * p["length"] for p in group) / total_len_in_group
                    )

                    if is_horizontal_merge:
                        group_avg_perp_coord = (
                            sum(
                                ((p["coords"][1] + p["coords"][3]) / 2) * p["length"] for p in group
                            )
                            / total_len_in_group
                        )
                    else:
                        group_avg_perp_coord = (
                            sum(
                                ((p["coords"][0] + p["coords"][2]) / 2) * p["length"] for p in group
                            )
                            / total_len_in_group
                        )

                    # Check all unmerged lines for potential merging
                    for j, candidate_props in enumerate(lines_list):
                        if merged_flags[j]:
                            continue

                        # 1. Check for parallelism (angle similarity)
                        angle_diff = abs(group_avg_angle - candidate_props["angle_deg"])
                        # Handle wraparound for angles near 0/180
                        if angle_diff > 90:
                            angle_diff = 180 - angle_diff
                        if angle_diff > merge_angle_tolerance:
                            continue

                        # 2. Check for closeness (perpendicular distance)
                        if is_horizontal_merge:
                            cand_perp_coord = (
                                candidate_props["coords"][1] + candidate_props["coords"][3]
                            ) / 2
                        else:
                            cand_perp_coord = (
                                candidate_props["coords"][0] + candidate_props["coords"][2]
                            ) / 2

                        perp_distance = abs(group_avg_perp_coord - cand_perp_coord)
                        if perp_distance > merge_distance_tolerance:
                            continue

                        # 3. Check for reasonable proximity along the primary axis
                        if is_horizontal_merge:
                            # For horizontal lines, check x-axis relationship
                            cand_x1, cand_x2 = (
                                candidate_props["coords"][0],
                                candidate_props["coords"][2],
                            )
                            # Check if there's overlap OR if the gap is reasonable
                            overlap = max(0, min(group_x2, cand_x2) - max(group_x1, cand_x1))
                            gap_to_group = min(abs(group_x1 - cand_x2), abs(group_x2 - cand_x1))

                            # Accept if there's overlap OR the gap is reasonable OR the candidate is contained within group span
                            if not (
                                overlap > 0
                                or gap_to_group <= merge_endpoint_tolerance
                                or (cand_x1 >= group_x1 and cand_x2 <= group_x2)
                            ):
                                continue
                        else:
                            # For vertical lines, check y-axis relationship
                            cand_y1, cand_y2 = (
                                candidate_props["coords"][1],
                                candidate_props["coords"][3],
                            )
                            overlap = max(0, min(group_y2, cand_y2) - max(group_y1, cand_y1))
                            gap_to_group = min(abs(group_y1 - cand_y2), abs(group_y2 - cand_y1))

                            if not (
                                overlap > 0
                                or gap_to_group <= merge_endpoint_tolerance
                                or (cand_y1 >= group_y1 and cand_y2 <= group_y2)
                            ):
                                continue

                        # If we reach here, lines should be merged
                        group.append(candidate_props)
                        merged_flags[j] = True
                        group_changed = True

                    if not group_changed:
                        break  # No more lines added in this pass, stop trying

                # Create final merged line from the group
                final_x1, final_y1 = min(p["coords"][0] for p in group), min(
                    p["coords"][1] for p in group
                )
                final_x2, final_y2 = max(p["coords"][2] for p in group), max(
                    p["coords"][3] for p in group
                )
                final_total_len = sum(p["length"] for p in group)
                if final_total_len == 0:
                    continue

                final_width = sum(p["width"] * p["length"] for p in group) / final_total_len
                final_nfa = sum(p["nfa_score"] * p["length"] for p in group) / final_total_len

                if is_horizontal_merge:
                    final_y = (
                        sum(((p["coords"][1] + p["coords"][3]) / 2) * p["length"] for p in group)
                        / final_total_len
                    )
                    merged_line_data = (
                        final_x1,
                        final_y,
                        final_x2,
                        final_y,
                        final_width,
                        final_nfa,
                    )
                else:
                    final_x = (
                        sum(((p["coords"][0] + p["coords"][2]) / 2) * p["length"] for p in group)
                        / final_total_len
                    )
                    merged_line_data = (
                        final_x,
                        final_y1,
                        final_x,
                        final_y2,
                        final_width,
                        final_nfa,
                    )
                merged_results.append(merged_line_data)
            return merged_results

        merged_h_lines = merge_lines_list(horizontal_lines, True)
        merged_v_lines = merge_lines_list(vertical_lines, False)
        all_merged = merged_h_lines + merged_v_lines

        final_lines_data = []
        for line_data_item in all_merged:
            x1, y1, x2, y2, width, nfa = line_data_item
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length > min_line_length:
                # Ensure x1 <= x2 for horizontal, y1 <= y2 for vertical
                if abs(y2 - y1) < abs(x2 - x1):  # Horizontal-ish
                    if x1 > x2:
                        x1_out, y1_out, x2_out, y2_out = x2, y2, x1, y1
                    else:
                        x1_out, y1_out, x2_out, y2_out = x1, y1, x2, y2
                else:  # Vertical-ish
                    if y1 > y2:
                        x1_out, y1_out, x2_out, y2_out = x2, y2, x1, y1
                    else:
                        x1_out, y1_out, x2_out, y2_out = x1, y1, x2, y2

                final_lines_data.append(
                    {
                        "x1": x1_out,
                        "y1": y1_out,
                        "x2": x2_out,
                        "y2": y2_out,
                        "width": width,
                        "nfa_score": nfa,
                        "length": length,
                    }
                )
        return final_lines_data

    def detect_blobs(
        self,
        k: Optional[int] = None,
        tolerance: float = 40.0,
        min_area_pts: float = 400.0,
        resolution: int = 150,
        replace: bool = True,
        source_label: str = "detected",
        overlap_threshold: float = 0.5,
    ) -> "ShapeDetectionMixin":
        """Detect colour blobs on a page/region and convert them to Region objects.

        Args:
            k: Desired number of colour clusters. ``None`` → automatically choose k
               (2‒15) using the elbow/knee method on inertia.
            tolerance: Maximum Delta-E CIE2000 distance at which two colour
               clusters are merged (40 ≈ perceptually "very similar"). Higher
               values merge more shades; set 0 to disable.
            min_area_pts: Ignore components whose bounding-box area in PDF points² is
               smaller than this value.
            resolution: DPI used for rasterising the page/region before detection.
            replace: If *True* purge existing ``region[type=blob]`` that share the
               same ``source_label`` before adding new ones.
            source_label: Stored in ``region.source`` so callers can distinguish
               between different detection passes.
            overlap_threshold: After blobs are built, discard any blob whose
                area overlaps vector elements (rects/words/lines/curves) by
                more than this fraction (0‒1).  Use this instead of pixel
                masking so large painted areas are not cut by text boxes.
        """
        import numpy as np
        from scipy.ndimage import find_objects
        from scipy.ndimage import label as nd_label

        # Acquire raster image & scale info
        cv_image, scale_factor, origin_offset_pdf, page_obj = self._get_image_for_detection(
            resolution
        )
        if cv_image is None or page_obj is None:
            return self  # nothing to do
        img_arr = cv_image.reshape(-1, 3).astype(np.float32) / 255.0  # normalised

        # No pre-masking of vector boxes; cluster entire image.
        h, w, _ = cv_image.shape
        unmasked_pixels = np.full(img_arr.shape[0], True, dtype=bool)
        img_arr_unmasked = img_arr  # cluster all pixels

        # ── choose k ────────────────────────────────────────────────────────────
        if k is None:
            inertias = []
            ks = list(range(2, 16))  # 2 … 15
            for _k in ks:
                km = MiniBatchKMeans(n_clusters=_k, random_state=0, batch_size=1024)
                km.fit(img_arr_unmasked[:: max(1, img_arr_unmasked.shape[0] // 50000)])  # subsample
                inertias.append(km.inertia_)
            # knee: biggest drop in inertia
            diffs = np.diff(inertias)
            knee_idx = int(np.argmin(diffs))  # most negative diff
            k = ks[knee_idx]
        # fit final model
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=1024)
        full_labels = kmeans.fit_predict(img_arr_unmasked)
        centroids = kmeans.cluster_centers_  # in 0-1 RGB
        h, w, _ = cv_image.shape
        full_label_flat = np.full(img_arr.shape[0], -1, dtype=int)
        full_label_flat[unmasked_pixels] = full_labels
        labels_img = full_label_flat.reshape(h, w)

        # ------------------------------------------------------------------
        # Merge clusters whose centroid colours are perceptually close
        # (Delta-E CIE2000 < tolerance).  We first identify the most frequent
        # cluster (likely background) and do NOT merge *into* it so that real
        # colourful blobs don't disappear when tolerance is large.
        # ------------------------------------------------------------------
        try:
            from colormath2.color_conversions import convert_color
            from colormath2.color_diff import delta_e_cie2000
            from colormath2.color_objects import LabColor, sRGBColor

            # Compute pixel counts per cluster to locate background
            counts = np.bincount(full_labels, minlength=k)
            bg_cluster = int(np.argmax(counts))  # largest cluster by pixel count

            lab_centroids = [convert_color(sRGBColor(*rgb), LabColor) for rgb in centroids]

            # Union-find parent array
            parent = list(range(k))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

            for i in range(k):
                for j in range(i + 1, k):
                    if bg_cluster in (i, j):
                        continue  # never merge INTO background
                    if delta_e_cie2000(lab_centroids[i], lab_centroids[j]) < tolerance:
                        union(i, j)

            # Remap every cluster id to its root parent
            root_map = [find(idx) for idx in range(k)]
            for old_id, new_id in enumerate(root_map):
                if old_id != new_id:
                    full_label_flat[full_label_flat == old_id] = new_id

        except ImportError:
            # colormath2 not available – skip merging
            pass

        labels_img = full_label_flat.reshape(h, w)

        # ── optional purge ──
        if replace and hasattr(page_obj, "_element_mgr"):
            old_blobs = [
                r
                for r in page_obj._element_mgr.regions
                if getattr(r, "region_type", None) == "blob"
                and getattr(r, "source", None) == source_label
            ]
            for r in old_blobs:
                page_obj._element_mgr.regions.remove(r)

        # ── iterate clusters ───────────────────────────────────────────────────
        unique_clusters = [cid for cid in np.unique(labels_img) if cid >= 0]
        for c_idx in unique_clusters:
            mask = labels_img == c_idx
            # clean tiny specks to avoid too many components
            mask_small = binary_opening(mask, structure=np.ones((3, 3)))
            # Bridge small gaps so contiguous paint isn't split by tiny holes
            if not mask_small.any():
                continue
            comp_labels, n_comps = nd_label(mask_small)
            if n_comps == 0:
                continue
            slices = find_objects(comp_labels)
            for comp_idx, sl in enumerate(slices):
                if sl is None:
                    continue
                y0, y1 = sl[0].start, sl[0].stop
                x0, x1 = sl[1].start, sl[1].stop
                # bbox area in pixels → in pts²
                area_pixels = (y1 - y0) * (x1 - x0)
                area_pts = area_pixels * (scale_factor**2)

                # Skip tiny regions
                if area_pts < min_area_pts:
                    continue

                # Skip page-background blocks (≥80 % page area)
                page_area_pts = page_obj.width * page_obj.height
                if area_pts / page_area_pts > 0.8:
                    continue

                # Compute mean colour of the component
                comp_pixels = cv_image[y0:y1, x0:x1]
                avg_rgb = comp_pixels.mean(axis=(0, 1)) / 255.0  # 0-1 range
                # Skip almost-white / almost-black near-grayscale areas (likely background or text)
                brightness = float(np.mean(avg_rgb))
                color_std = float(np.std(avg_rgb))
                if color_std < 0.05 and (brightness < 0.2 or brightness > 0.95):
                    continue

                # ----------------------------------------------------------------
                # Check overlap with characters BEFORE creating the Region.
                # If more than overlap_threshold of the blob area is covered by
                # any characters we discard it as likely text fill.
                # ----------------------------------------------------------------

                region_bbox_pdf = (
                    origin_offset_pdf[0] + x0 * scale_factor,
                    origin_offset_pdf[1] + y0 * scale_factor,
                    origin_offset_pdf[0] + x1 * scale_factor,
                    origin_offset_pdf[1] + y1 * scale_factor,
                )

                rx0, rtop, rx1, rbot = region_bbox_pdf
                region_area_pts = (rx1 - rx0) * (rbot - rtop)
                if region_area_pts == 0:
                    continue

                chars = getattr(page_obj, "chars", []) or []
                overlap_area = 0.0
                for ch in chars:
                    vx0, vtop, vx1, vbot = ch.x0, ch.top, ch.x1, ch.bottom
                    ix0 = max(rx0, vx0)
                    iy0 = max(rtop, vtop)
                    ix1 = min(rx1, vx1)
                    iy1 = min(rbot, vbot)
                    if ix1 > ix0 and iy1 > iy0:
                        overlap_area += (ix1 - ix0) * (iy1 - iy0)
                        if overlap_area / region_area_pts >= overlap_threshold:
                            break

                if overlap_area / region_area_pts >= overlap_threshold:
                    continue  # skip, mostly text

                # Map to PDF coords and create region after passing overlap test
                pdf_x0, pdf_top, pdf_x1, pdf_bottom = region_bbox_pdf

                from natural_pdf.elements.region import Region

                region = Region(page_obj, (pdf_x0, pdf_top, pdf_x1, pdf_bottom))
                region.region_type = "blob"
                region.normalized_type = "blob"
                region.source = source_label
                # Produce compact web colour using the 'colour' library if available
                try:
                    from colour import Color  # type: ignore

                    hex_str = str(Color(rgb=tuple(avg_rgb)))  # gives named/shortest rep
                except Exception:
                    hex_str = "#{:02x}{:02x}{:02x}".format(
                        int(avg_rgb[0] * 255), int(avg_rgb[1] * 255), int(avg_rgb[2] * 255)
                    )
                region.rgb = tuple(map(float, avg_rgb))  # numeric backup
                region.color = hex_str
                region.fill = hex_str

                # Store readable colour for inspection tables
                region.metadata["color_hex"] = hex_str

                page_obj._element_mgr.add_region(region)

        return self
