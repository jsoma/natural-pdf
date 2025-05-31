import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

if TYPE_CHECKING:
    from natural_pdf.core.page import Page
    from natural_pdf.core.pdf import PDF
    from natural_pdf.elements.collections import ElementCollection, PageCollection
    from natural_pdf.elements.line import LineElement
    # from natural_pdf.elements.rect import RectangleElement # Removed
    from natural_pdf.elements.region import Region

logger = logging.getLogger(__name__)

# Constants for default values of less commonly adjusted line detection parameters
LINE_DETECTION_PARAM_DEFAULTS = {
    "binarization_method": "otsu",
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

    def _get_image_for_detection(self, resolution: int) -> Tuple[Optional[np.ndarray], float, Tuple[float, float], Optional['Page']]:
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
        if hasattr(self, 'to_image') and hasattr(self, 'width') and hasattr(self, 'height'): # Page or Region
            if hasattr(self, 'x0') and hasattr(self, 'top') and hasattr(self, '_page'): # Region
                logger.debug(f"Shape detection on Region: {self}")
                page_obj = self._page
                pil_image = self.to_image(resolution=resolution, crop_only=True, include_highlights=False)
                if pil_image: # Ensure pil_image is not None before accessing attributes
                    origin_offset_pdf = (self.x0, self.top)
                    logger.debug(f"Region image rendered successfully: {pil_image.width}x{pil_image.height}, origin_offset: {origin_offset_pdf}")
            else: # Page
                logger.debug(f"Shape detection on Page: {self}")
                page_obj = self
                pil_image = self.to_image(resolution=resolution, include_highlights=False)
                logger.debug(f"Page image rendered successfully: {pil_image.width}x{pil_image.height}")
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
            scale_factor = ( (self.width / pil_image.width) + (self.height / pil_image.height) ) / 2.0
            logger.debug(f"Calculated scale_factor: {scale_factor:.4f} (PDF dimensions: {self.width:.1f}x{self.height:.1f}, Image: {pil_image.width}x{pil_image.height})")

        else:
            logger.warning("Could not determine page object or image dimensions for scaling.")
            scale_factor = 1.0 # Default to no scaling if info is missing

        return cv_image, scale_factor, origin_offset_pdf, page_obj


    def _process_image_for_lines(
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
        if cv_image is None:
            return []
        
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
        coords_arr, widths_arr, precs_arr, nfa_scores_arr = lsd.detect(gray_image)

        lines_raw = []
        if coords_arr is not None: # nfa_scores_arr can be None if no lines are found
            nfa_scores_list = nfa_scores_arr.flatten() if nfa_scores_arr is not None else [0.0] * len(coords_arr)
            widths_list = widths_arr.flatten() if widths_arr is not None else [1.0] * len(coords_arr)
            precs_list = precs_arr.flatten() if precs_arr is not None else [0.0] * len(coords_arr)

            for i in range(len(coords_arr)):
                lines_raw.append((
                    coords_arr[i][0],
                    widths_list[i] if i < len(widths_list) else 1.0,
                    precs_list[i] if i < len(precs_list) else 0.0,
                    nfa_scores_list[i] if i < len(nfa_scores_list) else 0.0
                ))
        
        def get_line_properties(line_data_item):
            l_coords, l_width, l_prec, l_nfa_score = line_data_item
            x1, y1, x2, y2 = l_coords
            angle_rad = np.arctan2(y2 - y1, x2 - x1)
            angle_deg = np.degrees(angle_rad)
            normalized_angle_deg = angle_deg % 180
            if normalized_angle_deg < 0:
                normalized_angle_deg += 180
            
            is_h = abs(normalized_angle_deg) <= off_angle or abs(normalized_angle_deg - 180) <= off_angle
            is_v = abs(normalized_angle_deg - 90) <= off_angle

            if is_h and x1 > x2: x1, x2, y1, y2 = x2, x1, y2, y1
            elif is_v and y1 > y2: y1, y2, x1, x2 = y2, y1, x2, x1
            
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            return {'coords': (x1, y1, x2, y2), 'width': l_width, 'prec': l_prec, 
                    'angle_deg': normalized_angle_deg, 'is_horizontal': is_h, 'is_vertical': is_v, 
                    'length': length, 'nfa_score': l_nfa_score}

        processed_lines = [get_line_properties(ld) for ld in lines_raw]
        
        filtered_lines = []
        for p in processed_lines:
            if p['length'] <= initial_min_line_length: continue
            if p['is_horizontal'] and p['nfa_score'] >= min_nfa_score_horizontal:
                filtered_lines.append(p)
            elif p['is_vertical'] and p['nfa_score'] >= min_nfa_score_vertical:
                filtered_lines.append(p)
        
        horizontal_lines = [p for p in filtered_lines if p['is_horizontal']]
        vertical_lines = [p for p in filtered_lines if p['is_vertical']]

        def merge_lines_list(lines_list, is_horizontal_merge):
            if not lines_list: return []
            key_sort = (lambda p: (p['coords'][1], p['coords'][0])) if is_horizontal_merge else (lambda p: (p['coords'][0], p['coords'][1]))
            lines_list.sort(key=key_sort)
            
            merged_results = []
            merged_flags = [False] * len(lines_list)

            for i, current_line_props in enumerate(lines_list):
                if merged_flags[i]: continue
                group = [current_line_props]; merged_flags[i] = True
                
                # Keep trying to expand the group until no more lines can be added
                # Use multiple passes to ensure transitive merging works properly
                for merge_pass in range(10):  # Up to 3 passes to catch complex merging scenarios
                    group_changed = False
                    
                    # Calculate current group boundaries
                    group_x1, group_y1 = min(p['coords'][0] for p in group), min(p['coords'][1] for p in group)
                    group_x2, group_y2 = max(p['coords'][2] for p in group), max(p['coords'][3] for p in group)
                    total_len_in_group = sum(p['length'] for p in group)
                    if total_len_in_group == 0: continue # Should not happen

                    # Calculate weighted averages for the group
                    group_avg_angle = sum(p['angle_deg'] * p['length'] for p in group) / total_len_in_group
                    
                    if is_horizontal_merge:
                        group_avg_perp_coord = sum(((p['coords'][1] + p['coords'][3]) / 2) * p['length'] for p in group) / total_len_in_group
                    else:
                        group_avg_perp_coord = sum(((p['coords'][0] + p['coords'][2]) / 2) * p['length'] for p in group) / total_len_in_group

                    # Check all unmerged lines for potential merging
                    for j, candidate_props in enumerate(lines_list):
                        if merged_flags[j]: continue
                        
                        # 1. Check for parallelism (angle similarity)
                        angle_diff = abs(group_avg_angle - candidate_props['angle_deg'])
                        # Handle wraparound for angles near 0/180
                        if angle_diff > 90:
                            angle_diff = 180 - angle_diff
                        if angle_diff > merge_angle_tolerance: continue
                        
                        # 2. Check for closeness (perpendicular distance)
                        if is_horizontal_merge:
                            cand_perp_coord = (candidate_props['coords'][1] + candidate_props['coords'][3]) / 2
                        else:
                            cand_perp_coord = (candidate_props['coords'][0] + candidate_props['coords'][2]) / 2
                        
                        perp_distance = abs(group_avg_perp_coord - cand_perp_coord)
                        if perp_distance > merge_distance_tolerance: continue
                        
                        # 3. Check for reasonable proximity along the primary axis
                        if is_horizontal_merge:
                            # For horizontal lines, check x-axis relationship
                            cand_x1, cand_x2 = candidate_props['coords'][0], candidate_props['coords'][2]
                            # Check if there's overlap OR if the gap is reasonable
                            overlap = max(0, min(group_x2, cand_x2) - max(group_x1, cand_x1))
                            gap_to_group = min(abs(group_x1 - cand_x2), abs(group_x2 - cand_x1))
                            
                            # Accept if there's overlap OR the gap is reasonable OR the candidate is contained within group span
                            if not (overlap > 0 or gap_to_group <= merge_endpoint_tolerance or (cand_x1 >= group_x1 and cand_x2 <= group_x2)):
                                continue
                        else:
                            # For vertical lines, check y-axis relationship  
                            cand_y1, cand_y2 = candidate_props['coords'][1], candidate_props['coords'][3]
                            overlap = max(0, min(group_y2, cand_y2) - max(group_y1, cand_y1))
                            gap_to_group = min(abs(group_y1 - cand_y2), abs(group_y2 - cand_y1))
                            
                            if not (overlap > 0 or gap_to_group <= merge_endpoint_tolerance or (cand_y1 >= group_y1 and cand_y2 <= group_y2)):
                                continue
                        
                        # If we reach here, lines should be merged
                        group.append(candidate_props)
                        merged_flags[j] = True
                        group_changed = True
                    
                    if not group_changed:
                        break  # No more lines added in this pass, stop trying
                
                # Create final merged line from the group
                final_x1, final_y1 = min(p['coords'][0] for p in group), min(p['coords'][1] for p in group)
                final_x2, final_y2 = max(p['coords'][2] for p in group), max(p['coords'][3] for p in group)
                final_total_len = sum(p['length'] for p in group)
                if final_total_len == 0: continue

                final_width = sum(p['width'] * p['length'] for p in group) / final_total_len
                final_nfa = sum(p['nfa_score'] * p['length'] for p in group) / final_total_len
                
                if is_horizontal_merge:
                    final_y = sum(((p['coords'][1] + p['coords'][3]) / 2) * p['length'] for p in group) / final_total_len
                    merged_line_data = (final_x1, final_y, final_x2, final_y, final_width, final_nfa)
                else:
                    final_x = sum(((p['coords'][0] + p['coords'][2]) / 2) * p['length'] for p in group) / final_total_len
                    merged_line_data = (final_x, final_y1, final_x, final_y2, final_width, final_nfa)
                merged_results.append(merged_line_data)
            return merged_results

        merged_h_lines = merge_lines_list(horizontal_lines, True)
        merged_v_lines = merge_lines_list(vertical_lines, False)
        all_merged = merged_h_lines + merged_v_lines
        
        final_lines_data = []
        for line_data_item in all_merged:
            x1, y1, x2, y2, width, nfa = line_data_item
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length > min_line_length:
                # Ensure x1 <= x2 for horizontal, y1 <= y2 for vertical
                if abs(y2 - y1) < abs(x2-x1): # Horizontal-ish
                    if x1 > x2: x1_out, y1_out, x2_out, y2_out = x2, y2, x1, y1
                    else: x1_out, y1_out, x2_out, y2_out = x1, y1, x2, y2
                else: # Vertical-ish
                    if y1 > y2: x1_out, y1_out, x2_out, y2_out = x2, y2, x1, y1
                    else: x1_out, y1_out, x2_out, y2_out = x1, y1, x2, y2

                final_lines_data.append({
                    'x1': x1_out, 'y1': y1_out, 'x2': x2_out, 'y2': y2_out, 
                    'width': width, 'nfa_score': nfa, 'length': length
                })
        return final_lines_data

    def _convert_line_to_element_data(
        self, line_data_img: Dict, scale_factor: float, origin_offset_pdf: Tuple[float, float], page_obj: 'Page', source_label: str
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

        x0 = origin_offset_pdf[0] + line_data_img['x1'] * effective_scale
        top = origin_offset_pdf[1] + line_data_img['y1'] * effective_scale
        x1 = origin_offset_pdf[0] + line_data_img['x2'] * effective_scale
        bottom = origin_offset_pdf[1] + line_data_img['y2'] * effective_scale # y2 is the second y-coord
        
        # For lines, width attribute in PDF points
        line_width_pdf = line_data_img['width'] * effective_scale

        # initial_doctop might not be loaded if page object is minimal
        initial_doctop = getattr(page_obj._page, 'initial_doctop', 0) if hasattr(page_obj, '_page') else 0

        return {
            "x0": x0, "top": top, "x1": x1, "bottom": bottom, # bottom here is y2_pdf
            "width": abs(x1 - x0), # This is bounding box width
            "height": abs(bottom - top), # This is bounding box height
            "linewidth": line_width_pdf, # Actual stroke width of the line
            "object_type": "line",
            "page_number": page_obj.page_number,
            "doctop": top + initial_doctop,
            "source": source_label,
            "stroking_color": (0,0,0), # Default, can be enhanced
            "non_stroking_color": (0,0,0), # Default
            # Add other raw data if useful
            "raw_line_thickness_px": line_data_img.get('line_thickness_px'), # Renamed from raw_nfa_score
            "raw_line_position_px": line_data_img.get('line_position_px'),   # Added for clarity
        }

    def _find_lines_on_image_data(
        self,
        cv_image: np.ndarray,
        pil_image_rgb: Image.Image, # For original dimensions
        horizontal: bool = True,
        vertical: bool = True,
        peak_threshold_h: float = 0.5,
        min_gap_h: int = 5,
        peak_threshold_v: float = 0.5,
        min_gap_v: int = 5,
        max_lines_h: Optional[int] = None,
        max_lines_v: Optional[int] = None,
        binarization_method: str = LINE_DETECTION_PARAM_DEFAULTS["binarization_method"],
        adaptive_thresh_block_size: int = LINE_DETECTION_PARAM_DEFAULTS["adaptive_thresh_block_size"],
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

        cv_gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        img_height, img_width = cv_gray.shape
        logger.debug(f"Line detection - Image dimensions: {img_width}x{img_height}")

        if binarization_method == "adaptive":
            binarized_image = cv2.adaptiveThreshold(cv_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv2.THRESH_BINARY_INV, adaptive_thresh_block_size, adaptive_thresh_C_val)
        elif binarization_method == "otsu":
            otsu_thresh_val, binarized_image = cv2.threshold(cv_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            logger.debug(f"Otsu's threshold applied. Value: {otsu_thresh_val}")
        else:
            logger.error(f"Invalid binarization_method: {binarization_method}. Supported: 'otsu', 'adaptive'. Defaulting to 'otsu'.")
            otsu_thresh_val, binarized_image = cv2.threshold(cv_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        binarized_norm = binarized_image / 255.0
        
        detected_lines_data = []
        profile_h_smoothed_for_viz: Optional[np.ndarray] = None
        profile_v_smoothed_for_viz: Optional[np.ndarray] = None

        def get_lines_from_profile(
            profile_data: np.ndarray,
            max_dimension_for_ratio: int,
            params_key_suffix: str,
            is_horizontal_detection: bool
        ) -> Tuple[List[Dict], np.ndarray]: # Ensure it always returns profile_smoothed
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
                profile_smoothed, height=current_peak_height_threshold, distance=find_peaks_distance,
                width=1, prominence=1, rel_height=peak_width_rel_height
            )
            
            final_peaks_indices = candidate_peaks_indices
            final_properties = candidate_properties

            if max_lines is not None:
                if len(candidate_peaks_indices) > 0 and 'prominences' in candidate_properties:
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
                            upper_bound = min(len(profile_smoothed), actual_profile_idx + min_gap + 1)
                            suppressed_profile_indices[lower_bound:upper_bound] = True
                            if num_selected >= max_lines: break
                    final_peaks_indices = candidate_peaks_indices[selected_peaks_original_indices]
                    final_properties = {key: val_array[selected_peaks_original_indices] for key, val_array in candidate_properties.items()}
                    logger.debug(f"Selected {len(final_peaks_indices)} {params_key_suffix.upper()}-lines for max_lines={max_lines}.")
                else:
                    final_peaks_indices = np.array([])
                    final_properties = {}
                    logger.debug(f"No {params_key_suffix.upper()}-peaks for max_lines selection.")
            elif not final_peaks_indices.size:
                 final_properties = {}
                 logger.debug(f"No {params_key_suffix.upper()}-lines found using threshold.")
            else:
                 logger.debug(f"Found {len(final_peaks_indices)} {params_key_suffix.upper()}-lines using threshold.")

            if final_peaks_indices.size > 0:
                sort_order = np.argsort(final_peaks_indices)
                final_peaks_indices = final_peaks_indices[sort_order]
                for key in final_properties: final_properties[key] = final_properties[key][sort_order]

            for i, peak_idx in enumerate(final_peaks_indices):
                center_coord = int(peak_idx)
                profile_thickness = final_properties.get("widths", [])[i] if "widths" in final_properties and i < len(final_properties["widths"]) else 1.0
                profile_thickness = max(1, int(round(profile_thickness)))
                
                current_img_width = pil_image_rgb.width # Use actual passed image dimensions
                current_img_height = pil_image_rgb.height

                if is_horizontal_detection: 
                    lines_info.append({
                        'x1': 0, 'y1': center_coord, 
                        'x2': current_img_width -1, 'y2': center_coord,
                        'width': profile_thickness, 
                        'length': current_img_width,
                        'line_thickness_px': profile_thickness,
                        'line_position_px': center_coord
                    })
                else: 
                    lines_info.append({
                        'x1': center_coord, 'y1': 0,
                        'x2': center_coord, 'y2': current_img_height -1,
                        'width': profile_thickness, 
                        'length': current_img_height,
                        'line_thickness_px': profile_thickness,
                        'line_position_px': center_coord
                    })
            return lines_info, profile_smoothed

        if horizontal:
            processed_image_h = binarized_norm.copy()
            if morph_op_h != "none":
                kernel_h_struct = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_h)
                op = cv2.MORPH_OPEN if morph_op_h == "open" else cv2.MORPH_CLOSE
                processed_image_h = cv2.morphologyEx(processed_image_h, op, kernel_h_struct)
            profile_h_raw = np.sum(processed_image_h, axis=1)
            horizontal_lines, smoothed_h = get_lines_from_profile(profile_h_raw, pil_image_rgb.width, 'h', True)
            profile_h_smoothed_for_viz = smoothed_h
            detected_lines_data.extend(horizontal_lines)
            logger.info(f"Detected {len(horizontal_lines)} horizontal lines.")

        if vertical:
            processed_image_v = binarized_norm.copy()
            if morph_op_v != "none":
                kernel_v_struct = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_v)
                op = cv2.MORPH_OPEN if morph_op_v == "open" else cv2.MORPH_CLOSE
                processed_image_v = cv2.morphologyEx(processed_image_v, op, kernel_v_struct)
            profile_v_raw = np.sum(processed_image_v, axis=0)
            vertical_lines, smoothed_v = get_lines_from_profile(profile_v_raw, pil_image_rgb.height, 'v', False)
            profile_v_smoothed_for_viz = smoothed_v
            detected_lines_data.extend(vertical_lines)
            logger.info(f"Detected {len(vertical_lines)} vertical lines.")
            
        return detected_lines_data, profile_h_smoothed_for_viz, profile_v_smoothed_for_viz

    def detect_lines(
        self,
        resolution: int = 192,
        source_label: str = "detected",
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
        adaptive_thresh_block_size: int = LINE_DETECTION_PARAM_DEFAULTS["adaptive_thresh_block_size"],
        adaptive_thresh_C_val: int = LINE_DETECTION_PARAM_DEFAULTS["adaptive_thresh_C_val"],
        morph_op_h: str = LINE_DETECTION_PARAM_DEFAULTS["morph_op_h"],
        morph_kernel_h: Tuple[int, int] = LINE_DETECTION_PARAM_DEFAULTS["morph_kernel_h"],
        morph_op_v: str = LINE_DETECTION_PARAM_DEFAULTS["morph_op_v"],
        morph_kernel_v: Tuple[int, int] = LINE_DETECTION_PARAM_DEFAULTS["morph_kernel_v"],
        smoothing_sigma_h: float = LINE_DETECTION_PARAM_DEFAULTS["smoothing_sigma_h"],
        smoothing_sigma_v: float = LINE_DETECTION_PARAM_DEFAULTS["smoothing_sigma_v"],
        peak_width_rel_height: float = LINE_DETECTION_PARAM_DEFAULTS["peak_width_rel_height"],
    ) -> "ShapeDetectionMixin": # Return type changed back to self
        """
        Detects lines on the Page or Region, or on all pages within a Collection
        using projection profiling and peak detection.
        Adds detected lines as LineElement objects to the ElementManager.

        Args:
            resolution: DPI for image rendering before detection.
            source_label: Label assigned to the 'source' attribute of created LineElements.
            horizontal: If True, detect horizontal lines.
            vertical: If True, detect vertical lines.
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

        Returns:
            Self for method chaining.
        """
        if not horizontal and not vertical:
            logger.info("Line detection skipped as both horizontal and vertical are False.")
            return self
            
        collection_params = {
            "resolution": resolution, "source_label": source_label,
            "horizontal": horizontal, "vertical": vertical,
            "peak_threshold_h": peak_threshold_h, "min_gap_h": min_gap_h,
            "peak_threshold_v": peak_threshold_v, "min_gap_v": min_gap_v,
            "max_lines_h": max_lines_h, "max_lines_v": max_lines_v,
            "replace": replace,
            "binarization_method": binarization_method,
            "adaptive_thresh_block_size": adaptive_thresh_block_size,
            "adaptive_thresh_C_val": adaptive_thresh_C_val,
            "morph_op_h": morph_op_h, "morph_kernel_h": morph_kernel_h,
            "morph_op_v": morph_op_v, "morph_kernel_v": morph_kernel_v,
            "smoothing_sigma_h": smoothing_sigma_h, "smoothing_sigma_v": smoothing_sigma_v,
            "peak_width_rel_height": peak_width_rel_height,
        }

        if hasattr(self, 'pdfs'):
            for pdf_doc in self.pdfs:
                for page_obj in pdf_doc.pages:
                    page_obj.detect_lines(**collection_params)
            return self
        elif hasattr(self, 'pages') and not hasattr(self, '_page'):
            for page_obj in self.pages:
                page_obj.detect_lines(**collection_params)
            return self

        cv_image, scale_factor, origin_offset_pdf, page_object_ctx = self._get_image_for_detection(resolution)
        if cv_image is None or page_object_ctx is None:
            logger.warning(f"Skipping line detection for {self} due to image error.")
            return self
        
        pil_image_for_dims = None
        if hasattr(self, 'to_image') and hasattr(self, 'width') and hasattr(self, 'height'):
            if hasattr(self, 'x0') and hasattr(self, 'top') and hasattr(self, '_page'):
                pil_image_for_dims = self.to_image(resolution=resolution, crop_only=True, include_highlights=False)
            else:
                pil_image_for_dims = self.to_image(resolution=resolution, include_highlights=False)
        if pil_image_for_dims is None:
            logger.warning(f"Could not re-render PIL image for dimensions for {self}.")
            pil_image_for_dims = Image.fromarray(cv_image) # Ensure it's not None

        if pil_image_for_dims.mode != "RGB":
            pil_image_for_dims = pil_image_for_dims.convert("RGB")

        if replace:
            from natural_pdf.elements.line import LineElement
            element_manager = page_object_ctx._element_mgr
            if hasattr(element_manager, '_elements') and 'lines' in element_manager._elements:
                original_count = len(element_manager._elements['lines'])
                element_manager._elements['lines'] = [
                    line for line in element_manager._elements['lines']
                    if getattr(line, 'source', None) != source_label
                ]
                removed_count = original_count - len(element_manager._elements['lines'])
                if removed_count > 0:
                    logger.info(f"Removed {removed_count} existing lines with source '{source_label}' from {page_object_ctx}")

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
            morph_op_h=morph_op_h, morph_kernel_h=morph_kernel_h,
            morph_op_v=morph_op_v, morph_kernel_v=morph_kernel_v,
            smoothing_sigma_h=smoothing_sigma_h, smoothing_sigma_v=smoothing_sigma_v,
            peak_width_rel_height=peak_width_rel_height,
        )

        from natural_pdf.elements.line import LineElement
        element_manager = page_object_ctx._element_mgr

        for line_data_item_img in lines_data_img:
            element_constructor_data = self._convert_line_to_element_data(
                line_data_item_img, scale_factor, origin_offset_pdf, page_object_ctx, source_label
            )
            try:
                line_element = LineElement(element_constructor_data, page_object_ctx)
                element_manager.add_element(line_element, element_type="lines")
            except Exception as e:
                logger.error(f"Failed to create or add LineElement: {e}. Data: {element_constructor_data}", exc_info=True)
        
        logger.info(f"Detected and added {len(lines_data_img)} lines to {page_object_ctx} with source '{source_label}'.")
        return self

    def detect_lines_preview(
        self,
        resolution: int = 72, # Preview typically uses lower resolution
        horizontal: bool = True,
        vertical: bool = True,
        peak_threshold_h: float = 0.5,
        min_gap_h: int = 5,
        peak_threshold_v: float = 0.5,
        min_gap_v: int = 5,
        max_lines_h: Optional[int] = None,
        max_lines_v: Optional[int] = None,
        binarization_method: str = LINE_DETECTION_PARAM_DEFAULTS["binarization_method"],
        adaptive_thresh_block_size: int = LINE_DETECTION_PARAM_DEFAULTS["adaptive_thresh_block_size"],
        adaptive_thresh_C_val: int = LINE_DETECTION_PARAM_DEFAULTS["adaptive_thresh_C_val"],
        morph_op_h: str = LINE_DETECTION_PARAM_DEFAULTS["morph_op_h"],
        morph_kernel_h: Tuple[int, int] = LINE_DETECTION_PARAM_DEFAULTS["morph_kernel_h"],
        morph_op_v: str = LINE_DETECTION_PARAM_DEFAULTS["morph_op_v"],
        morph_kernel_v: Tuple[int, int] = LINE_DETECTION_PARAM_DEFAULTS["morph_kernel_v"],
        smoothing_sigma_h: float = LINE_DETECTION_PARAM_DEFAULTS["smoothing_sigma_h"],
        smoothing_sigma_v: float = LINE_DETECTION_PARAM_DEFAULTS["smoothing_sigma_v"],
        peak_width_rel_height: float = LINE_DETECTION_PARAM_DEFAULTS["peak_width_rel_height"],
    ) -> Optional[Image.Image]:
        """
        Previews detected lines on a Page or Region without adding them to the PDF elements.
        Generates and returns a debug visualization image.
        This method is intended for Page or Region objects.
        See `detect_lines` for parameter descriptions. The main difference is a lower default `resolution`.
        """
        if hasattr(self, 'pdfs') or (hasattr(self, 'pages') and not hasattr(self, '_page')):
            logger.warning("preview_detected_lines is intended for single Page/Region objects. For collections, process pages individually.")
            return None

        if not horizontal and not vertical: # Check this early
            logger.info("Line preview skipped as both horizontal and vertical are False.")
            return None

        cv_image, _, _, page_object_ctx = self._get_image_for_detection(resolution) # scale_factor and origin_offset not needed for preview
        if cv_image is None or page_object_ctx is None: # page_object_ctx for logging context mostly
            logger.warning(f"Skipping line preview for {self} due to image error.")
            return None

        pil_image_for_dims = None
        if hasattr(self, 'to_image') and hasattr(self, 'width') and hasattr(self, 'height'):
            if hasattr(self, 'x0') and hasattr(self, 'top') and hasattr(self, '_page'):
                pil_image_for_dims = self.to_image(resolution=resolution, crop_only=True, include_highlights=False)
            else:
                pil_image_for_dims = self.to_image(resolution=resolution, include_highlights=False)
        
        if pil_image_for_dims is None:
            logger.warning(f"Could not render PIL image for preview for {self}. Using cv_image to create one.")
            pil_image_for_dims = Image.fromarray(cv_image)
        
        if pil_image_for_dims.mode != "RGB":
            pil_image_for_dims = pil_image_for_dims.convert("RGB")

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
            morph_op_h=morph_op_h, morph_kernel_h=morph_kernel_h,
            morph_op_v=morph_op_v, morph_kernel_v=morph_kernel_v,
            smoothing_sigma_h=smoothing_sigma_h, smoothing_sigma_v=smoothing_sigma_v,
            peak_width_rel_height=peak_width_rel_height,
        )
        
        if not lines_data_img: # Check if any lines were detected before visualization
             logger.info(f"No lines detected for preview on {page_object_ctx or self}")
             # Optionally return the base image if no lines, or None
             return pil_image_for_dims.convert("RGBA") # Return base image so something is shown


        # --- Visualization Logic (copied from previous debug block) ---
        final_viz_image: Optional[Image.Image] = None
        viz_image_base = pil_image_for_dims.convert("RGBA")
        draw = ImageDraw.Draw(viz_image_base)
        img_width, img_height = viz_image_base.size

        viz_params = {
            "draw_line_thickness_viz": 1,
            "debug_histogram_size": 100,
            "line_color_h": (255, 0, 0, 200), "line_color_v": (0, 0, 255, 200),
            "histogram_bar_color_h": (200, 0, 0, 200), "histogram_bar_color_v": (0, 0, 200, 200),
            "histogram_bg_color": (240, 240, 240, 255), "padding_between_viz": 10,
            "peak_threshold_h": peak_threshold_h, 
            "peak_threshold_v": peak_threshold_v, 
            "max_lines_h": max_lines_h, 
            "max_lines_v": max_lines_v, 
        }

        for line_info in lines_data_img:
            is_h_line = abs(line_info['y1'] - line_info['y2']) < abs(line_info['x1'] - line_info['x2']) 
            line_color = viz_params["line_color_h"] if is_h_line else viz_params["line_color_v"]
            draw.line([
                (line_info['x1'], line_info['y1']),
                (line_info['x2'], line_info['y2'])
            ], fill=line_color, width=viz_params["draw_line_thickness_viz"])

        hist_size = viz_params["debug_histogram_size"]
        hist_h_img = Image.new("RGBA", (hist_size, img_height), viz_params["histogram_bg_color"])
        hist_h_draw = ImageDraw.Draw(hist_h_img)
        
        if profile_h_smoothed is not None and profile_h_smoothed.size > 0:
            actual_max_h_profile = profile_h_smoothed.max()
            max_h_profile_val_for_scaling = actual_max_h_profile if actual_max_h_profile > 0 else img_width
            display_threshold_val_h = peak_threshold_h * img_width
            for y_coord, val in enumerate(profile_h_smoothed):
                bar_len = 0; thresh_bar_len = 0
                if max_h_profile_val_for_scaling > 0:
                    bar_len = int((val / max_h_profile_val_for_scaling) * hist_size)
                    if display_threshold_val_h >= 0:
                        thresh_bar_len = int((display_threshold_val_h / max_h_profile_val_for_scaling) * hist_size)
                bar_len = min(max(0, bar_len), hist_size)
                if bar_len > 0: hist_h_draw.line([(0, y_coord), (bar_len -1 , y_coord)], fill=viz_params["histogram_bar_color_h"], width=1)
                if viz_params["max_lines_h"] is None and display_threshold_val_h >=0 and \
                   thresh_bar_len > 0 and thresh_bar_len < hist_size:
                    hist_h_draw.line([(thresh_bar_len, y_coord), (thresh_bar_len, y_coord+1 if y_coord+1 < img_height else y_coord)], fill=(0,255,0,100), width=1)
        
        hist_v_img = Image.new("RGBA", (img_width, hist_size), viz_params["histogram_bg_color"])
        hist_v_draw = ImageDraw.Draw(hist_v_img)
        if profile_v_smoothed is not None and profile_v_smoothed.size > 0:
            actual_max_v_profile = profile_v_smoothed.max()
            max_v_profile_val_for_scaling = actual_max_v_profile if actual_max_v_profile > 0 else img_height
            display_threshold_val_v = peak_threshold_v * img_height
            for x_coord, val in enumerate(profile_v_smoothed):
                bar_height = 0; thresh_bar_h = 0
                if max_v_profile_val_for_scaling > 0:
                    bar_height = int((val / max_v_profile_val_for_scaling) * hist_size)
                    if display_threshold_val_v >=0:
                        thresh_bar_h = int((display_threshold_val_v / max_v_profile_val_for_scaling) * hist_size)
                bar_height = min(max(0, bar_height), hist_size)
                if bar_height > 0: hist_v_draw.line([(x_coord, hist_size -1 ), (x_coord, hist_size - bar_height)], fill=viz_params["histogram_bar_color_v"], width=1)
                if viz_params["max_lines_v"] is None and display_threshold_val_v >=0 and \
                   thresh_bar_h > 0 and thresh_bar_h < hist_size:
                    hist_v_draw.line([(x_coord, hist_size - thresh_bar_h), (x_coord+1 if x_coord+1 < img_width else x_coord, hist_size - thresh_bar_h)], fill=(0,255,0,100), width=1)

        padding = viz_params["padding_between_viz"]
        total_width = img_width + padding + hist_size
        total_height = img_height + padding + hist_size
        final_viz_image = Image.new("RGBA", (total_width, total_height), (255, 255, 255, 255))
        final_viz_image.paste(viz_image_base, (0, 0))
        final_viz_image.paste(hist_h_img, (img_width + padding, 0))
        final_viz_image.paste(hist_v_img, (0, img_height + padding))
        logger.info(f"Generated line preview visualization for {page_object_ctx or self}")
        return final_viz_image 

    def detect_table_structure_from_lines(
        self,
        source_label: str = "detected",
        ignore_outer_regions: bool = True,
        cell_padding: float = 0.5,  # Small padding inside cells, default to 0.5px
    ) -> "ShapeDetectionMixin":
        """
        Create table structure (rows, columns, cells) from previously detected lines.
        
        This method analyzes horizontal and vertical lines to create a grid structure,
        then generates Region objects for:
        - An overall table region that encompasses the entire table structure
        - Individual row regions spanning the width of the table
        - Individual column regions spanning the height of the table 
        - Individual cell regions at each row/column intersection
        
        Args:
            source_label: Filter lines by this source label (from detect_lines)
            ignore_outer_regions: If True, don't create regions outside the defined by lines grid.
                                  If False, include regions from page/object edges to the first/last lines.
            cell_padding: Internal padding for cell regions
            
        Returns:
            Self for method chaining
        """
        # Handle collections
        if hasattr(self, 'pdfs'):
            for pdf_doc in self.pdfs:
                for page_obj in pdf_doc.pages:
                    page_obj.detect_table_structure_from_lines(
                        source_label=source_label,
                        ignore_outer_regions=ignore_outer_regions,
                        cell_padding=cell_padding,
                    )
            return self
        elif hasattr(self, 'pages') and not hasattr(self, '_page'): # PageCollection
            for page_obj in self.pages:
                page_obj.detect_table_structure_from_lines(
                    source_label=source_label,
                    ignore_outer_regions=ignore_outer_regions,
                    cell_padding=cell_padding,
                )
            return self

        # Determine context (Page or Region) for coordinates and element management
        page_object_for_elements = None
        origin_x, origin_y = 0.0, 0.0
        context_width, context_height = 0.0, 0.0

        if hasattr(self, '_element_mgr') and hasattr(self, 'width') and hasattr(self, 'height'): # Likely a Page
            page_object_for_elements = self
            context_width = self.width
            context_height = self.height
            logger.debug(f"Operating on Page context: {self}")
        elif hasattr(self, '_page') and hasattr(self, 'x0') and hasattr(self, 'width'): # Likely a Region
            page_object_for_elements = self._page
            origin_x = self.x0
            origin_y = self.top
            context_width = self.width   # Region's own width/height for its boundary calculations
            context_height = self.height
            logger.debug(f"Operating on Region context: {self}, origin: ({origin_x}, {origin_y})")
        else:
            logger.warning(f"Could not determine valid page/region context for {self}. Aborting table structure detection.")
            return self
        
        element_manager = page_object_for_elements._element_mgr

        # Get lines with the specified source
        all_lines = element_manager.lines # Access lines from the correct element manager
        filtered_lines = [line for line in all_lines if getattr(line, 'source', None) == source_label]
        
        if not filtered_lines:
            logger.info(f"No lines found with source '{source_label}' for table structure detection on {self}.")
            return self

        # Separate horizontal and vertical lines
        # For regions, line coordinates are already absolute to the page.
        horizontal_lines = [line for line in filtered_lines if line.is_horizontal]
        vertical_lines = [line for line in filtered_lines if line.is_vertical]
        
        logger.info(f"Found {len(horizontal_lines)} horizontal and {len(vertical_lines)} vertical lines for {self} with source '{source_label}'.")

        # Define boundaries based on line positions (mid-points for sorting, actual edges for boundaries)
        # These coordinates are relative to the page_object_for_elements (which is always a Page)
        
        # Horizontal line Y-coordinates (use average y, effectively the line's y-position)
        h_line_ys = sorted(list(set([(line.top + line.bottom) / 2 for line in horizontal_lines])))
        
        # Vertical line X-coordinates (use average x, effectively the line's x-position)
        v_line_xs = sorted(list(set([(line.x0 + line.x1) / 2 for line in vertical_lines])))

        row_boundaries = []
        if horizontal_lines:
            if not ignore_outer_regions:
                row_boundaries.append(origin_y) # Region's top or Page's 0
            row_boundaries.extend(h_line_ys)
            if not ignore_outer_regions:
                row_boundaries.append(origin_y + context_height) # Region's bottom or Page's height
        elif not ignore_outer_regions : # No horizontal lines, but we might want full height cells
             row_boundaries.extend([origin_y, origin_y + context_height])
        row_boundaries = sorted(list(set(row_boundaries)))


        col_boundaries = []
        if vertical_lines:
            if not ignore_outer_regions:
                col_boundaries.append(origin_x) # Region's left or Page's 0
            col_boundaries.extend(v_line_xs)
            if not ignore_outer_regions:
                col_boundaries.append(origin_x + context_width) # Region's right or Page's width
        elif not ignore_outer_regions: # No vertical lines, but we might want full width cells
            col_boundaries.extend([origin_x, origin_x + context_width])
        col_boundaries = sorted(list(set(col_boundaries)))
        
        logger.debug(f"Row boundaries for {self}: {row_boundaries}")
        logger.debug(f"Col boundaries for {self}: {col_boundaries}")

        # Create overall table region that wraps the entire structure
        tables_created = 0
        if len(row_boundaries) >= 2 and len(col_boundaries) >= 2:
            table_left = col_boundaries[0]
            table_top = row_boundaries[0]
            table_right = col_boundaries[-1]
            table_bottom = row_boundaries[-1]
            
            if table_right > table_left and table_bottom > table_top:
                try:
                    table_region = page_object_for_elements.create_region(
                        table_left, table_top, table_right, table_bottom
                    )
                    table_region.source = source_label
                    table_region.region_type = "table"
                    table_region.normalized_type = "table"  # Add normalized_type for selector compatibility
                    table_region.metadata.update({
                        "source_lines_label": source_label,
                        "num_rows": len(row_boundaries) - 1,
                        "num_cols": len(col_boundaries) - 1,
                        "boundaries": {
                            "rows": row_boundaries,
                            "cols": col_boundaries
                        }
                    })
                    element_manager.add_element(table_region, element_type="regions")
                    tables_created += 1
                    logger.debug(f"Created table region: L{table_left:.1f} T{table_top:.1f} R{table_right:.1f} B{table_bottom:.1f}")
                except Exception as e:
                    logger.error(f"Failed to create or add table Region: {e}. Table abs coords: L{table_left} T{table_top} R{table_right} B{table_bottom}", exc_info=True)

        # Create cell regions
        cells_created = 0
        rows_created = 0
        cols_created = 0
        
        # Create Row Regions
        if len(row_boundaries) >= 2:
            # Determine horizontal extent for rows
            row_extent_x0 = origin_x
            row_extent_x1 = origin_x + context_width
            if col_boundaries: # If columns are defined, rows should span only across them
                if len(col_boundaries) >=2:
                    row_extent_x0 = col_boundaries[0]
                    row_extent_x1 = col_boundaries[-1]
                # If only one col_boundary (e.g. from ignore_outer_regions=False and one line), use context width
                # This case should be rare if lines are properly detected to form a grid.

            for i in range(len(row_boundaries) - 1):
                top_abs = row_boundaries[i]
                bottom_abs = row_boundaries[i+1]
                
                # Use calculated row_extent_x0 and row_extent_x1
                if bottom_abs > top_abs and row_extent_x1 > row_extent_x0: # Ensure valid region
                    try:
                        row_region = page_object_for_elements.create_region(
                            row_extent_x0, top_abs, row_extent_x1, bottom_abs
                        )
                        row_region.source = source_label
                        row_region.region_type = "table_row"
                        row_region.normalized_type = "table_row"  # Add normalized_type for selector compatibility
                        row_region.metadata.update({
                            "row_index": i,
                            "source_lines_label": source_label
                        })
                        element_manager.add_element(row_region, element_type="regions")
                        rows_created += 1
                    except Exception as e:
                        logger.error(f"Failed to create or add table_row Region: {e}. Row abs coords: L{row_extent_x0} T{top_abs} R{row_extent_x1} B{bottom_abs}", exc_info=True)

        # Create Column Regions
        if len(col_boundaries) >= 2:
            # Determine vertical extent for columns
            col_extent_y0 = origin_y
            col_extent_y1 = origin_y + context_height
            if row_boundaries: # If rows are defined, columns should span only across them
                if len(row_boundaries) >=2:
                    col_extent_y0 = row_boundaries[0]
                    col_extent_y1 = row_boundaries[-1]
                # If only one row_boundary, use context height - similar logic to rows

            for j in range(len(col_boundaries) - 1):
                left_abs = col_boundaries[j]
                right_abs = col_boundaries[j+1]
                
                # Use calculated col_extent_y0 and col_extent_y1
                if right_abs > left_abs and col_extent_y1 > col_extent_y0: # Ensure valid region
                    try:
                        col_region = page_object_for_elements.create_region(
                            left_abs, col_extent_y0, right_abs, col_extent_y1
                        )
                        col_region.source = source_label
                        col_region.region_type = "table_column"
                        col_region.normalized_type = "table_column"  # Add normalized_type for selector compatibility
                        col_region.metadata.update({
                            "col_index": j,
                            "source_lines_label": source_label
                        })
                        element_manager.add_element(col_region, element_type="regions")
                        cols_created += 1
                    except Exception as e:
                        logger.error(f"Failed to create or add table_column Region: {e}. Col abs coords: L{left_abs} T{col_extent_y0} R{right_abs} B{col_extent_y1}", exc_info=True)

        # Create Cell Regions (existing logic)
        if len(row_boundaries) < 2 or len(col_boundaries) < 2:
            logger.info(f"Not enough boundaries to form cells for {self}. Rows: {len(row_boundaries)}, Cols: {len(col_boundaries)}")
            # return self # Return will be at the end
        else:
            for i in range(len(row_boundaries) - 1):
                top_abs = row_boundaries[i]
                bottom_abs = row_boundaries[i+1]
                
                for j in range(len(col_boundaries) - 1):
                    left_abs = col_boundaries[j]
                    right_abs = col_boundaries[j+1]
                    
                    cell_left_abs = left_abs + cell_padding
                    cell_top_abs = top_abs + cell_padding
                    cell_right_abs = right_abs - cell_padding
                    cell_bottom_abs = bottom_abs - cell_padding
                    
                    cell_width = cell_right_abs - cell_left_abs
                    cell_height = cell_bottom_abs - cell_top_abs
                    
                    if cell_width <= 0 or cell_height <= 0:
                        logger.debug(f"Skipping cell (zero or negative dimension after padding): L{left_abs:.1f} T{top_abs:.1f} R{right_abs:.1f} B{bottom_abs:.1f} -> W{cell_width:.1f} H{cell_height:.1f}")
                        continue

                    try:
                        cell_region = page_object_for_elements.create_region(
                            cell_left_abs, cell_top_abs, cell_right_abs, cell_bottom_abs
                        )
                        cell_region.source = source_label
                        cell_region.region_type = "table_cell"
                        cell_region.normalized_type = "table_cell"  # Add normalized_type for selector compatibility
                        cell_region.metadata.update({
                            "row_index": i,
                            "col_index": j,
                            "source_lines_label": source_label,
                            "original_boundaries_abs": {
                                "left": left_abs, "top": top_abs, 
                                "right": right_abs, "bottom": bottom_abs
                            }
                        })
                        element_manager.add_element(cell_region, element_type="regions")
                        cells_created += 1
                    except Exception as e:
                        logger.error(f"Failed to create or add cell Region: {e}. Cell abs coords: L{cell_left_abs} T{cell_top_abs} R{cell_right_abs} B{cell_bottom_abs}", exc_info=True)

        logger.info(f"Created {tables_created} table, {rows_created} rows, {cols_created} columns, and {cells_created} table cells from detected lines (source: '{source_label}') for {self}.")
        return self

# Example usage would be:
# page.detect_lines(source_label="my_table_lines")
# page.detect_table_structure_from_lines(source_label="my_table_lines", cell_padding=0.5)
# 
# Now both selector styles work equivalently:
# table = page.find('table[source*="table_from"]')  # Direct type selector
# table = page.find('region[type="table"][source*="table_from"]')  # Region attribute selector
# cells = page.find_all('table-cell[source*="table_cells_from"]')  # Direct type selector
# cells = page.find_all('region[type="table-cell"][source*="table_cells_from"]')  # Region attribute selector 