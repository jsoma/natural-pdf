"""Visual similarity matching using perceptual hashing"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from tqdm.auto import tqdm


@dataclass
class MatchCandidate:
    """Candidate match during sliding window search"""

    bbox: Tuple[float, float, float, float]
    hash_value: int
    confidence: float


def compute_phash(image: Image.Image, hash_size: int = 8, blur_radius: float = 0) -> int:
    """
    Compute perceptual hash of an image using DCT.

    Args:
        image: PIL Image to hash
        hash_size: Size of the hash (8 = 64 bit hash)
        blur_radius: Optional blur to apply before hashing (makes more tolerant)

    Returns:
        Integer hash value
    """
    # Convert to grayscale
    if image.mode != "L":
        image = image.convert("L")

    # Optional blur to reduce sensitivity to minor variations
    if blur_radius > 0:
        from PIL import ImageFilter

        image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Resize to 32x32 (4x the hash size for DCT)
    highfreq_factor = 4
    img_size = hash_size * highfreq_factor
    image = image.resize((img_size, img_size), Image.Resampling.LANCZOS)

    # Convert to numpy array
    pixels = np.array(image, dtype=np.float32)

    # Apply DCT
    from scipy.fftpack import dct

    dct_coef = dct(dct(pixels, axis=0), axis=1)

    # Keep top-left 8x8 (low frequencies)
    dct_low = dct_coef[:hash_size, :hash_size]

    # Compute median excluding the DC component
    dct_low_no_dc = dct_low.flatten()[1:]  # Skip first element (DC)
    median = np.median(dct_low_no_dc)

    # Create binary hash
    diff = dct_low.flatten() > median

    # Convert to integer
    return sum(2**i for i, v in enumerate(diff) if v)


def hamming_distance(hash1: int, hash2: int, hash_size: int = 64) -> int:
    """Calculate Hamming distance between two hashes"""
    # XOR and count set bits
    xor = hash1 ^ hash2
    return bin(xor).count("1")


def hash_similarity(hash1: int, hash2: int, hash_size: int = 64) -> float:
    """Calculate similarity score between two hashes (0-1)"""
    distance = hamming_distance(hash1, hash2, hash_size)
    return 1.0 - (distance / hash_size)


class VisualMatcher:
    """Handles visual similarity matching using perceptual hashing"""

    def __init__(self, hash_size: int = 12):
        self.hash_size = hash_size
        self.hash_bits = hash_size * hash_size
        self._cache = {}

    def _get_search_scales(self, sizes: Optional[Union[float, Tuple, List]]) -> List[float]:
        """
        Convert various size input formats to a list of scales to search.

        Args:
            sizes: Can be:
                - None: just 1.0
                - float: ±percentage (e.g., 0.2 = 80%-120%)
                - tuple(min, max): range with smart logarithmic steps
                - tuple(min, max, step): explicit step size
                - list: exact sizes to use

        Returns:
            List of scale factors to search
        """
        if sizes is None:
            return [1.0]

        # List of exact sizes
        if isinstance(sizes, list):
            return sorted(sizes)

        # Single float: ±percentage
        if isinstance(sizes, (int, float)):
            if sizes <= 0:
                return [1.0]
            # Convert to min/max range
            min_scale = max(0.1, 1.0 - sizes)
            max_scale = 1.0 + sizes
            # Use tuple logic below
            sizes = (min_scale, max_scale)

        # Tuple handling
        if isinstance(sizes, tuple):
            if len(sizes) == 2:
                min_scale, max_scale = sizes
                if min_scale >= max_scale:
                    return [min_scale]

                # Smart defaults with logarithmic spacing
                # Calculate range ratio to determine number of steps
                ratio = max_scale / min_scale

                if ratio <= 1.5:  # Small range (e.g., 0.8-1.2)
                    num_steps = 5
                elif ratio <= 3.0:  # Medium range (e.g., 0.5-1.5)
                    num_steps = 7
                else:  # Large range (e.g., 0.5-2.0)
                    num_steps = 9

                # Generate logarithmically spaced scales
                log_min = np.log(min_scale)
                log_max = np.log(max_scale)
                log_scales = np.linspace(log_min, log_max, num_steps)
                scales = np.exp(log_scales).tolist()

                # Ensure 1.0 is included if in range
                if min_scale <= 1.0 <= max_scale and 1.0 not in scales:
                    # Find closest scale and replace with 1.0
                    closest_idx = np.argmin([abs(s - 1.0) for s in scales])
                    scales[closest_idx] = 1.0

                return scales

            elif len(sizes) == 3:
                # Explicit (min, max, step)
                min_scale, max_scale, step = sizes
                scales = []
                current = min_scale
                while current <= max_scale:
                    scales.append(current)
                    current += step
                # Ensure max is included if close
                if scales[-1] < max_scale and (max_scale - scales[-1]) < step * 0.1:
                    scales[-1] = max_scale
                return scales

        raise ValueError(f"Invalid sizes format: {sizes}")

    def find_matches_in_image(
        self,
        template: Image.Image,
        target: Image.Image,
        template_hash: Optional[int] = None,
        confidence_threshold: float = 0.6,
        step_factor: float = 0.1,
        sizes: Optional[Union[float, Tuple, List]] = None,
        show_progress: bool = True,
        progress_callback: Optional[Callable[[], None]] = None,
    ) -> List[MatchCandidate]:
        """
        Find all matches of template in target image using sliding window.

        Args:
            template: Template image to search for
            target: Target image to search in
            template_hash: Pre-computed hash of template (optional)
            confidence_threshold: Minimum similarity score (0-1)
            step_factor: Step size as fraction of template size
            sizes: Size variations to search. Can be:
                   - float: ±percentage (e.g., 0.2 = 80%-120%)
                   - tuple(min, max): search range with smart logarithmic steps
                   - tuple(min, max, step): explicit step size
                   - list: exact sizes to try (e.g., [0.8, 1.0, 1.2])
            show_progress: Show progress bar for sliding window search
            progress_callback: Optional callback function to call for each window checked

        Returns:
            List of MatchCandidate objects
        """
        matches = []

        # Compute template hash if not provided
        if template_hash is None:
            template_hash = compute_phash(template, self.hash_size)

        template_w, template_h = template.size
        target_w, target_h = target.size

        # Determine scales to search
        scales = self._get_search_scales(sizes)

        # Calculate total iterations for progress bar
        total_iterations = 0
        if show_progress and not progress_callback:
            for scale in scales:
                scaled_w = int(template_w * scale)
                scaled_h = int(template_h * scale)
                if scaled_w <= target_w and scaled_h <= target_h:
                    step_x = max(1, int(scaled_w * step_factor))
                    step_y = max(1, int(scaled_h * step_factor))
                    x_steps = len(range(0, target_w - scaled_w + 1, step_x))
                    y_steps = len(range(0, target_h - scaled_h + 1, step_y))
                    total_iterations += x_steps * y_steps

        # Setup progress bar if needed (only if no callback provided)
        progress_bar = None
        if show_progress and not progress_callback and total_iterations > 0:
            progress_bar = tqdm(total=total_iterations, desc="Scanning", unit="window", leave=False)

        # Search at each scale
        for scale in scales:
            # Scale template size
            scaled_w = int(template_w * scale)
            scaled_h = int(template_h * scale)

            if scaled_w > target_w or scaled_h > target_h:
                continue

            # Calculate step size
            step_x = max(1, int(scaled_w * step_factor))
            step_y = max(1, int(scaled_h * step_factor))

            # Sliding window search
            for y in range(0, target_h - scaled_h + 1, step_y):
                for x in range(0, target_w - scaled_w + 1, step_x):
                    # Extract window
                    window = target.crop((x, y, x + scaled_w, y + scaled_h))

                    # Resize to template size if scaled
                    if scale != 1.0:
                        window = window.resize((template_w, template_h), Image.Resampling.LANCZOS)

                    # Compute hash and similarity
                    window_hash = compute_phash(window, self.hash_size)
                    similarity = hash_similarity(template_hash, window_hash, self.hash_bits)

                    if similarity >= confidence_threshold:
                        # Convert back to target image coordinates
                        bbox = (x, y, x + scaled_w, y + scaled_h)
                        matches.append(MatchCandidate(bbox, window_hash, similarity))

                    # Update progress
                    if progress_bar:
                        progress_bar.update(1)
                    elif progress_callback:
                        progress_callback()

        # Close progress bar
        if progress_bar:
            progress_bar.close()

        # Remove overlapping matches (keep highest confidence)
        return self._filter_overlapping_matches(matches)

    def _filter_overlapping_matches(
        self, matches: List[MatchCandidate], overlap_threshold: float = 0.5
    ) -> List[MatchCandidate]:
        """Remove overlapping matches, keeping the highest confidence ones"""
        if not matches:
            return matches

        # Sort by confidence (highest first)
        sorted_matches = sorted(matches, key=lambda m: m.confidence, reverse=True)
        filtered = []

        for candidate in sorted_matches:
            # Check if this overlaps significantly with any already selected match
            keep = True
            for selected in filtered:
                overlap = self._calculate_overlap(candidate.bbox, selected.bbox)
                if overlap > overlap_threshold:
                    keep = False
                    break

            if keep:
                filtered.append(candidate)

        return filtered

    def _calculate_overlap(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate intersection over union (IoU) for two bboxes"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # Calculate intersection
        intersect_xmin = max(x1_min, x2_min)
        intersect_ymin = max(y1_min, y2_min)
        intersect_xmax = min(x1_max, x2_max)
        intersect_ymax = min(y1_max, y2_max)

        if intersect_xmax < intersect_xmin or intersect_ymax < intersect_ymin:
            return 0.0

        intersect_area = (intersect_xmax - intersect_xmin) * (intersect_ymax - intersect_ymin)

        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - intersect_area

        return intersect_area / union_area if union_area > 0 else 0.0
