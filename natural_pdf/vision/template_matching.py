"""Pure NumPy template matching implementation"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class TemplateMatch:
    """Result of template matching"""

    bbox: Tuple[int, int, int, int]  # x0, y0, x1, y1
    score: float  # 0-1, higher is better


class TemplateMatcher:
    """Pure NumPy template matching implementation"""

    def __init__(self, method: str = "zncc"):
        """
        Args:
            method: Matching method
                - "zncc": Zero-mean Normalized Cross-Correlation (default, recommended)
                - "ncc": Normalized Cross-Correlation
                - "ssd": Sum of Squared Differences
        """
        self.method = method

    def match_template(
        self,
        image: np.ndarray,
        template: np.ndarray,
        step: int = 1,
        mask_threshold: Optional[float] = None,
    ) -> np.ndarray:
        """
        Compute similarity map between image and template.

        Args:
            image: Target image (grayscale, normalized 0-1)
            template: Template to search for (grayscale, normalized 0-1)
            step: Step size for sliding window (1 = pixel perfect, >1 = faster)
            mask_threshold: If provided, pixels >= this value in template are masked (ignored).
                           Useful for ignoring white backgrounds (e.g., 0.95 for near-white)

        Returns:
            2D array of match scores
        """
        if self.method == "zncc":
            return self._zncc(image, template, step, mask_threshold)
        elif self.method == "ncc":
            return self._ncc(image, template, step, mask_threshold)
        elif self.method == "ssd":
            return self._ssd(image, template, step, mask_threshold)
        else:
            # Default to zncc
            return self._zncc(image, template, step, mask_threshold)

    def _zncc(
        self,
        image: np.ndarray,
        template: np.ndarray,
        step: int = 1,
        mask_threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Zero-mean Normalized Cross-Correlation - most robust"""
        h, w = template.shape
        img_h, img_w = image.shape

        out_h = (img_h - h) // step + 1
        out_w = (img_w - w) // step + 1
        result = np.zeros((out_h, out_w))

        # Create mask if threshold provided
        if mask_threshold is not None:
            mask = template < mask_threshold  # True for pixels to keep
            if np.sum(mask) == 0:
                # All pixels are masked - return zeros
                return result
        else:
            mask = np.ones_like(template, dtype=bool)

        # Precompute template statistics on non-masked pixels
        masked_template = template[mask]
        if len(masked_template) == 0:
            return result

        template_mean = np.mean(masked_template)
        template_centered = np.zeros_like(template)
        template_centered[mask] = template[mask] - template_mean
        template_std = np.sqrt(np.sum(template_centered[mask] ** 2))

        # Handle uniform template case
        if template_std == 0:
            # Template has no variation - fall back to checking if means match
            for i in range(out_h):
                for j in range(out_w):
                    y = i * step
                    x = j * step
                    window = image[y : y + h, x : x + w]
                    window_masked = window[mask]
                    window_mean = np.mean(window_masked)
                    window_std = np.std(window_masked)

                    # Perfect match if window also has same mean and no variation
                    if abs(window_mean - template_mean) < 0.01 and window_std < 0.01:
                        result[i, j] = 1.0
            return result

        for i in range(out_h):
            for j in range(out_w):
                y = i * step
                x = j * step
                window = image[y : y + h, x : x + w]

                # Apply mask to window
                window_masked = window[mask]
                window_mean = np.mean(window_masked)
                window_centered = np.zeros_like(window)
                window_centered[mask] = window[mask] - window_mean
                window_std = np.sqrt(np.sum(window_centered[mask] ** 2))

                if window_std > 0:
                    correlation = np.sum(window_centered[mask] * template_centered[mask])
                    result[i, j] = correlation / (template_std * window_std)

        return np.clip(result, -1, 1)

    def _ncc(
        self,
        image: np.ndarray,
        template: np.ndarray,
        step: int = 1,
        mask_threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Normalized Cross-Correlation"""
        h, w = template.shape
        img_h, img_w = image.shape

        out_h = (img_h - h) // step + 1
        out_w = (img_w - w) // step + 1
        result = np.zeros((out_h, out_w))

        # Create mask if threshold provided
        if mask_threshold is not None:
            mask = template < mask_threshold  # True for pixels to keep
            if np.sum(mask) == 0:
                return result
        else:
            mask = np.ones_like(template, dtype=bool)

        template_norm = np.sqrt(np.sum(template[mask] ** 2))
        if template_norm == 0:
            return result

        for i in range(out_h):
            for j in range(out_w):
                y = i * step
                x = j * step
                window = image[y : y + h, x : x + w]

                window_norm = np.sqrt(np.sum(window[mask] ** 2))
                if window_norm > 0:
                    correlation = np.sum(window[mask] * template[mask])
                    result[i, j] = correlation / (template_norm * window_norm)

        return result

    def _ssd(
        self,
        image: np.ndarray,
        template: np.ndarray,
        step: int = 1,
        mask_threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Sum of Squared Differences - converted to similarity score"""
        h, w = template.shape
        img_h, img_w = image.shape

        out_h = (img_h - h) // step + 1
        out_w = (img_w - w) // step + 1
        result = np.zeros((out_h, out_w))

        # Create mask if threshold provided
        if mask_threshold is not None:
            mask = template < mask_threshold  # True for pixels to keep
            if np.sum(mask) == 0:
                return result
        else:
            mask = np.ones_like(template, dtype=bool)

        # Number of valid pixels for normalization
        n_valid = np.sum(mask)
        if n_valid == 0:
            return result

        for i in range(out_h):
            for j in range(out_w):
                y = i * step
                x = j * step
                window = image[y : y + h, x : x + w]

                # Only compute SSD on non-masked pixels
                diff = window - template
                ssd = np.sum((diff[mask]) ** 2) / n_valid
                result[i, j] = 1.0 / (1.0 + ssd)  # Convert to similarity

        return result
