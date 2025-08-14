"""Pure NumPy template matching implementation"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


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

    def match_template(self, image: np.ndarray, template: np.ndarray, step: int = 1) -> np.ndarray:
        """
        Compute similarity map between image and template.

        Args:
            image: Target image (grayscale)
            template: Template to search for (grayscale)
            step: Step size for sliding window (1 = pixel perfect, >1 = faster)

        Returns:
            2D array of match scores
        """
        if self.method == "zncc":
            return self._zncc(image, template, step)
        elif self.method == "ncc":
            return self._ncc(image, template, step)
        elif self.method == "ssd":
            return self._ssd(image, template, step)
        else:
            # Default to zncc
            return self._zncc(image, template, step)

    def _zncc(self, image: np.ndarray, template: np.ndarray, step: int = 1) -> np.ndarray:
        """Zero-mean Normalized Cross-Correlation - most robust"""
        h, w = template.shape
        img_h, img_w = image.shape

        out_h = (img_h - h) // step + 1
        out_w = (img_w - w) // step + 1
        result = np.zeros((out_h, out_w))

        # Precompute template statistics
        template_mean = np.mean(template)
        template_centered = template - template_mean
        template_std = np.sqrt(np.sum(template_centered**2))

        # Handle uniform template case
        if template_std == 0:
            # Template has no variation - fall back to checking if means match
            for i in range(out_h):
                for j in range(out_w):
                    y = i * step
                    x = j * step
                    window = image[y : y + h, x : x + w]
                    window_mean = np.mean(window)
                    window_std = np.std(window)

                    # Perfect match if window also has same mean and no variation
                    if abs(window_mean - template_mean) < 0.01 and window_std < 0.01:
                        result[i, j] = 1.0
            return result

        for i in range(out_h):
            for j in range(out_w):
                y = i * step
                x = j * step
                window = image[y : y + h, x : x + w]

                window_mean = np.mean(window)
                window_centered = window - window_mean
                window_std = np.sqrt(np.sum(window_centered**2))

                if window_std > 0:
                    correlation = np.sum(window_centered * template_centered)
                    result[i, j] = correlation / (template_std * window_std)

        return np.clip(result, -1, 1)

    def _ncc(self, image: np.ndarray, template: np.ndarray, step: int = 1) -> np.ndarray:
        """Normalized Cross-Correlation"""
        h, w = template.shape
        img_h, img_w = image.shape

        out_h = (img_h - h) // step + 1
        out_w = (img_w - w) // step + 1
        result = np.zeros((out_h, out_w))

        template_norm = np.sqrt(np.sum(template**2))
        if template_norm == 0:
            return result

        for i in range(out_h):
            for j in range(out_w):
                y = i * step
                x = j * step
                window = image[y : y + h, x : x + w]

                window_norm = np.sqrt(np.sum(window**2))
                if window_norm > 0:
                    correlation = np.sum(window * template)
                    result[i, j] = correlation / (template_norm * window_norm)

        return result

    def _ssd(self, image: np.ndarray, template: np.ndarray, step: int = 1) -> np.ndarray:
        """Sum of Squared Differences - converted to similarity score"""
        h, w = template.shape
        img_h, img_w = image.shape

        out_h = (img_h - h) // step + 1
        out_w = (img_w - w) // step + 1
        result = np.zeros((out_h, out_w))

        for i in range(out_h):
            for j in range(out_w):
                y = i * step
                x = j * step
                window = image[y : y + h, x : x + w]

                ssd = np.sum((window - template) ** 2) / (h * w)
                result[i, j] = 1.0 / (1.0 + ssd)  # Convert to similarity

        return result
