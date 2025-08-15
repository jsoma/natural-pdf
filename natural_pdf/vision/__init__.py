"""Vision module for visual similarity and pattern matching"""

from .mixin import VisualSearchMixin
from .similarity import VisualMatcher, compute_phash

__all__ = ["VisualMatcher", "compute_phash", "VisualSearchMixin"]
