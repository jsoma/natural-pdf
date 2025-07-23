"""Vision module for visual similarity and pattern matching"""

from .mixin import VisualSearchMixin
from .results import Match, MatchResults
from .similarity import VisualMatcher, compute_phash

__all__ = ["VisualMatcher", "compute_phash", "Match", "MatchResults", "VisualSearchMixin"]
