"""
Highlighting utilities for natural-pdf.

This module primarily re-exports core highlighting utilities from the visualization module.
The main highlighting logic is now centralized in `natural_pdf.core.highlighting_service.HighlightingService`.
"""
# Re-export necessary functions from visualization
from .visualization import get_next_highlight_color, create_legend, merge_images_with_legend, reset_highlight_colors

# --- The Highlight class and HighlightManager class previously defined here have been removed ---
# --- The functionality is now handled by natural_pdf.core.highlighting_service.HighlightingService ---
# --- and its internal HighlightRenderer class. ---