"""
Analyzers for natural-pdf.
"""
from natural_pdf.analyzers.document_layout import (
    LayoutDetector,
    YOLODocLayoutDetector,
    TableTransformerDetector,
    convert_to_regions
)