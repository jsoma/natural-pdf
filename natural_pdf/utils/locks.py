"""
Shared locks for thread synchronization across the natural-pdf library.
"""

import threading

# Global lock for PDF rendering operations to prevent PDFium concurrency issues
pdf_render_lock = threading.RLock()
