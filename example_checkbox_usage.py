"""Example usage of checkbox detection in Natural PDF."""

import natural_pdf as npdf

# Load a PDF
pdf = npdf.PDF("pdfs/01-practice.pdf")
page = pdf[0]

# Basic checkbox detection
print("=== Basic Checkbox Detection ===")
checkboxes = page.detect_checkboxes()
print(f"Found {len(checkboxes)} checkboxes")

# Show what was found
for i, cb in enumerate(checkboxes[:3]):
    print(f"\nCheckbox {i}:")
    print(f"  State: {'Checked' if cb.is_checked else 'Unchecked'}")
    print(f"  Confidence: {cb.confidence:.2f}")
    print(f"  Position: {cb.bbox}")

# Using selectors to filter checkboxes
print("\n=== Using Selectors ===")
checked = page.find_all("checkbox:checked")
unchecked = page.find_all("checkbox:unchecked")
print(f"Checked boxes: {len(checked)}")
print(f"Unchecked boxes: {len(unchecked)}")

# Limit detection when you know expected count
print("\n=== Limited Detection ===")
# If you know there should be 10 checkboxes on a form
limited_checkboxes = page.detect_checkboxes(limit=10)
print(f"Found top {len(limited_checkboxes)} checkboxes by confidence")

# Multi-page detection
print("\n=== Multi-page Detection ===")
all_checkboxes = pdf.detect_checkboxes(show_progress=False)
print(f"Total checkboxes in PDF: {len(all_checkboxes)}")

# Visualize checkboxes
print("\n=== Visualization ===")
print("Showing detected checkboxes...")
checkboxes.show()

# Advanced: Using custom options
print("\n=== Advanced Options ===")
from natural_pdf.analyzers.checkbox import CheckboxOptions

# Higher confidence threshold
options = CheckboxOptions(confidence=0.5)
high_conf_checkboxes = page.detect_checkboxes(options=options)
print(f"High confidence checkboxes: {len(high_conf_checkboxes)}")

# GPU acceleration if available
gpu_checkboxes = page.detect_checkboxes(device="cuda")
print(f"GPU-detected checkboxes: {len(gpu_checkboxes)}")
