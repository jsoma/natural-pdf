"""
Simple test of PaddlePaddle layout analysis using minimal parameters.
"""
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from natural_pdf import PDF

# Get the current directory of this script
script_dir = os.path.dirname(os.path.realpath(__file__))
# Get the parent directory (project root)
root_dir = os.path.dirname(script_dir)

# Get PDF path from command line or use default
if len(sys.argv) > 1:
    pdf_path = sys.argv[1]
else:
    # Default PDF path
    pdf_path = os.path.join(root_dir, "pdfs", "2019 Statistics.pdf")

# Get page number from command line or use default
page_num = int(sys.argv[2]) if len(sys.argv) > 2 else 0

print(f"Analyzing PDF: {pdf_path}")
print(f"Page: {page_num}")

# Load the PDF
pdf = PDF(pdf_path)
page = pdf.pages[page_num]

print("Running PaddlePaddle layout analysis...")

# Run paddle layout analysis using our minimal approach
regions = page.analyze_layout(
    model="paddle",
    confidence=0.2,  # Lower confidence threshold to detect more regions
    model_params={
        "show_log": True
    }
)

print(f"Found {len(regions)} regions")

# Group regions by type and source
region_groups = {}
for region in regions:
    region_type = region.region_type
    source = getattr(region, 'source', 'unknown')
    group_key = f"{region_type} ({source})"
    
    if group_key not in region_groups:
        region_groups[group_key] = []
    region_groups[group_key].append(region)

# Print regions by type and source
for group_key, group_regions in region_groups.items():
    print(f"{group_key}: {len(group_regions)} regions")

# Highlight regions by type and source with different colors
print("Highlighting regions...")
for group_key, group_regions in region_groups.items():
    for region in group_regions:
        region.highlight(label=f"{group_key}")

# Save highlighted image
output_path = os.path.join(root_dir, "output", "paddle_layout_simple.png")
print(f"Saving highlighted image to {output_path}")
page.to_image(path=output_path, show_labels=True)

print("Done!")