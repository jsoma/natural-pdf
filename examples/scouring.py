"""
Example demonstrating section extraction with the get_sections method.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from natural_pdf import PDF

pdf = PDF("./pdfs/Nigeria 2021_MICS_SFR_English.pdf")

# Exclude "Page | 123" footer from all queries
pdf.add_exclusion(lambda page: page.find_all('text').lowest().below(include_element=True))

# There's a bold header for 'EQ.4.1W' on a few of these pages
header = pdf.pages[460:470].find('text:contains("EQ.4.1W"):bold')

header.highlight(label='table header')

(
    header
        .below()
        .find('text:contains("Total"):bold')
        .below(
            until='text:contains("MICS")',
            include_element=True,
            include_until=False
        )
        .highlight(label='table area')
)

header.page.to_image(path="output.png", show_labels=True)