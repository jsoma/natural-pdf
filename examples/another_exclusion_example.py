"""
Example demonstrating how to use exclusion zones in Natural PDF.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from natural_pdf import PDF

pdf = PDF('pdfs/Atlanta_Public_Schools_GA_sample.pdf')
pdf.add_exclusion(lambda page: page.find('line').above())
pdf.add_exclusion(lambda page: page.find_all('line')[-1].below())
page = pdf.pages[2]
page.find_all('text').highlight()
page.save('test.png', labels=True)

