"""
Example demonstrating section extraction with the get_sections method.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from natural_pdf import PDF

pdf = PDF("./pdfs/Atlanta_Public_Schools_GA_sample.pdf")

day_sections = pdf.pages.get_sections(start_elements='line[width>=2]')

rows = []
for day in day_sections:
    date = day.find('text').text
    book_sections = day.get_sections(start_elements='text:contains("(Removed:")')
    
    for book in book_sections:
        if book.height < 30:
            print("Not a book, skipping")
            continue

        # Bold big text is the title
        title = book.find_all('text[font_variant="AAAAAB"][size>=10]')
        price = book.find('text:contains("Price")').below(height=15, width="element").expand(right=30)
        acquired = book.find('text:contains("Acquired")').below(height=15, width="element").expand(right=30)
        removed_by = book.find('text[size<10]:contains("Removed")').below(height=17, width="element").expand(right=60)

        # Highlight them
        book.highlight(label=title.extract_text())
        title.highlight(label='title')
        price.highlight(label='price')
        acquired.highlight(label='acquired')
        removed_by.highlight(label='removed')

        # Save them
        data = {
            'Title': title.extract_text(),
            'Price': price.extract_text(),
            'Acquired': acquired.extract_text(),
            'Removed By': removed_by.extract_text()
        }
        rows.append(data)

pdf.pages[0].save("highlight-1.png", show_labels=True)
pdf.pages[1].save("highlight-2.png", show_labels=True)
pdf.pages[2].save("highlight-3.png", show_labels=True)