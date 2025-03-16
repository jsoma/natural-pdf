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

page = pdf.pages[0]
day_sections = page.get_sections(start_elements='line[width>=2]')

for day in day_sections:
    date = day.find('text').text
    book_sections = day.get_sections(start_elements='text:contains("(Removed:")')
    for j, book in enumerate(book_sections):
        print("-----")
        if book.height < 30:
            print("Not a book, skipping")
            continue
        book.highlight(label=f"Day {date} section {j}")

        title = book.find_all('text[font_variant="AAAAAB"][size>=10]')
        title.highlight(label='Title')
        
        price = book.find('text:contains("Price")').below(height=15, width="element").expand(right=30)
        price.highlight(label='Price')

        acquired = book.find('text:contains("Acquired")').below(height=15, width="element").expand(right=30)
        acquired.highlight(label='Acquired')

        removed_by = book.find('text[size<10]:contains("Removed")').below(height=17, width="element").expand(right=60)
        removed_by.highlight(label='Removed By')

        data = {
            'Title': title.extract_text(),
            'Price': price.extract_text(),
            'Acquired': acquired.extract_text(),
            'Removed By': removed_by.extract_text()
        }
        print(data)

page.save("highlight.png", show_labels=True)