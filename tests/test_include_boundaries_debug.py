"""
Debug script to find suitable text for testing include_boundaries.
"""

from pathlib import Path

import natural_pdf as npdf

# Try both PDFs
pdfs = [
    Path(__file__).parent.parent / "pdfs" / "24480polcompleted.pdf",
    Path(__file__).parent.parent / "pdfs" / "types-of-type.pdf",
]

for pdf_path in pdfs:
    if pdf_path.exists():
        print(f"\nAnalyzing: {pdf_path.name}")
        print("=" * 60)

        pdf = npdf.PDF(str(pdf_path))
        print(f"Number of pages: {len(pdf.pages)}")

        # Get text from first page
        if len(pdf.pages) > 0:
            page = pdf.pages[0]
            text = page.extract_text()

            # Find potential section headers (lines with specific characteristics)
            lines = text.split("\n")
            potential_headers = []

            for line in lines[:20]:  # Check first 20 lines
                line = line.strip()
                if len(line) > 3 and len(line) < 50:  # Reasonable header length
                    potential_headers.append(line)

            print("\nPotential section headers:")
            for i, header in enumerate(potential_headers[:10]):
                print(f"  {i+1}. '{header}'")

            # Try to find repeated text that might indicate sections
            print("\nSearching for repeated text patterns...")

            # Common section indicators
            patterns = [
                "section",
                "chapter",
                "part",
                "title",
                "heading",
                "occurrence",
                "violation",
                "complaint",
                "date",
                "name",
            ]

            for pattern in patterns:
                matches = pdf.find_all(f"text:contains({pattern})", case_sensitive=False)
                if len(matches) > 1:
                    print(f"  '{pattern}': {len(matches)} occurrences")

                    # Show first few matches
                    for i, match in enumerate(matches[:3]):
                        print(f"    - Page {match.page.number}: '{match.extract_text().strip()}'")

        print("\n")
