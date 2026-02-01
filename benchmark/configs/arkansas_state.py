"""
Arkansas State University Graduation Rates - Benchmark Configuration

PDF: pdfs/30.pdf
Structure: Header with summary stats + complex demographic tables with nested columns
"""

from dataclasses import dataclass
from typing import Optional

import natural_pdf as npdf
from benchmark.configs.prompts import CSV_ONLY, CSV_QUOTING, EXACT_COPY, JSON_ONLY


class ArkansasStateConfig:
    """Configuration for Arkansas State graduation rates benchmark."""

    name = "30"
    pdf_path = "pdfs/30.pdf"
    pdf_path_trap = "pdfs/30-trap.pdf"

    description = "University graduation rates with complex demographic tables"

    # Fields we're extracting
    header_fields = ["university_name", "graduation_rate", "four_class_average"]
    table_fields = ["demographic", "men_4class_n", "women_4class_n", "total_4class_n"]

    # =========================================================================
    # NATURAL PDF EXTRACTION (Ground Truth)
    # =========================================================================

    @staticmethod
    def extract_with_natural_pdf(pdf_path: str, page_num: int = 0) -> dict:
        """
        Extract graduation data using Natural PDF.

        This PDF has:
        - Title at top
        - Summary stats box
        - Complex nested demographic tables
        """
        pdf = npdf.PDF(pdf_path)
        page = pdf.pages[0]

        result = {
            "header": {},
            "all_students_table": [],
        }

        # Extract header info
        title = page.find("text[size>20]")
        if title:
            result["header"]["university_name"] = title.text.strip()

        # Find graduation rate (look for percentage near "Graduation Rate")
        grad_rate = page.find('text:contains("Graduation Rate")')
        if grad_rate:
            # The percentage might be to the right
            rate_region = grad_rate.right(width=200)
            if rate_region:
                # Find text that looks like a percentage
                pct_texts = rate_region.find_all('text:contains("%")')
                if pct_texts:
                    result["header"]["graduation_rate"] = pct_texts[0].text.strip()

        # Find four-class average
        four_class = page.find('text:contains("Four-Class Average")')
        if four_class:
            rate_region = four_class.right(width=200)
            if rate_region:
                pct_texts = rate_region.find_all('text:contains("%")')
                if pct_texts:
                    result["header"]["four_class_average"] = pct_texts[0].text.strip()

        # Extract the All Students table
        # Find the table region - it's below "a. All Students"
        all_students_header = page.find('text:contains("All Students")')
        if all_students_header:
            # Try extracting table from page
            table_data = page.extract_table(method="text")
            if table_data and len(table_data) > 0:
                # Process table rows - TableResult is directly iterable
                for row in table_data:
                    if row and len(row) >= 4:
                        # First column is demographic
                        demographic = str(row[0]).strip() if row[0] else ""
                        if demographic and demographic not in [
                            "",
                            "N",
                            "%",
                            "Men",
                            "Women",
                            "Total",
                        ]:
                            result["all_students_table"].append(
                                {
                                    "demographic": demographic,
                                    "raw_row": [str(c).strip() if c else "" for c in row],
                                }
                            )

        pdf.close()
        return result

    @staticmethod
    def extract_table_manual(pdf_path: str) -> list[dict]:
        """
        Manual table extraction for more precise control.
        Extracts specific cells by finding demographic labels and navigating.
        """
        pdf = npdf.PDF(pdf_path)
        page = pdf.pages[0]

        rows = []
        demographics = [
            "Am. Ind./AN",
            "Asian",
            "Black",
            "Hispanic",
            "Nat. Haw./PI",
            "US N-R",
            "Two or More",
            "Unknown",
            "White",
            "Total",
        ]

        for demo in demographics:
            demo_elem = page.find(f'text:contains("{demo}")')
            if demo_elem:
                # Get the entire row by looking at text on the same y-coordinate
                row_region = demo_elem.expand(right=500, left=0)
                row_texts = row_region.find_all("text")

                # Extract just the numbers
                numbers = []
                for t in row_texts:
                    txt = t.text.strip()
                    if txt and txt not in [demo, "-", "%"]:
                        numbers.append(txt)

                rows.append(
                    {
                        "demographic": demo,
                        "values": numbers,
                    }
                )

        pdf.close()
        return rows

    # =========================================================================
    # LLM PROMPTS
    # =========================================================================

    prompt = f"""Export the "a. All Students" graduation rates table as a CSV.

Columns: Demographic, Men_N, Men_Pct, Men_4Class_N, Men_4Class_Pct, Women_N, Women_Pct, Women_4Class_N, Women_4Class_Pct, Total_N, Total_Pct, Total_4Class_N, Total_4Class_Pct

Include all demographic rows (Am. Ind./AN through Total).

{EXACT_COPY}

{CSV_QUOTING}

{CSV_ONLY}"""

    json_prompt = f"""Extract the "a. All Students" table as JSON.

Return an array where each object has:
{{
  "demographic": "demographic name",
  "men_n": "value or -",
  "men_pct": "value or -",
  "men_4class_n": "number exactly as shown",
  "men_4class_pct": "number",
  "women_n": "value or -",
  "women_pct": "value or -",
  "women_4class_n": "number exactly as shown",
  "women_4class_pct": "number",
  "total_n": "value or -",
  "total_pct": "value or -",
  "total_4class_n": "number exactly as shown",
  "total_4class_pct": "number"
}}

{EXACT_COPY}

{JSON_ONLY}"""

    # =========================================================================
    # COMPARISON FIELDS
    # =========================================================================

    comparison_fields = [
        {
            "name": "white_men_4class_n",
            "description": "6/G confusion",
            "expected_original": "1623",
            "expected_trap": "1G23",
        },
        {
            "name": "white_total_4class_n",
            "description": "1/I confusion",
            "expected_original": "3681",
            "expected_trap": "368I",
        },
        {
            "name": "black_women_4class_n",
            "description": "1/I confusion",
            "expected_original": "311",
            "expected_trap": "3I1",
        },
        {
            "name": "total_women_4class_n",
            "description": "1/I confusion",
            "expected_original": "2621",
            "expected_trap": "262I",
        },
    ]
