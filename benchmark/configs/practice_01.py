"""
01-practice PDF - Health Inspection Report Benchmark Configuration

PDF: pdfs/01-practice.pdf (original), pdfs/01-practice-benchmark.pdf (with traps)
Structure: Form header + summary text + violations table with checkboxes
"""

import natural_pdf as npdf
from benchmark.configs.prompts import EXACT_COPY, JSON_ONLY


class Practice01Config:
    """Configuration for 01-practice health inspection benchmark."""

    name = "01-practice"
    pdf_path = "pdfs/01-practice.pdf"
    pdf_path_trap = "pdfs/01-practice-trap.pdf"

    description = "Health inspection report with form fields, summary text, and violations table"

    # =========================================================================
    # NATURAL PDF EXTRACTION (Ground Truth)
    # =========================================================================

    @staticmethod
    def extract_with_natural_pdf(pdf_path: str, page_num: int = 0) -> dict:
        """
        Extract all fields using Natural PDF.
        """
        pdf = npdf.PDF(pdf_path)
        page = pdf.pages[0]

        result = {}

        # Form ID (red text at top)
        result["form_id"] = page.find("text[color~=red]").extract_text()

        # Site name - text to the right of "Site" label
        result["site"] = (
            page.find("text:contains('Site')").right(until="text").expand(right=-1).extract_text()
        )

        # Date - text to the right of "Date" label
        result["date"] = page.find("text:contains('Date')").right().extract_text()

        # Location (grey text)
        result["location"] = page.find("text[color~=grey]").extract_text()

        # Summary text - below "Summary" until the line
        result["summary"] = (
            page.find("text:contains(Summary)")
            .below(until="line", include_source=True)
            .extract_text()
            .replace("Summary:", "")
            .strip()
        )

        # Violations table
        table_df = (
            page.find("text:contains(Violations)[size=max()]").below().extract_table().to_df()
        )

        # Extract Repeat? column separately (checkboxes)
        table_df["Repeat?"] = (
            page.find("text:contains(Repeat?)")
            .below(width="element")
            .find_all("rect")
            .apply(lambda cell: "yes" if cell.find("line") else "no")
        )

        result["violations_table"] = table_df.to_dict("records")

        pdf.close()
        return result

    # =========================================================================
    # LLM PROMPT
    # =========================================================================

    prompt = f"""Extract all data from this health inspection document as JSON.

{{
  "form_fields": {{
    "form_id": "...",
    "site": "...",
    "date": "...",
    "location": "...",
    "summary": "..."
  }},
  "violations": [
    {{"statute": "...", "description": "...", "level": "...", "repeat": "yes/no"}},
    ...
  ]
}}

For "repeat": use "yes" if checkbox is checked, "no" if not.

{EXACT_COPY}

{JSON_ONLY}"""

    # =========================================================================
    # COMPARISON FIELDS (traps to check)
    # =========================================================================

    comparison_fields = [
        {
            "name": "site",
            "description": "rh/nh swap",
            "expected_original": "Durham's Meatpacking",
            "expected_trap": "Dunham's Meatpacking",
        },
        {
            "name": "summary_fertilizer",
            "description": "US/UK spelling",
            "expected_original": "fertilizer",
            "expected_trap": "fertiliser",
        },
        {
            "name": "summary_visitor",
            "description": "or/er swap",
            "expected_original": "visitor",
            "expected_trap": "visiter",
        },
        {
            "name": "summary_hundred",
            "description": "ed/rd swap",
            "expected_original": "hundred",
            "expected_trap": "hunderd",
        },
        {
            "name": "summary_lard",
            "description": "L/B swap",
            "expected_original": "Leaf Lard",
            "expected_trap": "Beef Lard",
        },
        {
            "name": "statute_row0",
            "description": "7/T confusion",
            "expected_original": "4.12.7",
            "expected_trap": "4.12.T",
        },
        {
            "name": "statute_row2",
            "description": "6/8 + 9/O confusion",
            "expected_original": "6.3.9",
            "expected_trap": "8.3.O",
        },
        {
            "name": "statute_row6",
            "description": "1/I confusion",
            "expected_original": "10.2.7",
            "expected_trap": "I0.2.7",
        },
        {
            "name": "description_fire",
            "description": "í/i diacritic",
            "expected_original": "Fire",
            "expected_trap": "Fíre",
        },
    ]
