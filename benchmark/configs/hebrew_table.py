"""
Hebrew Table PDF - Economic Indicators Benchmark Configuration

PDF: pdfs/hebrew-table.pdf
Structure: Hebrew right-to-left economic data table with year columns (2009-2014)
"""

import natural_pdf as npdf
from benchmark.configs.prompts import EXACT_COPY, JSON_ONLY


class HebrewTableConfig:
    """Configuration for hebrew-table economic indicators benchmark."""

    name = "hebrew-table"
    pdf_path = "pdfs/hebrew-table.pdf"
    pdf_path_trap = None  # No trap version

    description = "Hebrew economic indicators table with year columns and numeric data"

    # =========================================================================
    # NATURAL PDF EXTRACTION (Ground Truth)
    # =========================================================================

    @staticmethod
    def extract_with_natural_pdf(pdf_path: str, page_num: int = 0) -> dict:
        """
        Extract economic data table using Natural PDF with guide-based extraction.
        """
        pdf = npdf.PDF(pdf_path)
        page = pdf.pages[0]

        result = {}

        # Find the vertical divider and use it to locate row boundaries
        divider = page.find_all("rect:vertical[height>100]")[1]
        rows = divider.right(height="element").find_all("text").dissolve()

        # Find headers (years) above the second horizontal line
        headers = page.find_all("rect:horizontal")[1].above().find_all("text")

        # Extract table using guides with outer boundaries
        table_df = page.extract_table(verticals=headers, horizontals=rows, outer=True).to_df()

        # Remove fully empty rows
        table_df = table_df.dropna(how="all")

        result["data_table"] = table_df.to_dict("records")
        result["row_count"] = len(table_df)
        result["columns"] = list(table_df.columns)

        pdf.close()
        return result

    # =========================================================================
    # LLM PROMPT
    # =========================================================================

    prompt = f"""Extract the economic data table from this Hebrew document as JSON.

The table has year columns: 2014, 2013, 2012, 2011, 2010, 2009
The rightmost column (2009) contains both a numeric value and a Hebrew row label.

Return as:
{{
  "rows": [
    {{
      "2014": "...",
      "2013": "...",
      "2012": "...",
      "2011": "...",
      "2010": "...",
      "2009": "...",
      "label": "..." (Hebrew row label from the 2009 column)
    }},
    ...
  ]
}}

{EXACT_COPY}

{JSON_ONLY}"""

    # =========================================================================
    # COMPARISON FIELDS (values to check for accuracy)
    # =========================================================================

    comparison_fields = [
        {
            "name": "row1_2014",
            "description": "Row 1 value for 2014",
            "expected_original": "2.6",
            "expected_trap": None,
        },
        {
            "name": "row1_2009",
            "description": "Row 1 value for 2009 (includes label)",
            "expected_original": "1.9",
            "expected_trap": None,
        },
        {
            "name": "row5_2014",
            "description": "Row 5 (construction) value for 2014",
            "expected_original": "-1.7",
            "expected_trap": None,
        },
        {
            "name": "row5_2011",
            "description": "Row 5 (construction) value for 2011",
            "expected_original": "11.6",
            "expected_trap": None,
        },
        {
            "name": "row15_2014",
            "description": "Row 15 value for 2014",
            "expected_original": "1.8",
            "expected_trap": None,
        },
        {
            "name": "row15_2010",
            "description": "Row 15 value for 2010",
            "expected_original": "2.7",
            "expected_trap": None,
        },
        {
            "name": "row4_2010",
            "description": "Row 4 (manufacturing) value for 2010",
            "expected_original": "12.0",
            "expected_trap": None,
        },
        {
            "name": "row4_2009",
            "description": "Row 4 (manufacturing) value for 2009",
            "expected_original": "-4.3",
            "expected_trap": None,
        },
        {
            "name": "total_rows",
            "description": "Total number of data rows",
            "expected_original": "35",
            "expected_trap": None,
        },
    ]
