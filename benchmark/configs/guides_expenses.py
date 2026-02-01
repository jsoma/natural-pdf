"""
Guides Expenses Sample PDF - Expense Report Benchmark Configuration

PDF: pdfs/guides-expenses-sample.pdf
Structure: Header row + expense line items table with various columns
"""

import natural_pdf as npdf
from benchmark.configs.prompts import EXACT_COPY, JSON_ONLY


class GuidesExpensesConfig:
    """Configuration for guides-expenses-sample expense report benchmark."""

    name = "guides-expenses"
    pdf_path = "pdfs/guides-expenses-sample.pdf"
    pdf_path_trap = None  # No trap version

    description = (
        "Expense report with multi-column table including dates, vendors, items, and costs"
    )

    # =========================================================================
    # NATURAL PDF EXTRACTION (Ground Truth)
    # =========================================================================

    @staticmethod
    def extract_with_natural_pdf(pdf_path: str, page_num: int = 0) -> dict:
        """
        Extract expense table using Natural PDF with header-based vertical guides.
        """
        pdf = npdf.PDF(pdf_path)
        page = pdf.pages[0]

        result = {}

        # Find headers by going right from "Date submitted" to "Total"
        headers = (
            page.find("text:contains(Date submitted)")
            .right(until="text:contains(Total)", include_source=True)
            .find_all("text")
        )

        # Extract table using headers as vertical guides
        table_df = page.extract_table(verticals=headers).to_df()

        # Remove empty rows (last row often empty)
        table_df = table_df.dropna(how="all")

        result["expense_table"] = table_df.to_dict("records")
        result["row_count"] = len(table_df)
        result["columns"] = list(table_df.columns)

        pdf.close()
        return result

    # =========================================================================
    # LLM PROMPT
    # =========================================================================

    prompt = f"""Extract the expense table from this document as JSON.

The table has these columns:
- Date submitted
- Date completed
- User ID
- PO
- Vendor
- Categories
- Item #
- Item Description
- Quantity
- Price
- Shipping
- Tax

Return as:
{{
  "expenses": [
    {{
      "date_submitted": "...",
      "date_completed": "...",
      "user_id": "...",
      "po": "...",
      "vendor": "...",
      "categories": "...",
      "item_number": "...",
      "item_description": "...",
      "quantity": "...",
      "price": "...",
      "shipping": "...",
      "tax": "..."
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
            "name": "row0_item_number",
            "description": "First item ASIN",
            "expected_original": "B07RW6Z692",
            "expected_trap": None,
        },
        {
            "name": "row0_price",
            "description": "First item price",
            "expected_original": "$117.99",
            "expected_trap": None,
        },
        {
            "name": "row0_categories",
            "description": "First item category",
            "expected_original": "Educational Materials/Items",
            "expected_trap": None,
        },
        {
            "name": "row6_vendor",
            "description": "Row 6 vendor",
            "expected_original": "Amazon Inc.",
            "expected_trap": None,
        },
        {
            "name": "row6_categories",
            "description": "Row 6 category",
            "expected_original": "Curricula",
            "expected_trap": None,
        },
        {
            "name": "row6_price",
            "description": "Row 6 price",
            "expected_original": "$38.35",
            "expected_trap": None,
        },
        {
            "name": "row12_item_number",
            "description": "Row 12 ISBN",
            "expected_original": "448478994",
            "expected_trap": None,
        },
        {
            "name": "row12_categories",
            "description": "Row 12 category",
            "expected_original": "Reading Books",
            "expected_trap": None,
        },
        {
            "name": "row22_item_description",
            "description": "Last data row item description",
            "expected_original": "History -- Modern Marvels Panama Canal",
            "expected_trap": None,
        },
        {
            "name": "row22_price",
            "description": "Last data row price",
            "expected_original": "$13.50",
            "expected_trap": None,
        },
        {
            "name": "total_rows",
            "description": "Total number of expense rows",
            "expected_original": "23",
            "expected_trap": None,
        },
    ]
