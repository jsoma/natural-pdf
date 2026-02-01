"""
Oklahoma License Listing - Benchmark Configuration

PDF: pdfs/m27.pdf
Structure: Dense tabular data with license numbers, company names, addresses
"""

import pandas as pd

import natural_pdf as npdf
from benchmark.configs.prompts import CSV_ONLY, CSV_QUOTING, EXACT_COPY, JSON_ONLY


class OklahomaLicenseConfig:
    """Configuration for Oklahoma license listing benchmark."""

    name = "m27"
    pdf_path = "pdfs/m27.pdf"
    pdf_path_trap = "pdfs/m27-trap.pdf"

    description = (
        "Dense license listing with license numbers, company names, addresses, phone numbers"
    )

    # Fields we're extracting
    fields = [
        "license_number",
        "type",
        "dba_name",
        "licensee_name",
        "premise_address",
        "city",
        "st",
        "zip",
        "phone_number",
        "expires",
    ]

    # =========================================================================
    # NATURAL PDF EXTRACTION (Ground Truth)
    # =========================================================================

    @staticmethod
    def _extract_page(page) -> pd.DataFrame:
        """Extract license data from a single page."""
        # Find headers in the header row area
        headers = page.region(top=55, height=30).find_all("text").expand(1).dissolve().expand(-1)

        if not headers:
            return pd.DataFrame()

        # Find data region below headers until page footer
        licensee_header = page.find("text:contains(LICENSEE NAME)")
        if not licensee_header:
            return pd.DataFrame()

        region = licensee_header.below(until="text:regex(Page \\d)", include_endpoint=False)

        if not region:
            return pd.DataFrame()

        # Find rows using text in first column
        rows = headers[0].below(width="element").find_all("text", overlap="partial")

        # Extract table using headers as vertical dividers and row bottoms as horizontal dividers
        return region.extract_table(
            verticals=headers, horizontals=rows.map(lambda r: r.bbox[3]), outer=True
        ).to_df(header=headers)

    @staticmethod
    def extract_with_natural_pdf(pdf_path: str, page_num: int = 0) -> pd.DataFrame:
        """Extract license data from a specific page."""
        pdf = npdf.PDF(pdf_path)
        df = OklahomaLicenseConfig._extract_page(pdf.pages[page_num])
        pdf.close()
        return df

    @staticmethod
    def extract_all_pages(pdf_path: str) -> pd.DataFrame:
        """Extract license data from all pages."""
        pdf = npdf.PDF(pdf_path)
        all_dfs = []

        for page_num, page in enumerate(pdf.pages):
            df = OklahomaLicenseConfig._extract_page(page)
            if not df.empty:
                df["page"] = page_num + 1
                all_dfs.append(df)

        pdf.close()
        return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    # =========================================================================
    # LLM PROMPTS
    # =========================================================================

    prompt = f"""Extract this license listing as a CSV.

Columns: LICENSE NUMBER, TYPE, DBA NAME, LICENSEE NAME, PREMISE ADDRESS, CITY, ST, ZIP, PHONE NUMBER, EXPIRES

{EXACT_COPY}

{CSV_QUOTING}

{CSV_ONLY}"""

    json_prompt = f"""Extract this license listing as JSON.

Return an array where each object has:
{{
  "license_number": "number exactly as shown",
  "type": "type code",
  "dba_name": "DBA name exactly as shown",
  "licensee_name": "licensee name exactly as shown",
  "premise_address": "address exactly as shown",
  "city": "city exactly as shown",
  "st": "state code",
  "zip": "ZIP exactly as shown",
  "phone_number": "phone or - if none",
  "expires": "date exactly as shown"
}}

{EXACT_COPY}

{JSON_ONLY}"""

    # =========================================================================
    # COMPARISON FIELDS
    # =========================================================================

    comparison_fields = [
        {
            "name": "bashu_license",
            "description": "G/6 confusion",
            "expected_original": "632575",
            "expected_trap": "G32575",
        },
        {
            "name": "backdoor_city",
            "description": "0/O confusion",
            "expected_original": "OKLAHOMA CITY",
            "expected_trap": "OKLAH0MA CITY",
        },
        {
            "name": "ajanta_zip",
            "description": "I/1 confusion",
            "expected_original": "73120",
            "expected_trap": "73I20",
        },
        {
            "name": "united_zip",
            "description": "I/1 confusion",
            "expected_original": "74115",
            "expected_trap": "74II5",
        },
        {
            "name": "southwest_licensee",
            "description": "VV/W confusion",
            "expected_original": "SOUTHWEST AIRLINES CO",
            "expected_trap": "SOUTHVVEST AIRLINES",
        },
    ]
