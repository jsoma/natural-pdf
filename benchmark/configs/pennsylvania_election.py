"""
Pennsylvania Election Results Benchmark Configuration

PDF: pdfs/0500000US42001.pdf
Structure: Multi-page election results with location headers, position subheaders, and vote tables
Output: CSV with columns [candidate, votes, position, location]
"""

import pandas as pd

import natural_pdf as npdf
from benchmark.configs.prompts import EXACT_COPY, JSON_ONLY
from natural_pdf import Guides


class PennsylvaniaElectionConfig:
    """Configuration for Pennsylvania election results benchmark."""

    name = "pennsylvania-election"
    pdf_path = "pdfs/0500000US42001.pdf"
    pdf_path_trap = None  # No trap version

    description = (
        "Pennsylvania county election results - multi-page with location/position hierarchy"
    )

    # Number of pages to process (first N pages)
    max_pages = 10

    # =========================================================================
    # NATURAL PDF EXTRACTION (Ground Truth)
    # =========================================================================

    @staticmethod
    def extract_with_natural_pdf(pdf_path: str, page_num: int = 0) -> list:
        """
        Extract election results from a single page using Natural PDF with guide-based table extraction.

        Args:
            pdf_path: Path to the PDF file
            page_num: Page number to extract (0-indexed)

        Returns:
            List of dicts with candidate, votes, position, location for this page
        """
        pdf = npdf.PDF(pdf_path)

        # Skip if page_num is beyond max_pages
        if page_num >= PennsylvaniaElectionConfig.max_pages:
            pdf.close()
            return []

        dfs = []
        page = pdf.pages[page_num]

        try:
            # Location is the largest text on the page (town name)
            location_elem = page.find("text[size=max()]")
            if not location_elem:
                pdf.close()
                return []
            location = location_elem.extract_text()

            # Positions are size=9 text (election race names)
            positions = page.find_all("text[size=9]")

            for position_elem in positions:
                try:
                    position = position_elem.extract_text()

                    # Get the table area: below position header, until Contest Totals
                    results = position_elem.below(until="text:contains(TOTAL)").below(
                        until="text:contains(Contest Totals)"
                    )

                    if not results:
                        continue

                    table_area = results.trim(method="any")

                    # Set up guides for table extraction
                    guides = Guides(table_area)
                    guides.vertical.divide(2)
                    guides.snap_to_whitespace()
                    guides.horizontal.from_lines(outer=True)

                    # Extract table
                    df = guides.extract_table().to_df(header=["candidate", "votes"])

                    # Clean votes column - remove commas, convert to int
                    df["votes"] = df["votes"].str.replace(",", "").astype(int)

                    # Add position and location
                    df["position"] = position
                    df["location"] = location

                    dfs.append(df)

                except Exception:
                    pass  # Skip positions that fail

        except Exception:
            pass  # Skip pages that fail

        pdf.close()

        # Combine all dataframes from this page
        if dfs:
            result_df = pd.concat(dfs).reset_index(drop=True)
            return result_df.to_dict("records")
        else:
            return []

    # =========================================================================
    # LLM PROMPT
    # =========================================================================

    prompt = f"""Extract election results from this document as a JSON array.

For EACH page, extract:
1. The location (town/city name - usually the largest text at the top)
2. For each election position/race on that page, extract all candidates and their vote counts

Return a JSON array with these exact fields per record:
- candidate: The candidate name (e.g., "DEM HARRIS and WALZ", "REP TRUMP and VANCE")
- votes: The vote count as an integer
- position: The election position (e.g., "Presidential Electors", "Representative in Congress")
- location: The town/city name (e.g., "Abbottstown", "Arendtsville")

Format (return ONLY the array, no wrapper object):
[
  {{"candidate": "DEM HARRIS and WALZ", "votes": 148, "position": "Presidential Electors", "location": "Abbottstown"}},
  {{"candidate": "REP TRUMP and VANCE", "votes": 348, "position": "Presidential Electors", "location": "Abbottstown"}},
  ...
]

Include ALL rows including "Write-In Totals", "Total Votes Cast", "Overvotes", "Undervotes", and "Contest Totals".

Process only the first 10 pages.

{EXACT_COPY}

{JSON_ONLY}"""

    # =========================================================================
    # COMPARISON FIELDS (values to check for accuracy)
    # =========================================================================

    comparison_fields = [
        {
            "name": "first_location",
            "description": "First location (town name)",
            "expected_original": "Abbottstown",
            "expected_trap": None,
        },
        {
            "name": "first_position",
            "description": "First election position",
            "expected_original": "Presidential Electors",
            "expected_trap": None,
        },
        {
            "name": "first_candidate",
            "description": "First candidate name",
            "expected_original": "DEM HARRIS and WALZ",
            "expected_trap": None,
        },
        {
            "name": "first_votes",
            "description": "First candidate vote count",
            "expected_original": "148",
            "expected_trap": None,
        },
        {
            "name": "harris_abbottstown_votes",
            "description": "Harris/Walz votes in Abbottstown",
            "expected_original": "148",
            "expected_trap": None,
        },
        {
            "name": "trump_abbottstown_votes",
            "description": "Trump/Vance votes in Abbottstown",
            "expected_original": "348",
            "expected_trap": None,
        },
        {
            "name": "unique_locations_10_pages",
            "description": "Number of unique locations in first 10 pages",
            "expected_original": "10",  # Approximate - each page typically has one location
            "expected_trap": None,
        },
        {
            "name": "positions_per_page",
            "description": "Typical positions per page",
            "expected_original": "5-7",  # Approximate range
            "expected_trap": None,
        },
    ]
