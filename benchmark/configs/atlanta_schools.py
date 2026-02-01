"""
Atlanta Public Schools Library Weeding Log - Benchmark Configuration

PDF: pdfs/Atlanta_Public_Schools_GA_sample.pdf
Structure: Header info + repeated book entry sections divided by horizontal lines
"""

import pandas as pd

import natural_pdf as npdf
from benchmark.configs.prompts import CSV_ONLY, CSV_QUOTING, EXACT_COPY, JSON_ONLY


class AtlantaSchoolsConfig:
    """Configuration for Atlanta Public Schools Library Weeding Log benchmark."""

    name = "Atlanta_Public_Schools_GA_sample"
    pdf_path = "pdfs/Atlanta_Public_Schools_GA_sample.pdf"
    pdf_path_trap = "pdfs/Atlanta_Public_Schools_GA_sample-trap.pdf"

    description = "Library weeding log with book entries containing ISBNs, barcodes, prices"

    # Fields we're extracting
    fields = [
        "title",
        "author",
        "isbn",
        "published",
        "site",
        "barcode",
        "price",
        "acquired",
        "removed_by",
        "removal_date",
    ]

    # =========================================================================
    # NATURAL PDF EXTRACTION (Ground Truth)
    # =========================================================================

    @staticmethod
    def _extract_page(page) -> pd.DataFrame:
        """Extract book entries from a single page."""
        # Find book titles by font variant and size
        titles = page.find_all("text[font_variant=AAAAAB][size=10]")
        if not titles:
            return pd.DataFrame()

        # Get sections for each book
        books = titles.below(
            until="text[font_variant=AAAAAB][size=10]", include_endpoint=False, include_source=True
        )

        # Build DataFrame with all fields
        df = pd.DataFrame(
            {
                "title": titles.extract_each_text(),
                "author": books.find("text:contains(Author)").extract_each_text(),
                "isbn": books.find("text:contains(ISBN)")
                .below(
                    until="text:contains(Price)",
                    width="element",
                    include_source=True,
                    include_endpoint=False,
                )
                .extract_each_text(),
                "published": books.find("text:contains(Published)").extract_each_text(),
                "site": books.find("text:contains(Site)")
                .below()
                .clip(books)
                .apply(lambda area: area.find_all("text[x0<47][size=10]").extract_text()),
                "barcode": books.find("text:contains(Barcode)")
                .below(width="element", height=12)
                .find("text", overlap="partial", default=None)
                .extract_each_text(default=""),
                "price": books.find("text:contains(Price)")
                .below(width="element", height=12)
                .find("text", overlap="partial", default=None)
                .extract_each_text(default=""),
                "acquired": books.find("text:contains(Acquired)")
                .below(width="element", height=12)
                .find("text", overlap="partial", default=None)
                .extract_each_text(default=""),
                "removed_by": books.find("text:contains(Removed By)")
                .below(width="element", height=12)
                .find("text", overlap="partial", default=None)
                .extract_each_text(default=""),
                "removal_date": books.above(until="text[size>10]").endpoints.extract_each_text(),
            }
        )

        # Clean up extracted text
        df["title"] = df["title"].str.replace(r"\(Removed.*", "", regex=True).str.strip()
        df["author"] = df["author"].str.replace("Author:", "").str.strip()
        df["isbn"] = (
            df["isbn"]
            .str.replace("ISBN:", "")
            .str.replace("\n", " ")
            .str.replace(r"\(.*\)", "", regex=True)
            .str.strip()
        )
        df["published"] = df["published"].str.replace("Published:", "").str.strip()
        df["removal_date"] = df["removal_date"].str.split(" - ").str[0]

        return df

    @staticmethod
    def extract_with_natural_pdf(pdf_path: str, page_num: int = 0) -> pd.DataFrame:
        """Extract book entries from a specific page."""
        pdf = npdf.PDF(pdf_path)
        pdf.add_exclusion(lambda page: page.find("line[width>=2]").above())
        pdf.add_exclusion(lambda page: page.find_all("line")[-1].below())

        df = AtlantaSchoolsConfig._extract_page(pdf.pages[page_num])
        pdf.close()
        return df

    @staticmethod
    def extract_all_pages(pdf_path: str) -> pd.DataFrame:
        """Extract book entries from all pages."""
        pdf = npdf.PDF(pdf_path)
        pdf.add_exclusion(lambda page: page.find("line[width>=2]").above())
        pdf.add_exclusion(lambda page: page.find_all("line")[-1].below())

        all_dfs = []
        for page_num, page in enumerate(pdf.pages):
            df = AtlantaSchoolsConfig._extract_page(page)
            if not df.empty:
                df["page"] = page_num + 1
                all_dfs.append(df)

        pdf.close()
        return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    # =========================================================================
    # LLM PROMPTS
    # =========================================================================

    # Simple CSV export prompt
    prompt = f"""Extract all book entries from this Library Weeding Log as a CSV.

Columns: Title, Author, ISBN, Published, Site, Barcode, Price, Acquired, Removed_By, Removal_Date

{EXACT_COPY}

{CSV_QUOTING}

{CSV_ONLY}"""

    json_prompt = f"""Extract all book entries as JSON. For each book include:
{{
  "title": "book title exactly as written",
  "author": "author name exactly as written",
  "isbn": "ISBN",
  "published": "year",
  "site": "school name",
  "barcode": "barcode number exactly as written",
  "price": "price or empty string",
  "acquired": "acquisition date",
  "removed_by": "ID or name",
  "removal_date": "date book was removed"
}}

Return a JSON array of book objects.

{EXACT_COPY}

{JSON_ONLY}"""

    # =========================================================================
    # COMPARISON FIELDS (what to check between Natural PDF and LLM)
    # =========================================================================

    comparison_fields = [
        {
            "name": "author_voodoo",
            "description": "ll/il swap",
            "expected_original": "Kelly Wand, book editor.",
            "expected_trap": "Keily Wand, book editor.",
            "natural_pdf_path": lambda df: (
                df[df["author"].str.contains("Wand", na=False)]["author"].iloc[0]
                if len(df[df["author"].str.contains("Wand", na=False)]) > 0
                else None
            ),
            "llm_csv_column": "Author",
            "llm_csv_filter": lambda row: "Voodoo" in row.get("Title", "")
            or "Wand" in row.get("Author", ""),
        },
        {
            "name": "author_kwame",
            "description": "m/rn confusion",
            "expected_original": "Mbalia, Kwame.",
            "expected_trap": "Mbalia, Kwarne.",
            "natural_pdf_path": lambda df: (
                df[df["author"].str.contains("Mbalia", na=False)]["author"].iloc[0]
                if len(df[df["author"].str.contains("Mbalia", na=False)]) > 0
                else None
            ),
            "llm_csv_column": "Author",
            "llm_csv_filter": lambda row: "Tristan" in row.get("Title", "")
            or "Mbalia" in row.get("Author", ""),
        },
        {
            "name": "isbn_tristan",
            "description": "1/I confusion",
            "expected_original": "978-1-36803993-2",
            "expected_trap": "978-I-36803993-2",
            "natural_pdf_path": lambda df: (
                df[df["author"].str.contains("Mbalia", na=False)]["isbn"].iloc[0]
                if len(df[df["author"].str.contains("Mbalia", na=False)]) > 0
                else None
            ),
            "llm_csv_column": "ISBN",
            "llm_csv_filter": lambda row: "Tristan" in row.get("Title", "")
            or "Mbalia" in row.get("Author", ""),
        },
        {
            "name": "barcode_tristan",
            "description": "1/I confusion (multiple)",
            "expected_original": "32441014018707",
            "expected_trap": "3244I0I40I8707",
            "natural_pdf_path": lambda df: (
                df[df["author"].str.contains("Mbalia", na=False)]["barcode"].iloc[0]
                if len(df[df["author"].str.contains("Mbalia", na=False)]) > 0
                else None
            ),
            "llm_csv_column": "Barcode",
            "llm_csv_filter": lambda row: "Tristan" in row.get("Title", "")
            or "Mbalia" in row.get("Author", ""),
        },
        {
            "name": "isbn_buddhism",
            "description": "0/O confusion",
            "expected_original": "0-8160-2442-1",
            "expected_trap": "O-8160-2442-1",
            "natural_pdf_path": lambda df: (
                df[df["author"].str.contains("Wangu", na=False)]["isbn"].iloc[0]
                if len(df[df["author"].str.contains("Wangu", na=False)]) > 0
                else None
            ),
            "llm_csv_column": "ISBN",
            "llm_csv_filter": lambda row: "Buddhism" in row.get("Title", "")
            or "Wangu" in row.get("Author", ""),
        },
        {
            "name": "price_upside",
            "description": "1/I confusion",
            "expected_original": "$15.00",
            "expected_trap": "$I5.00",
            "natural_pdf_path": lambda df: (
                df[df["author"].str.contains("Lamana", na=False)]["price"].iloc[0]
                if len(df[df["author"].str.contains("Lamana", na=False)]) > 0
                else None
            ),
            "llm_csv_column": "Price",
            "llm_csv_filter": lambda row: "Upside" in row.get("Title", "")
            or "Lamana" in row.get("Author", ""),
        },
    ]
