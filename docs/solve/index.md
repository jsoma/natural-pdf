---
skip: true
---

# Solve: real PDFs, worked end to end

Every page in this section starts from a genuinely bad PDF — the kind that shows up after a records request or a scrape — states why it resists extraction, and works the solution in runnable code. Each page is also downloadable as a notebook. Same-repository examples use checked-out files under `pdfs/` during the docs build, and notebook export rewrites those paths to raw GitHub URLs so you can still run them top to bottom in Colab. A few externally hosted records remain URL inputs because they are not committed to this repository.

If you're new to the library, start with [Learn](../learn/01-text-and-tables.md); come here when you have a document that looks like one of these.

## Tables without borders

<div class="grid cards" markdown>

-   ![Oklahoma alcohol licensee list](assets/zebra-stripe-table-thumb.png)

    **[Zebra-stripe tables with no ruling lines](zebra-stripe-table.md)**

    An Oklahoma licensee list where alternating row colors stand in for borders and cells wrap across lines — the table detector merges everything.

-   ![Vancouver use-of-force records](assets/use-of-force-logs-thumb.png)

    **[Use-of-force records in six-point type](use-of-force-logs.md)**

    Police use-of-force data in tiny type with acres of whitespace and not one ruling line — detectors see haze, not columns.

-   ![Niagara Falls 911 call log](assets/borderless-call-log-thumb.png)

    **[A 43-page borderless call log](borderless-call-log.md)**

    A spreadsheet printed to PDF: erratic column widths, columns that go empty for pages, and a "2770 Records Found" line aimed at your data.

</div>

## Tables that span pages

<div class="grid cards" markdown>

-   ![Serbian regulatory fee table](assets/serbian-multipage-table-thumb.png)

    **[A table that runs for eight pages of a Serbian law](serbian-multipage-table.md)**

    One fee table from page 90 to 97 with no boundaries between pages, section headers mid-stream, and a formula that's an image, not text.

</div>

## Scans and OCR

<div class="grid cards" markdown>

-   ![Pixelated call center report](assets/pixelated-scan-table-thumb.png)

    **[A pixelated FOIA scan with an invisible table](pixelated-scan-table.md)**

    A call-center report scanned so badly the numbers are hard for humans — no text layer, half-dissolved ruling lines, OCR required.

</div>

## Multi-column layouts

<div class="grid cards" markdown>

-   ![Three-column page with two tables](assets/multi-column-reflow-thumb.png)

    **[Multi-column pages that read in the wrong order](multi-column-reflow.md)**

    Three columns with tables buried in them — naive extraction reads straight across all three at once, interleaving unrelated sentences.

</div>

## Redactions and form printouts

<div class="grid cards" markdown>

-   ![Law enforcement complaint records](assets/complaint-database-printout-thumb.png)

    **[A relational database printed to PDF, with redactions](complaint-database-printout.md)**

    Complaint records as repeating form blocks with one-to-many tables inside — and redaction boxes that break column detection where it hurts.

</div>

## Non-Latin and right-to-left scripts

<div class="grid cards" markdown>

-   ![Tunisian election results in Arabic](assets/arabic-election-table-thumb.png)

    **[Arabic election results across four pages](arabic-election-table.md)**

    A Tunisian results table in right-to-left Arabic script with spanning and rotated headers, split over four pages.

</div>

## Where these come from

These documents are drawn from [Bad PDFs](https://badpdfs.com/), a collection of real problem PDFs submitted by journalists and researchers, each with a worked Natural PDF solution. If you have a PDF that deserves a page here, submit it there.
