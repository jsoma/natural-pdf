---
skip: true
---

# Solve: real PDFs, worked end to end

Every page in this section starts from a genuinely bad PDF — the kind that shows up after a records request or a scrape — states why it resists extraction, and works the solution in runnable code. Each page is also downloadable as a notebook. Same-repository examples use checked-out files under `pdfs/` during the docs build, and notebook export rewrites those paths to raw GitHub URLs so you can still run them top to bottom in Colab. A few externally hosted records remain URL inputs because they are not committed to this repository.

If you're new to the library, start with [Learn](../learn/01-text-and-tables.md); come here when you have a document that looks like one of these.

## Tables without borders

<div class="npdf-cards">
  <a class="npdf-card" href="zebra-stripe-table.md">
    <img src="assets/zebra-stripe-table-thumb.png" alt="Oklahoma alcohol licensee list">
    <strong>Zebra-stripe tables with no ruling lines</strong>
    <span class="npdf-card-desc">An Oklahoma licensee list where alternating row colors stand in for borders and cells wrap across lines — the table detector merges everything.</span>
  </a>
  <a class="npdf-card" href="use-of-force-logs.md">
    <img src="assets/use-of-force-logs-thumb.png" alt="Vancouver use-of-force records">
    <strong>Use-of-force records in six-point type</strong>
    <span class="npdf-card-desc">Police use-of-force data in tiny type with acres of whitespace and not one ruling line — detectors see haze, not columns.</span>
  </a>
  <a class="npdf-card" href="borderless-call-log.md">
    <img src="assets/borderless-call-log-thumb.png" alt="Niagara Falls 911 call log">
    <strong>A 43-page borderless call log</strong>
    <span class="npdf-card-desc">A spreadsheet printed to PDF: erratic column widths, columns that go empty for pages, and a "2770 Records Found" line aimed at your data.</span>
  </a>
</div>

## Tables that span pages

<div class="npdf-cards">
  <a class="npdf-card" href="serbian-multipage-table.md">
    <img src="assets/serbian-multipage-table-thumb.png" alt="Serbian regulatory fee table">
    <strong>A table that runs for eight pages of a Serbian law</strong>
    <span class="npdf-card-desc">One fee table from page 90 to 97 with no boundaries between pages, section headers mid-stream, and a formula that's an image, not text.</span>
  </a>
</div>

## Scans and OCR

<div class="npdf-cards">
  <a class="npdf-card" href="pixelated-scan-table.md">
    <img src="assets/pixelated-scan-table-thumb.png" alt="Pixelated call center report">
    <strong>A pixelated FOIA scan with an invisible table</strong>
    <span class="npdf-card-desc">A call-center report scanned so badly the numbers are hard for humans — no text layer, half-dissolved ruling lines, OCR required.</span>
  </a>
</div>

## Multi-column layouts

<div class="npdf-cards">
  <a class="npdf-card" href="multi-column-reflow.md">
    <img src="assets/multi-column-reflow-thumb.png" alt="Three-column page with two tables">
    <strong>Multi-column pages that read in the wrong order</strong>
    <span class="npdf-card-desc">Three columns with tables buried in them — naive extraction reads straight across all three at once, interleaving unrelated sentences.</span>
  </a>
</div>

## Redactions and form printouts

<div class="npdf-cards">
  <a class="npdf-card" href="complaint-database-printout.md">
    <img src="assets/complaint-database-printout-thumb.png" alt="Law enforcement complaint records">
    <strong>A relational database printed to PDF, with redactions</strong>
    <span class="npdf-card-desc">Complaint records as repeating form blocks with one-to-many tables inside — and redaction boxes that break column detection where it hurts.</span>
  </a>
</div>

## Non-Latin and right-to-left scripts

<div class="npdf-cards">
  <a class="npdf-card" href="arabic-election-table.md">
    <img src="assets/arabic-election-table-thumb.png" alt="Tunisian election results in Arabic">
    <strong>Arabic election results across four pages</strong>
    <span class="npdf-card-desc">A Tunisian results table in right-to-left Arabic script with spanning and rotated headers, split over four pages.</span>
  </a>
</div>

## Where these come from

These documents are drawn from [Bad PDFs](https://badpdfs.com/), a collection of real problem PDFs submitted by journalists and researchers, each with a worked Natural PDF solution. If you have a PDF that deserves a page here, submit it there.
