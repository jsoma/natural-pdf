# Learn

A five-part course on getting structured data out of PDFs. Each page builds
on the previous one, mostly using a single running document — a fake
inspection report — first as clean digital text, then as a scan.

Every page doubles as a notebook: use the "Open in Colab" badge to run it
yourself.

1. **[Text and tables](01-text-and-tables.md)** — open a PDF, find things
   with selectors, walk the page spatially, pull a table, exclude headers.
2. **[OCR](02-ocr.md)** — when there's no text layer: apply OCR, judge its
   quality, compare engines, re-OCR the stubborn parts.
3. **[AI extraction](03-ai-extraction.md)** — ask questions, extract
   structured records with LLMs, classify documents — and see exactly where
   AI extraction fails.
4. **[Page structure](04-page-structure.md)** — multi-column reflow, content
   that spans pages, and layout models.
5. **[Grids](05-grids.md)** — build table grids from headers, whitespace,
   content anchors, or pixel lines, and read drawn checkboxes.
6. **[Putting it together](06-putting-it-together.md)** — the capstone: turn
   a five-page library weeding report into one clean DataFrame, then batch
   the whole thing.

If you're working on a specific problem instead, jump to
[Solve](../solve/index.md) and pick the document that looks like yours.
