# PDF Extraction Task Catalog

A comprehensive catalog of PDF extraction tasks that users need to accomplish with Natural PDF, organized by category with difficulty levels and required techniques.

---

## 1. Basic Tasks
*First things users try when getting started*

### 1.1 Load and Extract All Text
- **Difficulty**: Beginner
- **Techniques**: `PDF()`, `page.extract_text()`
- **Variations**: Local file, URL, password-protected
- **Example PDFs**: Any native PDF

### 1.2 Get Document Metadata
- **Difficulty**: Beginner
- **Techniques**: `pdf.metadata`
- **Variations**: None
- **Notes**: Title, author, creation date, etc.

### 1.3 Count Pages and Navigate
- **Difficulty**: Beginner
- **Techniques**: `len(pdf.pages)`, `pdf.pages[n]`
- **Variations**: Single page, page ranges, iteration

### 1.4 Find Text Containing Specific Words
- **Difficulty**: Beginner
- **Techniques**: `page.find('text:contains("word")')`, `page.find_all()`
- **Variations**: Case sensitivity, regex patterns
- **Example PDFs**: `01-practice.pdf`

### 1.5 Visualize Found Elements
- **Difficulty**: Beginner
- **Techniques**: `element.show()`, `page.show()`, highlights
- **Variations**: Single element, collections, color customization

### 1.6 Extract Text from a Specific Region
- **Difficulty**: Beginner
- **Techniques**: `page.create_region()`, `region.extract_text()`
- **Variations**: Fixed coordinates, element-based regions

---

## 2. Table Tasks
*All table-related extraction challenges*

### 2.1 Extract Simple Table with Lines
- **Difficulty**: Beginner
- **Techniques**: `page.extract_table(method="pdfplumber")`
- **Variations**: Single table, multiple tables
- **Example PDFs**: `01-practice.pdf`

### 2.2 Extract Table with TATR (AI-based)
- **Difficulty**: Intermediate
- **Techniques**: `page.analyze_layout(engine="tatr")`, `region.extract_table()`
- **Variations**: Scanned documents, irregular layouts
- **Dependencies**: AI extras

### 2.3 Extract Table Without Visible Borders (Unruled)
- **Difficulty**: Intermediate
- **Techniques**: `method="text"`, whitespace clustering, `page.detect_lines()`
- **Variations**: Dense text, varying column widths
- **Example PDFs**: `m27.pdf` (Oklahoma license listing)
- **Notes**: Use vertical_strategy="text" in pdfplumber settings

### 2.4 Extract Table Using Custom Guides
- **Difficulty**: Intermediate
- **Techniques**: `Guides`, `verticals=`, `horizontals=`, `outer=True`
- **Variations**: Header-based guides, line-based guides
- **Example PDFs**: `guides-expenses-sample.pdf`, `hebrew-table.pdf`

### 2.5 Extract Multi-Page Table
- **Difficulty**: Intermediate
- **Techniques**: Flows system, `multipage=True`, page iteration
- **Variations**: Continuous tables, repeated headers
- **Example PDFs**: `multipage-table.pdf`, `multipage-table-african-recipes.pdf`

### 2.6 Extract Table with Nested/Complex Headers
- **Difficulty**: Advanced
- **Techniques**: Manual header parsing, column merging
- **Variations**: Multi-row headers, merged cells
- **Example PDFs**: `30.pdf` (Arkansas State graduation rates)

### 2.7 Extract RTL (Right-to-Left) Table
- **Difficulty**: Advanced
- **Techniques**: Standard extraction, proper guide ordering
- **Variations**: Hebrew, Arabic tables
- **Example PDFs**: `hebrew-table.pdf`, `arabic.pdf`

### 2.8 Extract Table from Dense/Microscopic Text
- **Difficulty**: Advanced
- **Techniques**: `x_tolerance`, `y_tolerance`, `dedupe_chars()`, high-resolution OCR
- **Variations**: 5pt fonts, overlapping characters
- **Example PDFs**: Sheriff's disciplinary log (q4DXYk8)
- **Notes**: Auto-detect dense text by character overlap analysis

### 2.9 Extract Table Spanning Multiple Columns
- **Difficulty**: Advanced
- **Techniques**: Column detection, flows, region splitting
- **Variations**: 2-column, 3-column layouts
- **Example PDFs**: `multicolumn.pdf`, `multipage-multicol.pdf`

### 2.10 Extract Table with Multi-Line Cells
- **Difficulty**: Advanced
- **Techniques**: Post-processing with pandas, row grouping
- **Variations**: Wrapped text, logical record spanning rows
- **Notes**: Natural PDF extracts cells; pandas cleans up

---

## 3. Text Extraction Tasks
*Text-related challenges beyond simple extraction*

### 3.1 Extract Text Below/Above a Header
- **Difficulty**: Beginner
- **Techniques**: `element.below()`, `element.above()`, `until=`
- **Variations**: Fixed height, until next element
- **Example PDFs**: `01-practice.pdf`

### 3.2 Extract Key-Value Pairs (Form Fields)
- **Difficulty**: Intermediate
- **Techniques**: `label.right()`, `.find('text:contains(":")')`, spatial navigation
- **Variations**: Labels on left, labels above
- **Example PDFs**: `01-practice.pdf`

### 3.3 Extract Text by Font Properties
- **Difficulty**: Intermediate
- **Techniques**: `text:bold`, `text[size>12]`, `text[fontname=X]`
- **Variations**: Bold, italic, specific sizes, font names

### 3.4 Extract Text by Color
- **Difficulty**: Intermediate
- **Techniques**: `text[color~=red]`, `text[color=rgb(0,0,255)]`
- **Variations**: Named colors, RGB values, fuzzy matching

### 3.5 Extract Sections Between Markers
- **Difficulty**: Intermediate
- **Techniques**: `page.get_sections()`, `start_elements`, `end_elements`
- **Variations**: Line dividers, heading-based sections
- **Example PDFs**: `Atlanta_Public_Schools_GA_sample.pdf`

### 3.6 Extract Text with Layout Preservation
- **Difficulty**: Intermediate
- **Techniques**: `page.extract_text(layout=True)`, spatial coordinates
- **Variations**: Multi-column text, forms

### 3.7 Handle Text with Formatting Codes
- **Difficulty**: Advanced
- **Techniques**: Element filtering, embedded code extraction
- **Variations**: Gray markers like [RP], [OL], [ST]
- **Example PDFs**: Financial disclosure (zxyRByM)

### 3.8 Handle PDF Text Corruption (Null Bytes)
- **Difficulty**: Advanced
- **Techniques**: Text cleaning, artifact detection
- **Variations**: Password-protected conversion artifacts
- **Example PDFs**: Corrupted financial disclosure (zxyRByM)
- **Notes**: Automatic detection and cleanup needed

### 3.9 Extract Underlined/Strikethrough Text
- **Difficulty**: Advanced
- **Techniques**: `rect[height<3]` detection, spatial association with text
- **Variations**: Legislative markup, legal documents
- **Example PDFs**: Georgia House Bill (obe1Vq5)
- **Notes**: Currently requires manual thin rect detection

### 3.10 Handle Character Overlap/Dense Text
- **Difficulty**: Advanced
- **Techniques**: pdfplumber `x_tolerance_ratio`, `dedupe_chars()`
- **Variations**: Microscopic fonts, overlapping characters
- **Notes**: Need auto-detection of character overlap patterns

---

## 4. Layout Tasks
*Handling complex document structures*

### 4.1 Detect Document Layout (YOLO)
- **Difficulty**: Intermediate
- **Techniques**: `page.analyze_layout(engine="yolo")`
- **Variations**: Tables, figures, headings, paragraphs
- **Dependencies**: YOLO extras

### 4.2 Detect Table Structure (TATR)
- **Difficulty**: Intermediate
- **Techniques**: `page.analyze_layout(engine="tatr")`
- **Variations**: Table-focused analysis
- **Dependencies**: AI extras

### 4.3 Handle Multi-Column Documents
- **Difficulty**: Intermediate
- **Techniques**: Column detection, reading order restoration
- **Variations**: 2-column, 3-column, mixed layouts
- **Example PDFs**: `multicolumn.pdf`

### 4.4 Exclude Headers and Footers
- **Difficulty**: Intermediate
- **Techniques**: `pdf.add_exclusion()`, lambda functions
- **Variations**: Fixed regions, element-based exclusions
- **Example PDFs**: `0500000US42007.pdf`

### 4.5 Group Pages by Content
- **Difficulty**: Intermediate
- **Techniques**: `pages.groupby()`, selector or callable grouping
- **Variations**: By header text, by content type

### 4.6 Navigate Across Pages (Multipage)
- **Difficulty**: Intermediate
- **Techniques**: `multipage=True`, `FlowRegion`, global `auto_multipage` option
- **Variations**: Content spanning pages, cross-page navigation

### 4.7 Handle Rotated Content
- **Difficulty**: Advanced
- **Techniques**: Page rotation detection, coordinate transformation
- **Variations**: 90-degree, 180-degree rotations

### 4.8 Detect and Navigate by Document Sections
- **Difficulty**: Intermediate
- **Techniques**: `page.get_sections()`, heading detection
- **Variations**: Numbered headings, styled headings

### 4.9 Handle Mixed Content Pages
- **Difficulty**: Advanced
- **Techniques**: Layout analysis + spatial navigation
- **Variations**: Text + tables + images on same page
- **Notes**: Spatial navigation designed for this

---

## 5. OCR Tasks
*Scanned document challenges*

### 5.1 Basic OCR Application
- **Difficulty**: Intermediate
- **Techniques**: `page.apply_ocr(engine="easyocr")`
- **Variations**: Different engines (easyocr, surya, paddle, doctr)
- **Example PDFs**: `needs-ocr.pdf`, `tiny-ocr.pdf`

### 5.2 OCR with Language Specification
- **Difficulty**: Intermediate
- **Techniques**: `languages=['en', 'es']`, multi-language support
- **Variations**: Single language, multiple languages

### 5.3 OCR with High Resolution
- **Difficulty**: Intermediate
- **Techniques**: `resolution=200`, `resolution=300`
- **Variations**: Speed vs. accuracy tradeoffs
- **Notes**: Higher resolution for small text

### 5.4 OCR Confidence Filtering
- **Difficulty**: Intermediate
- **Techniques**: `min_confidence=0.7`, `text[confidence>0.8]`
- **Variations**: Threshold tuning, visual confidence display

### 5.5 Multi-Engine OCR Workflow
- **Difficulty**: Advanced
- **Techniques**: `detect_only=True`, engine comparison
- **Variations**: Detection-only + recognition, side-by-side comparison
- **Notes**: Use one engine for detection, another for recognition

### 5.6 OCR Correction with LLM
- **Difficulty**: Advanced
- **Techniques**: `page.correct_ocr()`, LLM integration
- **Variations**: Context-aware correction

### 5.7 Save Searchable PDF
- **Difficulty**: Intermediate
- **Techniques**: `pdf.save_searchable("output.pdf")`
- **Variations**: Single page, full document

### 5.8 Detect Colored Areas in Scanned Documents
- **Difficulty**: Advanced
- **Techniques**: Image analysis, color blob detection (future feature)
- **Variations**: Yellow highlights, colored backgrounds
- **Notes**: User-requested: `area[color~=yellow][size>100]`

### 5.9 OCR Specific Region Only
- **Difficulty**: Intermediate
- **Techniques**: `region.apply_ocr()`
- **Variations**: Partial page OCR

---

## 6. Batch Processing Tasks
*Multi-document workflows*

### 6.1 Process Multiple PDFs
- **Difficulty**: Intermediate
- **Techniques**: `PDFCollection()`, glob patterns
- **Variations**: Directory, URL list, pattern matching

### 6.2 Extract Same Fields from Multiple Documents
- **Difficulty**: Intermediate
- **Techniques**: Loop over pages, consistent extraction logic
- **Variations**: Template-based extraction

### 6.3 Semantic Search Across Documents
- **Difficulty**: Intermediate
- **Techniques**: `collection.init_search()`, `collection.find_relevant()`
- **Variations**: Persisted index, different embedding models
- **Dependencies**: Search extras

### 6.4 Classify Documents by Type
- **Difficulty**: Intermediate
- **Techniques**: `pdf.classify_pages()`, `pdfs.classify_all()`
- **Variations**: Vision-based, text-based classification
- **Dependencies**: Classification extras

### 6.5 Export to DataFrame/CSV
- **Difficulty**: Beginner
- **Techniques**: `table.to_df()`, pandas integration
- **Variations**: Single table, multi-page aggregation

### 6.6 Filter Pages by Category
- **Difficulty**: Intermediate
- **Techniques**: `pages.filter(lambda p: p.category == 'invoice')`
- **Variations**: Save filtered pages, process filtered pages

### 6.7 Handle Massive Documents (10K+ pages)
- **Difficulty**: Advanced
- **Techniques**: Lazy loading, chunked processing (future feature)
- **Variations**: Memory management, progress tracking
- **Example PDFs**: Puerto Rico court (34,606 pages)
- **Notes**: Current limitation - memory issues on very large docs

---

## 7. AI/ML Tasks
*Using LLMs and document QA*

### 7.1 Document Question Answering
- **Difficulty**: Intermediate
- **Techniques**: `page.ask("What is the invoice total?")`
- **Variations**: Single question, batch questions
- **Dependencies**: QA extras

### 7.2 Batch Question Answering
- **Difficulty**: Intermediate
- **Techniques**: `page.ask([question1, question2, ...])`
- **Variations**: Multiple questions at once (faster than loop)

### 7.3 Structured Data Extraction with LLM
- **Difficulty**: Advanced
- **Techniques**: Pydantic schemas, LLM integration
- **Variations**: OpenAI, Gemini compatible APIs

### 7.4 Visualize QA Answer Sources
- **Difficulty**: Intermediate
- **Techniques**: `answer.show()`, source elements highlighting
- **Variations**: Single answer, multiple answers

### 7.5 Document Classification (Vision)
- **Difficulty**: Intermediate
- **Techniques**: `pdf.classify_pages(['diagram', 'text', 'blank'], using='vision')`
- **Variations**: Custom categories

### 7.6 Document Classification (Text)
- **Difficulty**: Intermediate
- **Techniques**: `pdf.classify_pages(['invoice', 'receipt'], using='text')`
- **Variations**: OCR-processed documents

### 7.7 Checkbox/Form Field Detection
- **Difficulty**: Advanced
- **Techniques**: Visual element detection, spatial analysis
- **Variations**: Checked vs. unchecked detection
- **Example PDFs**: `01-practice.pdf` (Repeat? column)

---

## 8. Edge Cases
*Unusual or difficult scenarios*

### 8.1 Extract from Password-Protected PDF
- **Difficulty**: Beginner
- **Techniques**: `PDF(path, password="secret")`
- **Variations**: User password, owner password
- **Example PDFs**: `confidential.pdf`

### 8.2 Handle Watermarked Documents
- **Difficulty**: Intermediate
- **Techniques**: Exclusions, layer filtering
- **Variations**: Text watermarks, image watermarks
- **Example PDFs**: `watermark.pdf`

### 8.3 Extract from Classified/Redacted Documents
- **Difficulty**: Intermediate
- **Techniques**: Handle black rectangles, detect redactions
- **Example PDFs**: `cia-doc.pdf`, `classified.pdf`

### 8.4 Handle Non-Latin Scripts
- **Difficulty**: Intermediate
- **Techniques**: Proper font handling, RTL support, OCR language settings
- **Variations**: Arabic, Hebrew, Chinese, Japanese, Cyrillic, etc.
- **Example PDFs**: `arabic.pdf`, `hebrew-table.pdf`, Russian doc (b5eVqGg)

### 8.5 Extract Mathematical Formulas
- **Difficulty**: Advanced
- **Techniques**: YOLO `isolate_formulas` region detection
- **Variations**: Complex notation, subscripts, superscripts
- **Example PDFs**: Russian government document (b5eVqGg)
- **Notes**: YOLO can detect formula regions

### 8.6 Handle Documents with Figures/Charts
- **Difficulty**: Intermediate
- **Techniques**: `region[type=figure]`, figure detection
- **Variations**: Separate figures from text
- **Notes**: YOLO detects figures and figure captions

### 8.7 Extract from Scanned Historical Documents
- **Difficulty**: Advanced
- **Techniques**: High-resolution OCR, preprocessing
- **Variations**: Degraded quality, unusual fonts

### 8.8 Handle Very Wide Tables (Landscape)
- **Difficulty**: Intermediate
- **Techniques**: Proper coordinate handling
- **Variations**: Multi-page wide tables
- **Example PDFs**: Serbian wide tables (lbODqev)

### 8.9 Extract Book Entry Logs (Repeating Sections)
- **Difficulty**: Intermediate
- **Techniques**: Section extraction, section boundaries
- **Variations**: Library weeding logs, inventory lists
- **Example PDFs**: `Atlanta_Public_Schools_GA_sample.pdf`

### 8.10 Handle Trap Characters (Adversarial)
- **Difficulty**: Advanced
- **Techniques**: Character validation, known trap detection
- **Variations**: 1/I, 0/O, 6/G, m/rn confusions
- **Example PDFs**: `01-practice-trap.pdf`, `m27-trap.pdf`, `Atlanta_Public_Schools_GA_sample-trap.pdf`
- **Notes**: Natural PDF extracts exact characters; LLMs may hallucinate

---

## Real-World Benchmark Examples

Based on benchmark configurations, these represent complete end-to-end extraction workflows:

### Pennsylvania Election Results
- **PDF**: `0500000US42001.pdf`
- **Task**: Extract candidate names, vote counts, positions, and locations across multiple pages
- **Techniques**: Size-based element finding (`text[size=max()]`), spatial navigation, Guides system, table extraction, DataFrame output
- **Difficulty**: Advanced

### Atlanta Public Schools Library Weeding Log
- **PDF**: `Atlanta_Public_Schools_GA_sample.pdf`
- **Task**: Extract book entries with title, author, ISBN, barcode, price, dates
- **Techniques**: Font-based selection, `.below(until=)`, exclusions, multi-page extraction, DataFrame cleaning
- **Difficulty**: Advanced

### Oklahoma License Listing
- **PDF**: `m27.pdf`
- **Task**: Extract dense license data with numbers, names, addresses
- **Techniques**: Header-based guides, row detection, `expand().dissolve()`, outer boundaries
- **Difficulty**: Advanced

### Hebrew Economic Indicators
- **PDF**: `hebrew-table.pdf`
- **Task**: Extract RTL economic data table with year columns
- **Techniques**: Vertical divider detection, guide-based extraction, `outer=True`
- **Difficulty**: Intermediate

### Health Inspection Report
- **PDF**: `01-practice.pdf`
- **Task**: Extract form fields, summary text, and violations table with checkboxes
- **Techniques**: Color selection, spatial navigation, checkbox detection via rect+line
- **Difficulty**: Intermediate

### Arkansas State Graduation Rates
- **PDF**: `30.pdf`
- **Task**: Extract demographic tables with nested column headers
- **Techniques**: Size-based title finding, region-based extraction, demographic row iteration
- **Difficulty**: Advanced

---

## Difficulty Legend

- **Beginner**: Basic API usage, single method calls, straightforward documents
- **Intermediate**: Combining multiple techniques, handling variations, configuration tuning
- **Advanced**: Complex workflows, edge cases, AI integration, performance considerations

## Technique Quick Reference

| Technique | When to Use |
|-----------|-------------|
| `page.extract_text()` | Basic full-page extraction |
| `page.find()` / `page.find_all()` | Locate specific elements |
| `.below()` / `.above()` / `.left()` / `.right()` | Spatial navigation |
| `Guides` | Custom table column/row definitions |
| `page.analyze_layout()` | AI-based document structure detection |
| `page.apply_ocr()` | Scanned document text extraction |
| `page.ask()` | Natural language question answering |
| `pdf.add_exclusion()` | Remove headers/footers/watermarks |
| `page.get_sections()` | Split document into logical sections |
| `PDFCollection` | Multi-document processing |
