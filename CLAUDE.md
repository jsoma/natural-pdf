# Natural PDF Notes

This file contains important information about the Natural PDF codebase for reference.

## Development Setup

### Installation
The library is built on top of pdfplumber and provides a more intuitive interface for working with PDFs.

```python
# From source repository
pip install -e .
```

### Running Examples
```python
# Run the basic usage example
python examples/basic_usage.py [optional_pdf_path]

# Run the select_until example
python examples/select_until_example.py [optional_pdf_path]

# Run the font-aware text extraction example
python examples/font_aware_example.py [optional_pdf_path]

# Run the region sections example
python examples/region_sections_example.py [optional_pdf_path]

# Run the region expand example
python examples/region_expand_example.py [optional_pdf_path]

# Run the font variant example
python examples/font_variant_example.py [optional_pdf_path]

# Run the region image example 
python examples/region_image_example.py [optional_pdf_path]
```

## Design Principles

### Fluent API
The library follows a fluent API design pattern:
- Method chaining for building complex operations step by step
- Intuitive, discoverable methods that read like natural language
- Focus on user experience rather than implementation details
- Consistent return types that allow for further operations

### Space-First Approach
PDF extraction should reflect how humans think about document space:
- Spatial navigation (above, below, until) rather than raw coordinate math
- Region-based operations that mirror how humans describe document areas
- CSS-like selectors for finding elements based on visual properties
- Natural language descriptions of document structure

## Key Concepts

### Core Architecture
- `PDF` class: Main entry point for working with PDF documents
- `Page` class: Represents a single page with methods for extraction and navigation
- `Element` classes: Represent PDF elements (text, rect, line, etc.)
- `Region` class: Represents a rectangular area of a page
- `ElementCollection`: Represents a collection of elements with batch operations

### Spatial Navigation
- `.above()` and `.below()` methods create regions above/below elements
- `.select_until()` creates a region from one element to another
- All spatial methods can use `full_width=True` (default) to span the full page width

### Selectors
- CSS-like selector syntax: `element_type[attribute_op=value]:pseudo_class(args)`
- Element types: `text`, `rect`, `line`, etc.
- Attributes: `width`, `height`, `fontname`, `color`, etc.
- Operators: `=`, `~=`, `>=`, `<=`, `>`, `<`, `*=`
- Pseudo-classes: `:contains()`, `:starts-with()`, `:ends-with()`, `:bold`, `:italic`

### Text Style Analysis
- Automatic grouping of text by style properties (font, size, weight, etc.)
- Groups text into "Text Style 1", "Text Style 2", etc. based on visual properties
- Font variants (prefixes like 'AAAAAB+' in font names) can be used to distinguish visually different text
- Methods:
  - `page.analyze_text_styles()`: Returns labeled element collections
  - `page.highlight_text_styles()`: Visualize the text styles with highlighting
  - `page.highlight_all(include_text_styles=True)`: Include text styles in highlight_all

## Important Code Details

### Type Matching
- The `text` selector matches elements with type `text`, `char`, or `word`
- Most "text" elements are actually stored as "word" elements after grouping

### Region Elements
- `Region._is_element_in_region()` determines if an element is within a region
- Elements are considered "in a region" if their center point is within the region's boundaries
- Regions support direct image extraction and visualization with `to_image()` and `save_image()`
- Region boundaries have a 1-pixel tolerance to avoid including border elements

### Element Reading Order
- Elements are sorted in reading order (top-to-bottom, left-to-right)
- Future work will include more sophisticated reading order algorithms

### Enhanced Text Search
- By default, spaces are now preserved in word elements for better multi-word search capabilities (`keep_spaces=True`)
- The `:contains()` pseudo-class supports several advanced search options:
  - Case-insensitive search: `page.find_all('text:contains("annual report")', case=False)`
  - Regular expression search: `page.find_all('text:contains("\\d{4}\\s+report")', regex=True)`
  - Combined options: `page.find_all('text:contains("summary")', regex=True, case=False)`
- Control spaces behavior via PDF constructor: `PDF("document.pdf", keep_spaces=False)` to revert to legacy behavior
- Legacy mode breaks text at spaces, making it harder to search for multi-word phrases

### Positional Methods
- `ElementCollection` provides methods to find positional extremes of elements:
  - `.highest()`: Element with the smallest top y-coordinate (highest on page)
  - `.lowest()`: Element with the largest bottom y-coordinate (lowest on page)
  - `.leftmost()`: Element with the smallest x0 coordinate (leftmost on page)
  - `.rightmost()`: Element with the largest x1 coordinate (rightmost on page)
- All positional methods throw an error if elements are on multiple pages
- Use to find the boundaries of elements: `page.find_all('line').lowest()`

### Text Extraction
- All text extraction methods use `keep_blank_chars=True` by default to preserve text block structure
- This keeps blank spaces and helps maintain proper paragraph formatting
- Can be overridden with `extract_text(keep_blank_chars=False)` if needed

## Examples

### Selecting Text From One Element To Another
```python
# Find content from Summary to the thick line
summary = page.find('text:contains("Summary:")')
thick_line = page.find('line[width>=2]')
region = summary.select_until('line[width>=2]')
content = region.extract_text()
```

### Using Attribute Filters
```python
# Find bold text with size >= 12
headings = page.find_all('text[size>=12]:bold')

# Find thick red lines - multiple ways to specify colors
red_lines = page.find_all('line[width>=2][color~=(1,0,0)]')  # RGB tuple
red_lines = page.find_all('line[width>=2][color~=red]')      # Named color
red_lines = page.find_all('line[width>=2][color~=#ff0000]')  # Hex color

# Find text with a specific font variant (useful for distinguishing visually different text)
variant_text = page.find_all('text[font-variant="AAAAAB"]')

# Filter by both font variant and other attributes
bold_variant = page.find_all('text[font-variant="AAAAAB"][size>=10]:bold')

# Find text with various colors using different formats
blue_text = page.find_all('text[color~=blue]')            # Named color
green_text = page.find_all('text[color~=#00ff00]')        # Hex color
yellow_text = page.find_all('text[color~=(1,1,0)]')       # RGB tuple

# Find text containing specific content (multi-word search)
annual_reports = page.find_all('text:contains("Annual Report")')

# Case-insensitive search
reports = page.find_all('text:contains("annual report")', case=False)

# Regular expression search
year_reports = page.find_all('text:contains("report\\s+\\d{4}")', regex=True)
```

### Spatial Navigation with Chaining
```python
# Find all text below a heading
heading = page.find('text:contains("Financial Summary")')
content = heading.below().find_all('text')

# Create and expand regions using spatial navigation
removed_by = page.find('text:contains("Removed")').below(height=17, full_width=False)
expanded_removed_by = removed_by.expand(right=50)  # Make it wider to the right

# Expand regions with percentage factors
result_box = page.find('text:contains("Result")').below()
expanded_box = result_box.expand(width_factor=1.5, height_factor=1.2)  # 50% wider, 20% taller

# Create regions with bounded endpoints using until
# From this heading down to the next heading
heading1 = page.find('text:contains("Summary")')
heading2 = page.find('text:contains("Conclusion")')
section = heading1.below(until='text:contains("Conclusion")')

# Without including the endpoint
section_exclude_conclusion = heading1.below(until='text:contains("Conclusion")', include_until=False)

# Same works for above() method
footnotes = page.find('text:contains("References")').above(until='text:contains("Summary")')
```

### Document Section Extraction
```python
# Extract sections from a page using elements as section starts
sections = page.get_sections(start_elements='text[size>=12]')  # Using large text as section starts

# Extract sections with both start and end elements
sections = page.get_sections(
    start_elements='text[size>=12]',  # Section starts
    end_elements='line[width>=1]'     # Section ends
)

# Control how boundary elements are included
sections = page.get_sections(
    start_elements='text:bold',
    boundary_inclusion='both'  # Include both start and end elements in their sections
)
# Other boundary_inclusion options: 'start', 'end', 'none'

# Get sections from just a region instead of the full page
region = page.create_region(50, 50, page.width - 50, page.height - 50)
sections = region.get_sections(start_elements='text:bold')

# Expand the region around a section
expanded_section = sections[0].expand(left=20, right=20, top_expand=10, bottom_expand=30)

# Use percentage-based expansion
expanded_section = sections[0].expand(width_factor=1.5, height_factor=1.2)  # 50% wider, 20% taller

# Multiple ways to specify elements:
# 1. Using selector strings directly
sections = page.get_sections(start_elements='text:contains("Title")')

# 2. Using pre-selected elements
headings = page.find_all('text[size>=12]')
sections = page.get_sections(start_elements=headings)

# Work with sections across page boundaries (PageCollection)
sections = pdf.pages.get_sections(
    start_elements='text[size>=14]',
    new_section_on_page_break=True  # Create new sections at page boundaries
)

# Process extracted sections
for i, section in enumerate(sections):
    print(f"Section {i+1}:")
    if hasattr(section, 'start_element') and section.start_element:
        print(f"  Starts with: {section.start_element.text}")
    print(f"  Content: {section.extract_text()[:50]}...")
```

### Visual Debugging with Highlighting
```python
# Highlight elements with different colors and labels
page.find('text:contains("Summary:")').highlight(label="Summary")
page.find_all('line[width>=2]').highlight(label="Thick Lines")
page.find_all('text:bold').highlight(label="Bold Text")

# Show or save the highlighted PDF
page.show(labels=True)  # Returns a PIL Image
page.save_image("highlighted.png", labels=True)

# Highlight regions
region = summary.select_until('line[width>=2]')
region.highlight(label="Summary Section")

# Generate an image of just a specific region
region_image = region.to_image(resolution=150)  # Get a PIL Image of just the region
region.save_image("region_only.png")  # Save the region image to a file

# Control region image rendering
region.save_image("region_no_border.png", crop_only=True)  # Don't add a border
region.save_image("region_high_res.png", resolution=300)  # Higher resolution

# Clear all highlights
page.clear_highlights()

# Highlight all elements on a page with one command
page.highlight_all()  # Highlights all element types with different colors
page.highlight_all(include_types=['text', 'line'])  # Only specific types

# Analyzing text styles
styles = page.analyze_text_styles()  # Returns dict of labeled element collections
for label, elements in styles.items():
    print(f"{label}: {len(elements)} elements")

# Highlight the text styles
page.highlight_text_styles()
page.save_image("text_styles.png", labels=True)

# Combine with highlight_all
page.highlight_all(include_text_styles=True)

# Display attributes directly on highlighted elements (must be explicitly requested)
regions = page.analyze_layout()  # Run layout detection
for region in regions:
    # Show both confidence score and type directly on the highlight
    region.highlight(include_attrs=['confidence', 'region_type'])
    
# Group regions by confidence level for better visualization
high_conf = [r for r in regions if r.confidence >= 0.8]
med_conf = [r for r in regions if 0.5 <= r.confidence < 0.8]
low_conf = [r for r in regions if r.confidence < 0.5]

from natural_pdf.elements.collections import ElementCollection
ElementCollection(high_conf).highlight(
    label="High Confidence",
    color=(0, 1, 0, 0.3),  # Green
    include_attrs=['region_type', 'confidence']
)
ElementCollection(med_conf).highlight(
    label="Medium Confidence",
    color=(1, 1, 0, 0.3),  # Yellow
    include_attrs=['region_type', 'confidence']
)
ElementCollection(low_conf).highlight(
    label="Low Confidence",
    color=(1, 0, 0, 0.3),  # Red
    include_attrs=['region_type', 'confidence']
)
```

## Implementation Notes

### Highlighting System
The highlighting system uses:
- `HighlightManager` class to track and manage highlights within a page
- Color cycling for multiple highlights via `get_next_highlight_color()`
- Labels for grouping related highlights (all elements with same label get same color)
- Legend generation for visualizing different highlight groups

### Text Style Analysis
The text style analyzer works by:
- Extracting style properties (font, size, weight, style) from text elements
- Grouping elements with identical style properties
- Assigning "Text Style N" labels based on the order styles are encountered
- Using a tuple of properties as a hashable key for grouping

### Debugging Features
- Visual highlighting system for inspecting elements, regions, and text styles
- `highlight_all()` for quickly visualizing all elements on a page
- `highlight_text_styles()` for visualizing text grouped by visual style properties
- Color-coded legends with labels for identifying different element types
- `save()` and `show()` methods for exporting and displaying visualizations

### Areas for Improvement
- Word grouping algorithm could be enhanced for better handling of spacing variations
- Text style analysis could incorporate more properties (line spacing, alignment, etc.)
- Reading order algorithm could be improved for complex layouts
- Region selection could handle non-rectangular regions for irregular layouts

### Exclusion Zones
The library provides a flexible system for excluding headers, footers, or other regions from text extraction:

```python
# Page-level exclusion
page.add_exclusion(page.find('text:contains("Page")').above())
page.add_exclusion(page.find_all('line')[-1].below())

# Extract text with exclusions applied
text = page.extract_text()  # Apply exclusions by default
text = page.extract_text(apply_exclusions=False)  # Ignore exclusions if needed

# PDF-level exclusion with lambdas for dynamic determination per page
pdf.add_exclusion(
    lambda page: page.find('text:contains("Header")').above() if page.find('text:contains("Header")') else None,
    label="headers"
)

pdf.add_exclusion(
    lambda page: page.find_all('line')[-1].below() if page.find_all('line') else None,
    label="footers"
)

# Extract content from multiple pages with consistent exclusions
text = pdf.extract_text()
```

The exclusion system supports:
- Using existing region objects directly (`header.above()`, `footer.below()`)
- Lambda functions for page-specific logic that gets applied across the document
- Optional labeling for organizing exclusions by purpose
- Control over whether exclusions are applied via the `apply_exclusions` parameter

## Implementation Notes

### Logging System

The library implements a structured logging system using Python's standard `logging` module:

#### Logging Configuration

```python
# Import the configuration utility
from natural_pdf import configure_logging
import logging

# Basic console logging at INFO level
configure_logging(level=logging.INFO)

# Advanced file logging with custom format
import logging
handler = logging.FileHandler("natural_pdf.log")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
configure_logging(level=logging.DEBUG, handler=handler)
```

#### Logger Hierarchy

The library uses a hierarchical logger structure mirroring the module structure:

- `natural_pdf` - Root logger for the library
  - `natural_pdf.core` - Core PDF and page operations
    - `natural_pdf.core.pdf` - PDF object operations
  - `natural_pdf.analyzers` - Layout analysis modules
    - `natural_pdf.analyzers.layout` - Layout detector operations
    - `natural_pdf.analyzers.layout.paddle` - PaddleOCR layout detector
  - `natural_pdf.ocr` - OCR engine operations
    - `natural_pdf.ocr.engine` - Base engine operations
    - `natural_pdf.ocr.EasyOCREngine` - EasyOCR specific operations
    - `natural_pdf.ocr.PaddleOCREngine` - PaddleOCR specific operations

#### Log Levels

- `DEBUG` - Detailed information, typically for diagnosis
- `INFO` - Confirmation that things are working as expected
- `WARNING` - Indication that something unexpected happened
- `ERROR` - Due to a more serious problem, some functionality couldn't be performed
- `CRITICAL` - A very serious error, indicating the program may be unable to continue

#### Controlling Component Verbosity

Many components accept a `verbose` parameter that enables more detailed DEBUG-level logging for just that component:

```python
# Enable verbose logging for layout detection
regions = page.analyze_layout(
    model="paddle", 
    model_params={"verbose": True}
)

# Regular log level for other components
text = page.extract_text()
```

#### Command-line Example Scripts

Example scripts like `paddle_layout_example.py` include command-line options for logging:

```bash
# Run with verbose output
python examples/paddle_layout_example.py --verbose

# Set specific log level
python examples/paddle_layout_example.py --log-level=DEBUG
```

#### Implementation Details

- Library uses Python's best practice of adding a `NullHandler` by default
- Each module sets up its own logger using `logging.getLogger("natural_pdf.module_name")`
- The central `configure_logging()` function handles logger configuration
- Components with `verbose` parameter temporarily adjust their logger to DEBUG level

### Font-Aware Text Extraction

The library supports font-aware text extraction to preserve formatting and style during extraction:

1. **Font Attribute Grouping**:
   - Characters are grouped into words based on specified font attributes
   - Default grouping is by `fontname` and `size`
   - Can be customized when creating PDF objects:
   ```python
   # Default: Group by font name and size
   pdf = PDF("document.pdf")
   
   # Spatial only: Group only by position
   pdf = PDF("document.pdf", font_attrs=[])
   
   # Custom: Group by font name, size, and color
   pdf = PDF("document.pdf", font_attrs=['fontname', 'size', 'non_stroking_color'])
   ```

2. **Font Information Access**:
   - All text elements provide font information in their string representation
   - Enhanced `bold` and `italic` detection using multiple indicators from PDF spec
   - `font_info()` method provides detailed font properties for debugging
   ```python
   element = page.find('text')
   print(element)  # Shows font name, size, style, etc.
   print(element.font_info())  # Shows all available font properties
   ```

3. **Implementation Details**:
   - Word grouping algorithm breaks words when font attributes change
   - Characters with different font properties will be treated as separate words
   - Improves preservation of textual styles in extracted content
   - Font attributes are used in addition to spatial grouping logic

### Important Code Conventions

#### Property vs Method Access
- Use properties for read-only attributes that require no parameters (e.g., `element.first`, not `element.first()`)
- Methods with similar names should require parameters (e.g., `element.extract_text(keep_blank_chars=True)`)
- ElementCollection uses `.first` and `.last` as properties, not methods

#### Parameter Consistency
- All extraction methods accept `keep_blank_chars=True` by default
- The new `apply_exclusions=True` parameter follows the same pattern
- When extending methods in subclasses, maintain the same parameter signatures

### Security Considerations

The selector parser uses secure parsing methods:
- `ast.literal_eval()` for parsing literals (numbers, tuples, etc.)
- Specialized color parsing with the `colour` library
- No `eval()` functions are used, eliminating potential security risks

#### Color Name Support

The selector parser supports multiple ways to specify colors:
- RGB tuples: `[color~=(1,0,0)]` 
- Color names: `[color~=red]`
- Hex colors: `[color~=#ff0000]`

This works for any color attribute like `color`, `fill`, `stroke`, etc. The color parsing is handled by the `colour` library, which supports:
- 147 named colors from the CSS3/SVG specification
- Hex colors (3 or 6 digits with or without # prefix)
- RGB, HSL, and other color formats

All colors are converted to RGB tuples internally for matching, with a tolerance to account for minor variations.

### Exclusion System Design

The exclusion system allows for the exclusion of regions (like headers and footers) from text extraction:

1. **Page-level exclusions**:
   - `page.add_exclusion(region)` - Add a region to exclude
   - Regions can be created using `.above()`, `.below()`, etc.
   - Example: `page.add_exclusion(header.above())`

2. **PDF-level exclusions with lambdas**:
   - `pdf.add_exclusion(lambda page: page.find(selector).above())`
   - Dynamically applied to each page, with fallback for missing elements
   - Example: `pdf.add_exclusion(lambda page: page.find_all('line')[-1].below() if page.find_all('line') else None)`

3. **Element filtering**:
   - Elements in excluded regions are filtered out during text extraction
   - Uses `_is_element_in_region()` to determine element inclusion
   - Element's center point must be within region boundaries

4. **Exclusions in searching and extraction**:
   - `extract_text(apply_exclusions=True)` - Apply exclusions during text extraction (default)
   - `extract_text(apply_exclusions=False)` - Ignore exclusions for a single extraction
   - `find(selector, apply_exclusions=True)` - Exclude elements in exclusion regions from search results
   - `find_all(selector, apply_exclusions=True)` - Filter out excluded elements from search results
   - Parameter cascades through object hierarchy (PDF → Page → Region → Element)
   - Consistency: All methods that work with elements respect exclusions by default

5. **Smart Region Exclusion Handling** (March 2025 update):
   - Optimized for efficiency and better text extraction
   - Three distinct handling strategies:
     - **No intersection**: If region doesn't intersect with any exclusion, exclusions are ignored entirely
     - **Header/footer exclusions**: For rectangular regions intersecting with full-width exclusions (e.g., headers/footers), uses efficient cropping approach
     - **Complex exclusions**: For regions with partial/complex exclusions, uses element filtering with warning
   - Example for intersecting with header but not footer:
     ```python
     # Add header exclusion
     page.add_exclusion(page.find('text:contains("HEADER")').above())
     
     # Create a region in the middle of the page
     middle_region = page.create_region(50, page.height * 0.25, page.width - 50, page.height * 0.75)
     
     # Extract text - will automatically use cropping to exclude the header
     middle_text = middle_region.extract_text()
     ```

### ElementCollection Handling

The ElementCollection class provides batch operations for multiple elements:
- Add the `exclude_regions()` method to filter elements outside exclusion zones
- Handle potentially missing parameters with `try/except` in methods to ensure backward compatibility:
```python
try:
    element.extract_text(keep_blank_chars=True, apply_exclusions=True)
except TypeError:
    element.extract_text(keep_blank_chars=True)  # Fallback if apply_exclusions not supported
```
- Remember `.first` and `.last` are properties returning the first/last element

### Highlighting System

For Region highlighting:
- Directly use the page's `_highlight_mgr`
- Add `highlight()`, `show()`, and `save()` methods with consistent signatures
- Ensure these methods match the Element class methods

## Recent Updates

### API Simplification (March 2025)
The section extraction API has been simplified to focus on the most common use cases:

1. **Simplified Parameters**:
   - Removed legacy parameters (`start_selector`, `end_selector`, `separator_selector`, `separator_elements`)
   - Standardized on `start_elements` and `end_elements` as the primary parameters
   - Simplified implementation across `Page`, `Region`, and `PageCollection` classes

2. **Consistent Parameter Interface**:
   - All implementations now accept the same core parameters:
     - `start_elements`: Elements that mark the start of sections
     - `end_elements`: Elements that mark the end of sections
     - `boundary_inclusion`: How to include boundary elements ('start', 'end', 'both', 'none')
   - `PageCollection` has the additional `new_section_on_page_break` parameter for multi-page operations

3. **Legacy Support Removed**:
   - Support for separator-based sectioning has been removed
   - All separator-based parameters have been removed

4. **Usage Recommendations**:
   - Use `start_elements` for section starts (required)
   - Optionally provide `end_elements` for section ends
   - Control boundary inclusion with the `boundary_inclusion` parameter

## Document Layout Analysis

The library supports document layout analysis using machine learning models to detect and work with different types of regions in the document:

```python
# Analyze document layout
page.analyze_layout(confidence=0.2)  # Uses Docling model by default
# The detected regions are stored in page.detected_layout_regions

# Find specific region types using selectors
titles = page.find_all('region[type=title]')
tables = page.find_all('region[type=table]')
paragraphs = page.find_all('region[type=plain-text]')

# Filter by confidence score
high_conf_titles = page.find_all('region[type=title][confidence>=0.8]')

# Extract content from specific regions
for table in tables:
    table_text = table.extract_text()
    print(f"Table content: {table_text}")
    
# Visualize detected regions
page.highlight_layout()  # Dedicated method for layout regions
page.highlight_all(include_layout_regions=True)  # Include with all elements, showing confidence scores
page.to_image(path="layout_analysis.png", show_labels=True)

# Control confidence threshold for layout regions
page.highlight_all(include_layout_regions=True, layout_confidence=0.5)  # Only show high confidence regions
page.highlight_all(include_layout_regions=True, layout_confidence=True)  # Show all regions regardless of confidence

# Display attributes directly on highlights
page.find_all('region[type=table]').highlight(include_attrs=['confidence'])  # Show confidence
page.find_all('region[type=title]').highlight(include_attrs=['confidence', 'model'])  # Show multiple attributes
page.find_all('text[source=ocr]').highlight(include_attrs=['confidence', 'text'])  # Show OCR confidence and text

# Work with specific regions
if tables:
    table = tables[0]
    # Find text elements within the table
    table_text_elements = table.find_all('text')
    # Expand the region slightly for better coverage
    expanded_table = table.expand(width_factor=1.1, height_factor=1.1)
    
# Method chaining for more concise code
page.analyze_layout(confidence=0.3)\
    .highlight_layout()\
    .to_image(path="layout_analysis.png", show_labels=True)
    
# Chain with specific filters
page.clear_highlights()\
    .analyze_layout(engine="tatr", confidence=0.4)\
    .find_all('region[type=table]')\
    .highlight(label="Tables", color=(1, 0, 0, 0.3))
```

### Supported Layout Models

The library supports multiple layout detection models:

#### YOLO Layout Detection
The default YOLO model supports these region types:
- `title`: Document titles and headings
- `plain-text`: Regular paragraph text
- `table`: Tabular data
- `figure`: Images and figures
- `figure_caption`: Captions for figures
- `table_caption`: Captions for tables
- `table_footnote`: Footnotes for tables
- `isolate_formula`: Mathematical formulas
- `formula_caption`: Captions for formulas
- `abandon`: Abandoned or unstructured text

#### Table Transformer (TATR)
The Table Transformer model provides detailed table structure analysis:
- `table`: The table as a whole
- `table row`: Individual rows in the table
- `table column`: Individual columns in the table
- `table column header`: Column headers in the table

#### Docling
Docling provides hierarchical document understanding with semantic structure recognition:
- Automatically detects document structure with semantic labels (section_header, text, table, etc.)
- Preserves parent-child relationships between document elements
- Performs OCR and text extraction in one pass
- Provides table structure and content extraction
- Supports hierarchical navigation through document elements
- Common types include: `section-header`, `text`, `figure`, `table`, etc.
- Automatically supports all label types provided by the Docling library

Installation: `pip install docling`

### Region Type Selector Format

Note that space-separated region types like "plain text" are converted to hyphenated format in selectors:
```python
# For "plain text" region type:
page.find_all('region[type=plain-text]')

# For "figure_caption" region type:
page.find_all('region[type=figure_caption]')
```

### Element Source Attributes

Text elements have source attributes to identify their origin:
```python
# Find original document text elements
native_text = page.find_all('text[source=native]')

# Find OCR-extracted text elements
ocr_text = page.find_all('text[source=ocr]')

# Find elements from a specific model
docling_elements = page.find_all('region[model=docling]')
yolo_elements = page.find_all('region[model=yolo]')

# Combine attributes in selectors
important_native = page.find_all('text[source=native][size>=12]')
```

### Customization Options

You can customize the layout analysis:
```python
# Use a custom model
regions = page.analyze_layout(
    model_path="path/to/custom_model.pt",
    confidence=0.3,
    device="cuda:0"  # Use GPU for faster processing
)

# Extract only specific region types
page.analyze_layout(classes=["title", "table", "figure"])

# Exclude specific region types
page.analyze_layout(exclude_classes=["table", "figure"])

# Set different confidence thresholds for highlighting
page.highlight_layout(confidence=0.5)
```

### Hierarchical Document Analysis

For documents with hierarchical structure, you can use Docling model and the hierarchical navigation methods:

```python
# Run Docling analysis 
page.analyze_layout(
    model="docling",
    confidence=0.3,  # Not used by Docling but kept for API consistency
    model_params={
        "verbose": True  # Additional parameters are passed to DocumentConverter
    }
)

# Find regions by type
headers = page.find_all('section-header')
paragraphs = page.find_all('text')

# Navigate hierarchy
if headers:
    header = headers[0]
    
    # Get direct children of this header
    children = header.get_children()
    print(f"Header has {len(children)} direct children")
    
    # Get all descendants recursively
    descendants = header.get_descendants()
    print(f"Header has {len(descendants)} total descendants")
    
    # Find specific types of children
    text_children = header.get_children('text')
    print(f"Header has {len(text_children)} direct text children")
    
    # Recursive search within a section
    section_figures = header.find_all('figure', recursive=True)
    print(f"Section contains {len(section_figures)} figures")
    
    # Extract text from the entire section hierarchy
    section_text = header.extract_text()
```

### Table Structure Analysis

For detailed table structure analysis, use the Table Transformer model:

```python
# First run general layout analysis, excluding tables
general_regions = page.analyze_layout(
    model="yolo",
    exclude_classes=["table", "table_caption", "table_footnote"]
)

# Then run table structure detection
table_regions = page.analyze_layout(
    model="tatr",  # Table Transformer model
    confidence=0.5,
    existing="append"  # Add to existing regions rather than replacing
)

# Find tables and their components
tables = page.find_all('region[type=table]')
rows = page.find_all('region[type=table-row]')
columns = page.find_all('region[type=table-column]')
headers = page.find_all('region[type=table-column-header]')

# Filter regions by model
yolo_regions = page.find_all('region[model=yolo]')
tatr_regions = page.find_all('region[model=tatr]')

# Combine multiple attributes in selectors
table_headers = page.find_all('region[type=table-column-header][model=tatr][confidence>=0.7]')

# Process a specific table
if tables:
    table = tables[0]
    # Extract text from the table
    table_text = table.extract_text()
    
    # Find text elements within the table
    table_content = table.find_all('text')
    
    # Highlight the table structure
    table.highlight(color=(1, 0, 0, 0.3))
    
    # Highlight all regions from a specific model
    page.find_all('region[model=tatr]').highlight(label="Table Structure")
```

## Table Extraction

The library provides multiple methods for table extraction:

```python
# Simple extraction with pdfplumber
table_data = page.extract_table()

# Using Table Transformer (TATR) detection for more accurate extraction
# First detect tables and structure
page.analyze_layout(engine="tatr")

# Then find the tables
tables = page.find_all('region[type=table]')
table = tables[0]

# Extract table data using auto-detected method (TATR for TATR regions)
data = table.extract_table()

# Or explicitly specify the method to use
data_tatr = table.extract_table(method='tatr')    # Uses detected table structure
data_plumber = table.extract_table(method='plumber')  # Uses pdfplumber's algorithm

# Work with table components directly
rows = page.find_all('region[type=table-row][model=tatr]')
columns = page.find_all('region[type=table-column][model=tatr]')
headers = page.find_all('region[type=table-column-header][model=tatr]')

# Extract a specific cell at row/column intersection
from natural_pdf.elements.region import Region
cell_bbox = (column.x0, row.top, column.x1, row.bottom)
cell = Region(page, cell_bbox)
cell_text = cell.extract_text()

# Create a dictionary representation
table_dict = []
header_texts = [header.extract_text().strip() for header in headers]

for row in rows:
    row_dict = {}
    for i, col in enumerate(columns):
        if i < len(header_texts):
            # Create cell
            cell_bbox = (col.x0, row.top, col.x1, row.bottom)
            cell = Region(page, cell_bbox)
            
            # Extract text
            row_dict[header_texts[i]] = cell.extract_text().strip()
    
    table_dict.append(row_dict)
```

## Element Navigation

The library provides methods for navigating between elements in reading order:

```python
# Find the next element in reading order
next_element = element.next()  # Next element regardless of type
next_text = element.next('text')  # Next text element
next_bold = element.next('text:bold', limit=20)  # Next bold text within 20 elements

# Find the previous element in reading order
prev_element = element.prev()  # Previous element regardless of type
prev_heading = element.prev('text[size>=12]')  # Previous large text
prev_rect = element.prev('rect', limit=5)  # Previous rectangle within 5 elements

# Find the nearest element by Euclidean distance
nearest_element = element.nearest('rect')  # Nearest rectangle
nearest_with_limit = element.nearest('text:contains("Table")', max_distance=100)  # Within 100 points
```

All navigation methods respect exclusion zones by default and can be customized with the same parameters used in `find()` and `find_all()`.

## OCR Integration

The library supports OCR (Optical Character Recognition) for extracting text from scanned documents or image-based PDFs. PaddleOCR is the default OCR engine with better performance for most languages, especially Asian languages.

### Basic OCR Usage

```python
# Enable OCR at PDF initialization with auto mode (only applies when needed)
pdf = PDF("scanned_document.pdf", ocr={
    "enabled": "auto",  # "auto", True, or False
    "languages": ["en"],
    "min_confidence": 0.5  # Minimum confidence threshold
})

# Extract text with OCR automatically applied when needed
text = page.extract_text()  # OCR auto-applied if little/no text found

# Force OCR on a page regardless of existing text
ocr_text = page.extract_text(ocr=True)

# Apply OCR explicitly and get OCR elements
ocr_elements = page.apply_ocr()
print(f"Found {len(ocr_elements)} OCR text elements")

# Access OCR confidence scores and other metadata
for elem in ocr_elements:
    print(f"Text: '{elem.text}', Confidence: {elem.confidence:.2f}")
```

### OCR Engine Selection

The library supports multiple OCR engines to better handle different languages and document types:

```python
# Use PaddleOCR instead of the default EasyOCR
pdf = PDF("document.pdf", ocr_engine="paddleocr")

# Or create an engine with custom settings
from natural_pdf.ocr import PaddleOCREngine
engine = PaddleOCREngine(models_dir="/path/to/models")
pdf = PDF("document.pdf", ocr_engine=engine)

# Combined with regular OCR configuration
pdf = PDF("document.pdf", 
          ocr_engine="paddleocr",
          ocr={
              "enabled": True,
              "languages": ["zh", "en"],  # Multiple languages
              "min_confidence": 0.3
          })
```

Different engines have different strengths:
- **EasyOCR**: Better for European languages, simpler setup
- **PaddleOCR**: Excellent for Asian languages, especially Chinese

### PaddleOCR-Specific Parameters

When using PaddleOCR, you can specify these engine-specific parameters:

```python
engine = PaddleOCREngine(
    use_angle_cls=False,   # Use text direction classification
    lang="en",             # Language code (mapped from ISO code)
    det=True,              # Use text detection
    rec=True,              # Use text recognition
    cls=False,             # Use text direction classification
    det_model_dir=None,    # Custom detection model directory
    rec_model_dir=None,    # Custom recognition model directory
    det_limit_side_len=960,# Limit of max image size
    det_db_thresh=0.3,     # Binarization threshold
    det_db_box_thresh=0.5, # Box confidence threshold
    rec_batch_num=6,       # Recognition batch size
    rec_algorithm="CRNN"   # Recognition algorithm
)
```

### Engine Comparison

You can compare OCR engines for specific documents to find which one works best:

```python
# Compare both engines with the same configuration
from natural_pdf.ocr import EasyOCREngine, PaddleOCREngine

# EasyOCR version
easy_pdf = PDF("document.pdf", 
              ocr_engine=EasyOCREngine(),
              ocr={"enabled": True, "languages": ["en"]})
easy_text = easy_pdf.pages[0].extract_text()

# PaddleOCR version
paddle_pdf = PDF("document.pdf", 
                ocr_engine=PaddleOCREngine(),
                ocr={"enabled": True, "languages": ["en"]})
paddle_text = paddle_pdf.pages[0].extract_text()

# Compare results
print(f"EasyOCR length: {len(easy_text)}")
print(f"PaddleOCR length: {len(paddle_text)}")
```

### Working with OCR Elements

```python
# Get OCR text elements without modifying the page
ocr_elements = page.extract_ocr_elements()

# Filter OCR elements by confidence
high_confidence = page.find_all('text[source=ocr][confidence>=0.8]')

# Highlight OCR elements with confidence scores
for elem in ocr_elements:
    elem.highlight(label=f"OCR ({elem.confidence:.2f})")

# Apply OCR to just a specific region
region = page.create_region(100, 100, 400, 200)
region_ocr = region.apply_ocr()

# Use selectors to find OCR elements containing specific text
matching_ocr = page.find_all('text[source=ocr]:contains("invoice")')
```

### OCR Configuration

The OCR system supports multiple configuration formats:

```python
# Simple flag
pdf = PDF("document.pdf", ocr=True)  # Enable with defaults

# Auto mode (only when needed)
pdf = PDF("document.pdf", ocr="auto")

# Language list
pdf = PDF("document.pdf", ocr=["en", "fr"])  # English and French

# Detailed configuration
pdf = PDF("document.pdf", ocr={
    "enabled": True,
    "engine": "easyocr",
    "languages": ["en"],
    "min_confidence": 0.6,
    # Additional EasyOCR parameters
    "paragraph": False,
    "detail": 1
})
```

#### OCR Parameters

You can pass EasyOCR parameters directly in the configuration object:

```python
pdf = PDF("document.pdf", ocr={
    "enabled": True,
    "languages": ["en"],
    # Text detection parameters
    "text_threshold": 0.1,      # Lower threshold to detect more text (default: 0.7)
    "low_text": 0.3,            # Text low-bound score (default: 0.4)
    "link_threshold": 0.3,      # Link confidence threshold (default: 0.4)
    "canvas_size": 2560,        # Maximum image size
    "mag_ratio": 1.5,           # Image magnification ratio
    # Text recognition parameters
    "decoder": "greedy",        # Options: 'greedy', 'beamsearch', 'wordbeamsearch'
    "batch_size": 4,            # Larger batches use more memory but are faster
    "contrast_ths": 0.05,       # Lower threshold to handle low contrast text
    "adjust_contrast": 0.5      # Target contrast for low contrast text
})
```

These parameters can also be specified directly when calling OCR methods:

```python
# Apply OCR with custom parameters
ocr_elements = page.apply_ocr(
    text_threshold=0.1,
    link_threshold=0.1,
    mag_ratio=1.5,
    batch_size=4
)

# Extract text with OCR parameters
text = page.extract_text(ocr={
    "enabled": True,
    "text_threshold": 0.1,
    "contrast_ths": 0.05
})
```
```

OCR configuration can be overridden at extraction time:

```python
# Override OCR settings for this extraction
text = page.extract_text(ocr={
    "languages": ["fr"],  # Switch to French
    "min_confidence": 0.4,  # Lower threshold
    "detection_params": {
        "text_threshold": 0.1  # More sensitive text detection
    }
})
```

## Future Enhancements
- Further enhance cross-page operations for content spanning multiple pages
- Improve spatial relationship detection for complex layouts
- Expand support for different document layout models