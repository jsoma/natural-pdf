# API Reference

This section provides detailed documentation for all the classes and methods in Natural PDF.

## Core Classes

### PDF Class

The main entry point for working with PDFs.

```python
class PDF:
    """
    The main entry point for working with PDFs.
    
    Parameters:
        path (str): Path to the PDF file.
        password (str, optional): Password for encrypted PDFs. Default: None
        reading_order (bool, optional): Sort elements in reading order. Default: True
        keep_spaces (bool, optional): Keep spaces in word elements. Default: True
        font_attrs (list, optional): Font attributes to use for text grouping. 
                                    Default: ['fontname', 'size']
        ocr (bool/dict/str, optional): OCR configuration. Default: False
        ocr_engine (str/Engine, optional): OCR engine to use. Default: "easyocr"
    """
```

**Main Methods**

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `pages` | Access pages in the document | N/A (property) | `PageCollection` |
| `extract_text(keep_blank_chars=True, apply_exclusions=True)` | Extract text from all pages | `keep_blank_chars`: Whether to keep blank characters<br>`apply_exclusions`: Whether to apply exclusion zones | `str`: Extracted text |
| `find(selector, case=True, regex=False, apply_exclusions=True)` | Find first element matching selector across all pages | `selector`: CSS-like selector string<br>`case`: Case-sensitive search<br>`regex`: Use regex for :contains()<br>`apply_exclusions`: Whether to apply exclusion zones | `Element` or `None` |
| `find_all(selector, case=True, regex=False, apply_exclusions=True)` | Find all elements matching selector across all pages | `selector`: CSS-like selector string<br>`case`: Case-sensitive search<br>`regex`: Use regex for :contains()<br>`apply_exclusions`: Whether to apply exclusion zones | `ElementCollection` |
| `add_exclusion(func, label=None)` | Add a document-wide exclusion zone | `func`: Function taking a page and returning region<br>`label`: Optional label for the exclusion | `None` |
| `get_sections(start_elements, end_elements=None, boundary_inclusion='start')` | Get sections across all pages | `start_elements`: Elements marking section starts<br>`end_elements`: Elements marking section ends<br>`boundary_inclusion`: How to include boundaries ('start', 'end', 'both', 'none') | `list[Region]` |
| `ask(question, min_confidence=0.0, model=None)` | Ask a question about the document content | `question`: Question to ask<br>`min_confidence`: Minimum confidence threshold<br>`model`: Optional model name or path | `dict`: Result with answer and metadata |

### Page Class

Represents a single page in a PDF document.

```python
class Page:
    """
    Represents a single page in a PDF document.
    
    Properties:
        page_number (int): 1-indexed page number
        page_index (int): 0-indexed page position
        width (float): Page width in points
        height (float): Page height in points
        pdf (PDF): Parent PDF object
    """
```

**Main Methods**

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `extract_text(keep_blank_chars=True, apply_exclusions=True, ocr=None)` | Extract text from the page | `keep_blank_chars`: Whether to keep blank characters<br>`apply_exclusions`: Whether to apply exclusion zones<br>`ocr`: Whether to force OCR | `str`: Extracted text |
| `find(selector, case=True, regex=False, apply_exclusions=True)` | Find the first element matching selector | `selector`: CSS-like selector string<br>`case`: Case-sensitive search<br>`regex`: Use regex for :contains()<br>`apply_exclusions`: Whether to apply exclusion zones | `Element` or `None` |
| `find_all(selector, case=True, regex=False, apply_exclusions=True)` | Find all elements matching selector | `selector`: CSS-like selector string<br>`case`: Case-sensitive search<br>`regex`: Use regex for :contains()<br>`apply_exclusions`: Whether to apply exclusion zones | `ElementCollection` |
| `create_region(x0, top, x1, bottom)` | Create a region at specific coordinates | `x0`: Left coordinate<br>`top`: Top coordinate<br>`x1`: Right coordinate<br>`bottom`: Bottom coordinate | `Region` |
| `highlight(elements, color=None, label=None)` | Highlight elements on the page | `elements`: Elements to highlight<br>`color`: RGBA color tuple<br>`label`: Label for the highlight | `Page` (self) |
| `highlight_all(include_types=None, include_text_styles=False, include_layout_regions=False)` | Highlight all elements on the page | `include_types`: Element types to include<br>`include_text_styles`: Whether to include text styles<br>`include_layout_regions`: Whether to include layout regions | `Page` (self) |
| `save_image(path, resolution=72, labels=True)` | Save an image of the page with highlights | `path`: Path to save image<br>`resolution`: Image resolution in DPI<br>`labels`: Whether to include labels | `None` |
| `to_image(resolution=72, labels=True)` | Get a PIL Image of the page with highlights | `resolution`: Image resolution in DPI<br>`labels`: Whether to include labels | `PIL.Image` |
| `analyze_text_styles()` | Group text by visual style properties | None | `dict`: Mapping of style name to elements |
| `analyze_layout(engine="yolo", confidence=0.2, existing="replace")` | Detect layout regions using ML models | `model`: Model to use ("yolo", "tatr")<br>`confidence`: Confidence threshold<br>`existing`: How to handle existing regions | `ElementCollection`: Detected regions |
| `add_exclusion(region, label=None)` | Add an exclusion zone to the page | `region`: Region to exclude<br>`label`: Optional label for the exclusion | `Region`: The exclusion region |
| `get_sections(start_elements, end_elements=None, boundary_inclusion='start')` | Get sections from the page | `start_elements`: Elements marking section starts<br>`end_elements`: Elements marking section ends<br>`boundary_inclusion`: How to include boundaries | `list[Region]` |
| `ask(question, min_confidence=0.0, model=None, debug=False)` | Ask a question about the page content | `question`: Question to ask<br>`min_confidence`: Minimum confidence threshold<br>`model`: Optional model name or path<br>`debug`: Whether to save debug files | `dict`: Result with answer and metadata |
| `apply_ocr(languages=None, min_confidence=0.0, **kwargs)` | Apply OCR to the page | `languages`: Languages to use<br>`min_confidence`: Minimum confidence threshold<br>`**kwargs`: Additional OCR engine parameters | `ElementCollection`: OCR text elements |

### Region Class

Represents a rectangular area on a page.

```python
class Region:
    """
    Represents a rectangular area on a page.
    
    Properties:
        x0 (float): Left coordinate
        top (float): Top coordinate
        x1 (float): Right coordinate
        bottom (float): Bottom coordinate
        width (float): Width of the region
        height (float): Height of the region
        page (Page): Parent page object
    """
```

**Main Methods**

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `extract_text(keep_blank_chars=True, apply_exclusions=True, ocr=None)` | Extract text from the region | `keep_blank_chars`: Whether to keep blank characters<br>`apply_exclusions`: Whether to apply exclusion zones<br>`ocr`: Whether to force OCR | `str`: Extracted text |
| `find(selector, case=True, regex=False, apply_exclusions=True)` | Find the first element matching selector within the region | `selector`: CSS-like selector string<br>`case`: Case-sensitive search<br>`regex`: Use regex for :contains()<br>`apply_exclusions`: Whether to apply exclusion zones | `Element` or `None` |
| `find_all(selector, case=True, regex=False, apply_exclusions=True)` | Find all elements matching selector within the region | `selector`: CSS-like selector string<br>`case`: Case-sensitive search<br>`regex`: Use regex for :contains()<br>`apply_exclusions`: Whether to apply exclusion zones | `ElementCollection` |
| `expand(left=0, top=0, right=0, bottom=0, width_factor=1.0, height_factor=1.0)` | Expand the region in specified directions | `left/top/right/bottom`: Points to expand in each direction<br>`width_factor/height_factor`: Scale width/height by this factor | `Region`: Expanded region |
| `highlight(color=None, label=None, include_attrs=None)` | Highlight the region | `color`: RGBA color tuple<br>`label`: Label for the highlight<br>`include_attrs`: Region attributes to display | `Region` (self) |
| `to_image(resolution=72, crop_only=False)` | Get a PIL Image of just the region | `resolution`: Image resolution in DPI<br>`crop_only`: Whether to exclude border | `PIL.Image` |
| `save_image(path, resolution=72, crop_only=False)` | Save an image of just the region | `path`: Path to save image<br>`resolution`: Image resolution in DPI<br>`crop_only`: Whether to exclude border | `None` |
| `get_sections(start_elements, end_elements=None, boundary_inclusion='start')` | Get sections within the region | `start_elements`: Elements marking section starts<br>`end_elements`: Elements marking section ends<br>`boundary_inclusion`: How to include boundaries | `list[Region]` |
| `ask(question, min_confidence=0.0, model=None, debug=False)` | Ask a question about the region content | `question`: Question to ask<br>`min_confidence`: Minimum confidence threshold<br>`model`: Optional model name or path<br>`debug`: Whether to save debug files | `dict`: Result with answer and metadata |
| `extract_table(method=None, table_settings=None, use_ocr=False)` | Extract table data from the region | `method`: Extraction method ("plumber", "tatr")<br>`table_settings`: Custom settings for extraction<br>`use_ocr`: Whether to use OCR text | `list`: Table data as rows and columns |
| `intersects(other)` | Check if this region intersects with another | `other`: Another region | `bool`: True if regions intersect |
| `contains(x, y)` | Check if a point is within the region | `x`: X coordinate<br>`y`: Y coordinate | `bool`: True if point is in region |

## Element Types

### Element Base Class

The base class for all PDF elements.

```python
class Element:
    """
    Base class for all PDF elements.
    
    Properties:
        x0 (float): Left coordinate
        top (float): Top coordinate
        x1 (float): Right coordinate
        bottom (float): Bottom coordinate
        width (float): Width of the element
        height (float): Height of the element
        page (Page): Parent page object
    """
```

**Main Methods**

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `above(height=None, full_width=True, until=None, include_until=True)` | Create a region above the element | `height`: Height of region<br>`full_width`: Whether to span page width<br>`until`: Selector for boundary<br>`include_until`: Whether to include boundary | `Region` |
| `below(height=None, full_width=True, until=None, include_until=True)` | Create a region below the element | `height`: Height of region<br>`full_width`: Whether to span page width<br>`until`: Selector for boundary<br>`include_until`: Whether to include boundary | `Region` |
| `select_until(selector, include_endpoint=True, full_width=True)` | Create a region from this element to another | `selector`: Selector for endpoint<br>`include_endpoint`: Whether to include endpoint<br>`full_width`: Whether to span page width | `Region` |
| `highlight(color=None, label=None, include_attrs=None)` | Highlight this element | `color`: RGBA color tuple<br>`label`: Label for the highlight<br>`include_attrs`: Element attributes to display | `Element` (self) |
| `extract_text(keep_blank_chars=True, apply_exclusions=True)` | Extract text from this element | `keep_blank_chars`: Whether to keep blank characters<br>`apply_exclusions`: Whether to apply exclusion zones | `str`: Extracted text |
| `next(selector=None, limit=None, apply_exclusions=True)` | Get the next element in reading order | `selector`: Optional selector to filter<br>`limit`: How many elements to search<br>`apply_exclusions`: Whether to apply exclusion zones | `Element` or `None` |
| `prev(selector=None, limit=None, apply_exclusions=True)` | Get the previous element in reading order | `selector`: Optional selector to filter<br>`limit`: How many elements to search<br>`apply_exclusions`: Whether to apply exclusion zones | `Element` or `None` |
| `nearest(selector, max_distance=None, apply_exclusions=True)` | Get the nearest element matching selector | `selector`: Selector for elements<br>`max_distance`: Maximum distance in points<br>`apply_exclusions`: Whether to apply exclusion zones | `Element` or `None` |

### TextElement

Represents text elements in the PDF.

```python
class TextElement(Element):
    """
    Represents text elements in the PDF.
    
    Additional Properties:
        text (str): The text content
        fontname (str): The font name
        size (float): The font size
        bold (bool): Whether the text is bold
        italic (bool): Whether the text is italic
        color (tuple): The text color as RGB tuple
        confidence (float): OCR confidence (for OCR text)
        source (str): 'pdf' or 'ocr'
    """
```

**Main Properties**

| Property | Type | Description |
|----------|------|-------------|
| `text` | `str` | The text content |
| `fontname` | `str` | The font name |
| `size` | `float` | The font size |
| `bold` | `bool` | Whether the text is bold |
| `italic` | `bool` | Whether the text is italic |
| `color` | `tuple` | The text color as RGB tuple |
| `confidence` | `float` | OCR confidence (for OCR text) |
| `source` | `str` | 'pdf' or 'ocr' |
| `font_variant` | `str` | Font variant identifier (e.g., 'AAAAAB+') |

**Additional Methods**

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `font_info()` | Get detailed font information | None | `dict`: Font properties |

## Collections

### ElementCollection

A collection of elements with batch operations.

```python
class ElementCollection:
    """
    A collection of elements with batch operations.
    
    This class provides operations that can be applied to multiple elements at once.
    """
```

**Main Methods**

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `extract_text(keep_blank_chars=True, apply_exclusions=True)` | Extract text from all elements | `keep_blank_chars`: Whether to keep blank characters<br>`apply_exclusions`: Whether to apply exclusion zones | `str`: Extracted text |
| `filter(selector)` | Filter elements by selector | `selector`: CSS-like selector string | `ElementCollection` |
| `highlight(color=None, label=None, include_attrs=None)` | Highlight all elements | `color`: RGBA color tuple<br>`label`: Label for the highlight<br>`include_attrs`: Attributes to display | `ElementCollection` (self) |
| `first` | Get the first element in the collection | N/A (property) | `Element` or `None` |
| `last` | Get the last element in the collection | N/A (property) | `Element` or `None` |
| `highest()` | Get the highest element on the page | None | `Element` or `None` |
| `lowest()` | Get the lowest element on the page | None | `Element` or `None` |
| `leftmost()` | Get the leftmost element on the page | None | `Element` or `None` |
| `rightmost()` | Get the rightmost element on the page | None | `Element` or `None` |
| `__len__()` | Get the number of elements | None | `int` |
| `__getitem__(index)` | Get an element by index | `index`: Index or slice | `Element` or `ElementCollection` |

### PageCollection

A collection of pages with cross-page operations.

```python
class PageCollection:
    """
    A collection of pages with cross-page operations.
    
    This class provides operations that can be applied across multiple pages.
    """
```

**Main Methods**

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `extract_text(keep_blank_chars=True, apply_exclusions=True)` | Extract text from all pages | `keep_blank_chars`: Whether to keep blank characters<br>`apply_exclusions`: Whether to apply exclusion zones | `str`: Extracted text |
| `find(selector, case=True, regex=False, apply_exclusions=True)` | Find the first element matching selector across all pages | `selector`: CSS-like selector string<br>`case`: Case-sensitive search<br>`regex`: Use regex for :contains()<br>`apply_exclusions`: Whether to apply exclusion zones | `Element` or `None` |
| `find_all(selector, case=True, regex=False, apply_exclusions=True)` | Find all elements matching selector across all pages | `selector`: CSS-like selector string<br>`case`: Case-sensitive search<br>`regex`: Use regex for :contains()<br>`apply_exclusions`: Whether to apply exclusion zones | `ElementCollection` |
| `get_sections(start_elements, end_elements=None, boundary_inclusion='start', new_section_on_page_break=False)` | Get sections spanning multiple pages | `start_elements`: Elements marking section starts<br>`end_elements`: Elements marking section ends<br>`boundary_inclusion`: How to include boundaries<br>`new_section_on_page_break`: Whether to start new sections at page breaks | `list[Region]` |
| `__len__()` | Get the number of pages | None | `int` |
| `__getitem__(index)` | Get a page by index | `index`: Index or slice | `Page` or `PageCollection` |

## OCR Classes

### OCREngine

Base class for OCR engines.

```python
class OCREngine:
    """
    Base class for OCR engines.
    
    This class provides the interface for OCR engines.
    """
```

**Main Methods**

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `process_image(image, languages=None, min_confidence=0.0, **kwargs)` | Process an image with OCR | `image`: PIL Image<br>`languages`: Languages to use<br>`min_confidence`: Minimum confidence threshold | `list`: OCR results |

### EasyOCREngine

OCR engine using EasyOCR.

```python
class EasyOCREngine(OCREngine):
    """
    OCR engine using EasyOCR.
    
    Parameters:
        model_dir (str, optional): Directory for models. Default: None
    """
```

### PaddleOCREngine

OCR engine using PaddleOCR.

```python
class PaddleOCREngine(OCREngine):
    """
    OCR engine using PaddleOCR.
    
    Parameters:
        use_angle_cls (bool, optional): Use text direction classification. Default: False
        lang (str, optional): Language code. Default: "en"
        det (bool, optional): Use text detection. Default: True
        rec (bool, optional): Use text recognition. Default: True
        cls (bool, optional): Use text direction classification. Default: False
        det_model_dir (str, optional): Detection model directory. Default: None
        rec_model_dir (str, optional): Recognition model directory. Default: None
        verbose (bool, optional): Enable verbose output. Default: False
    """
```

## Document QA Classes

### DocumentQA

Class for document question answering.

```python
class DocumentQA:
    """
    Class for document question answering.
    
    Parameters:
        model (str, optional): Model name or path. Default: "microsoft/layoutlmv3-base"
        device (str, optional): Device to use. Default: "cpu"
        verbose (bool, optional): Enable verbose output. Default: False
    """
```

**Main Methods**

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `ask(question, image, word_boxes, min_confidence=0.0, max_answer_length=None, language=None)` | Ask a question about a document | `question`: Question to ask<br>`image`: Document image<br>`word_boxes`: Text positions<br>`min_confidence`: Minimum confidence threshold<br>`max_answer_length`: Maximum answer length<br>`language`: Language code | `dict`: Result with answer and metadata |

## Selector Syntax

Natural PDF uses a CSS-like selector syntax to find elements in PDFs.

### Basic Selectors

| Selector | Description | Example |
|----------|-------------|---------|
| `element_type` | Match elements of this type | `text`, `rect`, `line` |
| `[attribute=value]` | Match elements with this attribute value | `[fontname=Arial]`, `[size=12]` |
| `[attribute>=value]` | Match elements with attribute >= value | `[size>=12]` |
| `[attribute<=value]` | Match elements with attribute <= value | `[size<=10]` |
| `[attribute~=value]` | Match elements with attribute approximately equal | `[color~=red]`, `[color~=(1,0,0)]` |
| `[attribute*=value]` | Match elements with attribute containing value | `[fontname*=Arial]` |

### Pseudo-Classes

| Pseudo-Class | Description | Example |
|--------------|-------------|---------|
| `:contains("text")` | Match elements containing text | `text:contains("Summary")` |
| `:starts-with("text")` | Match elements starting with text | `text:starts-with("Summary")` |
| `:ends-with("text")` | Match elements ending with text | `text:ends-with("2023")` |
| `:bold` | Match bold text | `text:bold` |
| `:italic` | Match italic text | `text:italic` |

### Attribute Names

| Attribute | Element Types | Description |
|-----------|--------------|-------------|
| `fontname` | text | Font name |
| `size` | text | Font size |
| `color` | text, rect, line | Color |
| `width` | rect, line | Width |
| `height` | rect | Height |
| `confidence` | text (OCR) | OCR confidence score |
| `source` | text | Source ('pdf' or 'ocr') |
| `type` | region | Region type (e.g., 'table', 'title') |
| `model` | region | Layout model that detected the region |
| `font-variant` | text | Font variant identifier |

## Constants and Configuration

### Color Names

Natural PDF supports color names in selectors.

| Color Name | RGB Value | Example |
|------------|-----------|---------|
| `red` | (1, 0, 0) | `[color~=red]` |
| `green` | (0, 1, 0) | `[color~=green]` |
| `blue` | (0, 0, 1) | `[color~=blue]` |
| `black` | (0, 0, 0) | `[color~=black]` |
| `white` | (1, 1, 1) | `[color~=white]` |

### Region Types

Layout analysis models detect the following region types:

| Model | Region Types |
|-------|-------------|
| YOLO | `title`, `plain-text`, `table`, `figure`, `figure_caption`, `table_caption`, `table_footnote`, `isolate_formula`, `formula_caption`, `abandon` |
| TATR | `table`, `table-row`, `table-column`, `table-column-header` |