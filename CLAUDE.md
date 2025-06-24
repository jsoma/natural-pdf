# Natural PDF Library Analysis

## Library Overview
Natural PDF is a Python library for intelligent PDF document processing that combines traditional PDF parsing with modern AI capabilities. It provides a jQuery-like API for selecting and manipulating PDF elements with spatial awareness.

## Core Goals & Purpose
- **Intelligent PDF Processing**: Goes beyond simple text extraction to understand document structure and spatial relationships
- **AI-Enhanced Workflows**: Integrates OCR, document Q&A, classification, and LLM-based data extraction
- **Spatial Navigation**: Provides methods like `.below()`, `.above()`, `.left()` for intuitive element selection
- **Multi-format Support**: Handles both text-based PDFs and image-based (OCR-required) documents

## Key Use Cases & Workflows

### 1. Basic Text and Table Extraction
- Load PDFs from local files or URLs
- Extract text with layout preservation
- Find and extract tables automatically
- Use spatial selectors: `page.find('text:contains(Violations)').below()`

### 2. OCR Integration
- Multiple OCR engines supported: EasyOCR (default), Surya, PaddleOCR, DocTR
- Configurable resolution and detection modes
- OCR correction using LLMs
- Human-in-the-loop correction workflows with exportable packages

### 3. AI-Powered Data Extraction
- **Document Q&A**: Extractive question answering with confidence scores
- **Structured Data**: Extract specific fields with schema validation using Pydantic
- **LLM Integration**: OpenAI/Gemini compatible for advanced extraction
- **Classification**: Document/page categorization using text or vision models

### 4. Advanced Document Processing
- **Multi-column/Page Flows**: Reflow content across columns or pages for proper reading order
- **Layout Analysis**: YOLO, TATR for automatic document structure detection  
- **Visual Element Detection**: Checkbox classification, form field extraction
- **Table Structure Detection**: Manual line detection for complex tables

## Documentation Structure Analysis

### Tutorial Organization
The library follows a progressive learning approach:
1. **Basics**: Loading, text extraction, simple table extraction
2. **OCR**: Multiple engines, correction workflows, image-based documents
3. **AI Integration**: Q&A, structured extraction, classification
4. **Advanced**: Multi-column flows, layout analysis, spatial navigation

### Code Examples Pattern
- Jupyter notebook format with visual outputs
- Real-world document examples (inspection reports, CIA documents)
- Progressive complexity from simple to advanced use cases
- Visual debugging with `.show()` methods throughout

### Installation Patterns
- Modular installation with extras: `"natural-pdf[ocr-export,ai]"`
- Optional dependencies for different AI/OCR backends
- Clear separation of core vs. advanced features

## Common API Patterns

### Element Selection (jQuery-style)
```python
# CSS-like selectors
page.find('text:contains("Summary")')
page.find_all('text[size>12]:bold')
page.find('rect[width>10]')

# Spatial navigation
element.below(until='text:contains("End")')
element.above().left_of(other_element)
```

### Processing Workflows
```python
# Standard pattern
pdf = npdf.PDF(source)
page = pdf.pages[0]
elements = page.find_all(selector)
results = elements.apply(processing_function)
```

### Visualization & Debugging
- Consistent use of `.show()` for visual debugging
- `.inspect()` for detailed element analysis
- Cropping and grouping options for complex layouts

## Integration Points

### AI Services
- OpenAI API compatible (including Gemini, other providers)
- Local models for classification (CLIP, YOLO)
- Hugging Face transformers integration

### Export Formats
- PDF manipulation and creation
- HOCR output for OCR results
- Structured data (JSON, CSV via pandas)

## Comprehensive Feature Set Analysis

### Installation & Architecture
- **Modular Dependencies**: CLI tool (`npdf install`) for managing optional engines/models
- **Core Package**: Basic PDF parsing and spatial navigation (`pip install natural-pdf`)
- **Optional Extras**: `[ai]`, `[deskew]`, `[search]` for specialized features
- **Engine Detection**: Library provides clear error messages with install commands when engines missing

### OCR Engine Ecosystem
- **Supported Engines**: EasyOCR (default), PaddleOCR, Surya, DocTR with easy switching
- **Engine-Specific Options**: Configurable through dedicated options classes (PaddleOCROptions, etc.)
- **Detection vs Recognition**: Support for detect-only mode followed by LLM correction
- **Resolution Control**: Configurable OCR resolution for quality vs speed tradeoffs
- **Fine-tuning Support**: Complete workflow for training custom PaddleOCR models with LLM-generated training data

### Layout Analysis Models
- **YOLO (Default)**: Fast general-purpose layout detection
- **TATR**: Microsoft's table transformer - specialized for detailed table structure
- **Surya**: High-accuracy document layout analysis
- **Paddle**: Alternative layout analysis option
- **Gemini Integration**: LLM-based layout analysis via OpenAI-compatible API

### AI Integration Patterns
- **Local Models**: CLIP for classification, LayoutLM for document Q&A
- **Remote APIs**: OpenAI-compatible endpoints (OpenAI, Anthropic, Gemini, local LLMs)
- **Offline Capability**: Core extraction works without internet using local models
- **Confidence Thresholding**: All AI results include confidence scores and filtering

### Advanced Data Extraction
- **Document Q&A**: Extractive question answering with spatial context preservation
- **Structured Extraction**: Pydantic schema support for complex data structures
- **Table Intelligence**: TATR provides detailed table structure (rows, columns, headers, cells)
- **Classification**: Text vs vision-based categorization for documents/pages/regions
- **Semantic Search**: Vector-based content discovery within documents

### Visual Debugging & Development
- **Interactive Viewer**: Jupyter widget for document exploration
- **Highlighting System**: Color-coded visualization with grouping and labeling
- **Crop & Show**: Focused visualization of specific regions
- **OCR Correction Tool**: Web-based interface for human-in-the-loop OCR correction

### Spatial Navigation System
- **CSS-like Selectors**: `text:contains("Summary"):bold`, `rect[width>10]`
- **Spatial Operators**: `.below()`, `.above()`, `.left_of()`, `.right()`, `.near()`
- **Relationship Queries**: `:above(selector)`, `:below(selector)`, `:near(selector)`
- **Exclusion Zones**: Page and PDF-level content exclusions for headers/footers

### Table Processing Pipeline
- **Multiple Extraction Methods**: pdfplumber (line-based), TATR (structure-aware), custom settings
- **Automatic Method Selection**: Smart defaults based on detection engine used
- **Manual Line Detection**: Custom threshold-based table structure detection
- **Cell-Level Access**: Individual table cell extraction and processing

### Collection & Batch Processing
- **PDF Collections**: Batch processing across multiple documents with parallel execution
- **Element Collections**: Functional programming patterns for element manipulation
- **Progress Tracking**: Built-in progress bars for long-running operations
- **Memory Management**: Proper resource cleanup with context managers

## AI-Generated Content Issues Identified - CONFIRMED BY AUTHOR

### Documentation Quality Concerns - VERIFIED
1. **Inconsistent Code Examples**: Some code snippets reference undefined variables or incorrect method calls
2. **Feature Availability**: Confirmed by author - "the docs thanks to AI do have a lot of 'we WILL have this' which maybe is just a hallucination"  
3. **Installation Instructions**: Inconsistencies between different installation methods mentioned
4. **Model Availability**: References to models that may not be readily available or properly integrated

### Implementation Reality vs Documentation
1. **Performance Characteristics**: Author confirms "performance is solid, honestly" - documentation overstates issues
2. **Dependency Management**: Well-structured pyproject.toml with clear optional dependencies and modular installation via CLI
3. **Error Handling**: Robust pattern with `is_available()` checks and clear installation guidance when engines missing
4. **Architecture**: Clean separation of concerns with pdfplumber as foundation + AI/spatial intelligence layers

## Architecture Deep Dive

### Spatial Navigation System (CONFIRMED)
- Built on pdfplumber bounding boxes as the user confirmed
- DirectionalMixin in `natural_pdf/elements/base.py` implements core spatial navigation
- Methods like `.above()`, `.below()`, `.left()`, `.right()` use coordinate calculations
- Region class in `natural_pdf/elements/region.py` builds regions using bounding box operations
- All spatial relationships computed from coordinate geometry

### Analysis Engine Pattern (CONFIRMED)
- Analysis engines store results in metadata fields (not separate database)
- Each engine works differently but follows similar patterns
- OCR engines use abstract base class pattern in `natural_pdf/ocr/engine.py`
- Implementations in separate files: `engine_easyocr.py`, `engine_surya.py`, etc.
- Dependency checking via `is_available()` methods prevents import errors
- TextRegion class standardizes OCR output format across engines

### CSS-like Selector System (CONFIRMED)
- Sophisticated parsing in `natural_pdf/selectors/parser.py`
- Supports OR operators with complex syntax: `text:contains("A")|text:bold`
- Color matching using Delta E color distance calculations
- Handles nested parentheses, quoted strings, spatial relationships
- Regex support for advanced pattern matching

### Lazy Loading Architecture (CONFIRMED)
- Pages are lazy-loaded via `_LazyPageList` class in `natural_pdf/core/pdf.py`
- Pages created on-demand to optimize memory usage
- Manager registry pattern for different analysis engines
- Context manager support with automatic cleanup

### Modular Installation System (CONFIRMED)
- CLI-based installation system in pyproject.toml
- Optional extras like `[search]`, `[ocr]`, etc.
- Dependencies separated to avoid bloat
- Engine availability checking prevents runtime errors

### Performance Reality vs Documentation
- User confirmed: "performance is solid, honestly"
- Documentation issues: AI-generated content contains "we WILL have this" aspirational features
- Actual implementation is more mature than docs suggest
- Spatial indexing and engine patterns are production-ready

## Mixin Architecture Analysis

### Heavy Use of Mixins Pattern
The Page class (`natural_pdf/core/page.py`) demonstrates extensive use of mixins:
```python
class Page(ClassificationMixin, ExtractionMixin, ShapeDetectionMixin, DescribeMixin):
```

This creates a rich API where pages can:
- **Classify** content using text or vision models (`ClassificationMixin`)
- **Extract** structured data with Pydantic schemas (`ExtractionMixin`) 
- **Detect** shapes and geometric patterns (`ShapeDetectionMixin`)
- **Describe** and inspect content with summaries (`DescribeMixin`)

### Analysis Storage Pattern
All mixins use a standardized `self.analyses[key]` storage pattern:
- Results stored in `analyses` dictionary with configurable keys
- Prevents naming conflicts between different analysis types
- Supports multiple analyses of same type (e.g., different classification models)
- Result objects maintain success flags, confidence scores, and error messages

### Manager Registry Pattern
Pages maintain instances of specialized managers via parent PDF:
- `OCRManager` - handles all OCR engines and operations
- `LayoutManager` - manages layout detection models
- `ClassificationManager` - handles text/vision classification
- `StructuredDataManager` - manages LLM-based extraction
- `ElementManager` - handles element loading and creation

### Content Abstraction Methods
Mixins define abstract content retrieval methods:
- `_get_classification_content(model_type, **kwargs)` - gets text or image for classification
- `_get_extraction_content(using, **kwargs)` - gets content for structured extraction
- Auto-fallback logic (text→vision when text empty)

### Element Management Deep Dive
`ElementManager` class centralizes element lifecycle:
- **Lazy Loading**: Elements created on-demand to optimize memory
- **Custom Word Extraction**: `NaturalWordExtractor` respects font boundaries for word splitting
- **OCR Integration**: Seamlessly merges native PDF and OCR-derived elements
- **Unified Element Types**: chars, words, rects, lines, regions all managed consistently

### Fresh Eye Observations & Questions

#### Potential Architecture Concerns:
1. **Mixin Explosion**: Page class inherits from 4 mixins - could become unwieldy as features grow
2. **Manager Coupling**: Heavy dependency on parent PDF having correct manager instances
3. **Error Handling**: What happens when managers are missing or unavailable?
4. **Memory Management**: With lazy loading and multiple analysis results, what's the memory footprint?

#### Coordinate System Hints Found:
- OCR coordinate scaling: `scale_x`, `scale_y` parameters in `create_text_elements_from_ocr`
- Coordinate normalization between image and PDF spaces
- `pdf_render_lock` used extensively throughout codebase

#### Thread Safety & Rendering Issues:
- **pdf_render_lock Usage**: Extensive locking required because pdfplumber's underlying rendering tool isn't thread-safe
- This explains the pervasive locking pattern across the codebase
- Potential performance bottleneck in multi-threaded scenarios

#### Questions for Discussion:
1. **Coordinate Consistency**: How do coordinates work between PDF space, image space, and different resolution renders?
2. **Manager Dependencies**: How robust is the system when managers are unavailable?
3. **Performance Scaling**: With multiple mixins and managers, how does performance scale with document size?
4. **Error Recovery**: How does the system handle partial failures in multi-step analysis workflows?
5. **Rendering Performance**: Does the thread-unsafe rendering create significant bottlenecks in concurrent processing?
6. **Coordinate Transformation**: What are the specific pain points in coordinate space transformations?

## Scale Parameter Investigation

### Current Scale Usage Analysis
- **Scale Purpose**: Multiplier of base 72 DPI (scale=2.0 → 144 DPI)  
- **Deep Integration**: Used throughout OCR coordinate transformation, layout analysis, and rendering
- **Formula**: `render_resolution = scale * 72` then `actual_scale = resolution / 72.0`

### Scale vs Resolution/Width
- **Scale**: User-friendly multiplier (2x, 3x size)
- **Resolution**: Explicit DPI control  
- **Width**: Pixel dimension constraints
- **Coordinate Mapping**: Scale factors critical for image↔PDF coordinate transformations

### Key Dependencies Found:
1. **OCR Pipeline**: `create_text_elements_from_ocr(scale_x, scale_y)` for coordinate transformation
2. **Layout Analysis**: Image-to-PDF coordinate scaling in `layout_analyzer.py`
3. **Rendering**: Scale-to-resolution conversion throughout highlighting service
4. **API Compatibility**: Extensive existing usage in user-facing methods

### Scale Parameter Removal Plan
**User decision: Remove scale entirely** - DPI is more intuitive for professional users, no backward compatibility needed.

**Refactoring Strategy:**
1. **Convert scale to resolution**: `scale * 72 = resolution` (scale=2.0 → resolution=144)
2. **Update all method signatures**: Remove scale parameters, use resolution/width only
3. **Coordinate transformation updates**: Replace scale calculations with `resolution / 72.0`
4. **Default values**: Use resolution=144 (equivalent to previous scale=2.0 default)

**Files requiring updates:**
- `core/page.py` - to_image(), show(), save_image() methods
- `core/highlighting_service.py` - render_page(), render_preview() methods  
- `elements/region.py` - all image generation methods
- `elements/base.py` - Element show/save methods
- `analyzers/layout/layout_analyzer.py` - layout analysis coordinate scaling
- `core/element_manager.py` - OCR coordinate transformation
- `utils/visualization.py` - rendering utilities

**Breaking changes acceptable** - cleaner DPI-based API preferred over scale abstraction.

## Scale Parameter Removal Status

### Current State Analysis
**Page.to_image() already converted** ✅ - Uses `resolution=144` default instead of scale

**Remaining scale usage found:**
1. **Flow classes** (`flows/region.py`, `flows/collections.py`) - `show(scale=2.0)` methods
2. **Element Manager** (`core/element_manager.py`) - OCR coordinate transformation `scale_x`, `scale_y`
3. **Shape Detection** (`analyzers/shape_detection_mixin.py`) - Internal scale_factor calculations  
4. **Highlighting Service** (`core/highlighting_service.py`) - Internal scale_factor usage

### Issue Discovered & Fixed ✅
Page.to_image() resolution parameter **was not working correctly** - global resolution setting was unconditionally overriding explicit parameters.

**Root cause**: Line 1591-1592 in page.py always overwrote resolution parameter with global setting
**Fix**: Changed resolution parameter to Optional[float] = None and only use global setting when not explicitly provided

**Results after fix:**
- 72 DPI: 612×792 pixels (scale 1.0) ✅
- 144 DPI: 1224×1584 pixels (scale 2.0) ✅  
- 216 DPI: 1836×2376 pixels (scale 3.0) ✅

### Updated Refactoring Plan
1. ~~**Fix page.to_image() resolution handling**~~ ✅ **COMPLETED**
2. ~~**Convert Flow classes**~~ ✅ **COMPLETED** - replaced `scale` parameters with `resolution` 
3. **OCR coordinate transformation** ✅ **PRESERVED** - `scale_x`/`scale_y` are coordinate conversion factors, not rendering scale
4. **Internal scale_factor calculations** ✅ **PRESERVED** - correctly derive from resolution in highlighting service

## Scale Parameter Removal - COMPLETED ✅

### Final Status
**All user-facing scale parameters successfully removed and replaced with resolution parameters:**

✅ **Page.to_image()** - Fixed resolution parameter override bug
✅ **FlowRegion.show()** - Converted scale=2.0 → resolution=None 
✅ **FlowElementCollection.show()** - Converted scale=2.0 → resolution=None
✅ **FlowRegionCollection.show()** - Converted scale=2.0 → resolution=None

### Verification Results
**Pixel-perfect consistency maintained** - Before/after images are identical:
- 72 DPI: 612×792 pixels ✅
- 108 DPI: 918×1188 pixels ✅  
- 144 DPI: 1224×1584 pixels ✅
- 180 DPI: 1530×1980 pixels ✅
- 216 DPI: 1836×2376 pixels ✅

### What Was Preserved
- **OCR coordinate transformation**: `scale_x`/`scale_y` parameters kept (these convert image pixels → PDF points)
- **Internal scale calculations**: Highlighting service correctly derives scale factors from resolution  
- **Mathematical precision**: All coordinate transformations remain exact

**Scale parameter removal successfully completed with zero breaking changes to coordinate accuracy.**

## Coordinate System Discussion Pending
User mentioned: "coordinate system is something we need to work on...I can talk about coordinates once you review those."

Since I've now reviewed the architecture, ready for coordinate system discussion. I see hints of coordinate transformation issues between PDF/image spaces and potential thread safety concerns with rendering.

## Development Notes
- Heavy use of method chaining and fluent APIs with jQuery-like patterns
- Visual debugging is a first-class feature throughout the entire workflow
- Modular design with optional AI/OCR components and clear dependency boundaries
- Clear separation between document structure analysis and AI enhancement features
- Emphasis on "show, don't tell" debugging with immediate visual feedback
- Strong focus on spatial relationships and document understanding rather than just text extraction

## Future Enhancement Ideas

### Image Generation Enhancements
**Image Format & Quality Options**
- **Multiple formats**: Support JPEG, WebP, TIFF alongside current PNG default
- **Quality controls**: Compression settings for different use cases (preview vs. export)
- **Format-specific options**: JPEG quality levels, PNG compression, WebP lossless/lossy
- **Configuration API**: `npdf.options.image.configure({'format': 'JPEG', 'quality': 85})`
- **Use cases**: High-quality exports vs. fast preview generation

**Memory Management**
- **LRU page cache**: Automatic cleanup of least recently used pages when memory pressure detected
- **Rendered image limits**: Configurable limits on cached images per page and total memory usage
- **Memory monitoring**: Track usage of pages, elements, and cached images with automatic cleanup thresholds
- **Explicit cleanup**: Context managers for large operations (`with pdf.memory_managed(): ...`)
- **User-configurable limits**: `npdf.options.memory.max_cached_pages = 10`

## Bad PDF Analysis Project - Key Learnings

### User Feedback on Missed Existing Capabilities (2025-06-22)

While analyzing challenging PDF submissions, I suggested several features that Natural PDF already supports. The user provided crucial corrections:

#### 1. **Exclusion Zones Already Exist**
**My suggestion:** Line number filtering  
**Reality:** Natural PDF already has `add_exclusions` feature  
**Example:** `pdf.add_exclusion(lambda page: page.find('text[x0<100]:contains(\d)', regex=True).below(include_source=True))`  
**Lesson:** Check existing exclusion capabilities before suggesting new filtering features

#### 2. **Multi-page Tables Already Supported**
**My suggestion:** Cross-page table continuation  
**Reality:** Flows system already handles this at `@natural_pdf/flows/flow.py` and `@docs/reflowing-pages/index.md`  
**Lesson:** Investigate flows documentation for multi-page content handling

#### 3. **Mixed Content Is The Expected Pattern**
**My suggestion:** Smart document structure detection for mixed content  
**Reality:** Natural PDF assumes mixed content by design  
**Pattern:** Use spatial navigation like `.find('text:contains(Violations')`, then `.below()`, `.right()`, etc. with `until='text[size>12]:bold'` to find sections, build regions, then `.extract_table()` on those regions  
**Lesson:** Mixed content handling is core to the spatial navigation approach

#### 4. **Streaming/Chunking May Not Be Critical**
**My suggestion:** Document chunking and streaming for memory management  
**User question:** "do you think streaming/chunking is important? we already lazy-load the pages"  
**Reality:** Pages are already lazy-loaded, so this may not be a priority  
**Lesson:** Understand existing memory management before suggesting new approaches

#### 5. **Accessibility Assessment Out of Scope**
**My suggestion:** ADA/accessibility compliance checking  
**Reality:** "we are also not concerned with accessibility assessment: this tool is only to extract structured data from pdfs"  
**Lesson:** Natural PDF is focused on data extraction, not document compliance or accessibility

### Initial Analysis Approach Issues

**Problem:** I focused too much on application domains (government transparency, police accountability, cultural heritage) rather than structural/technical challenges  
**Correction:** Focus on document structure patterns and extraction mechanics:
- Tables with multi-line cells spanning visual rows
- Mixed table formats requiring different strategies  
- Massive documents needing performance optimization
- Text formatting stored as separate visual elements

**Better approach:** Describe the visual/structural pattern, not the subject matter domain

### Questions for Further Investigation

Based on this feedback, I should explore:

1. **Exclusion system capabilities:** What types of exclusion patterns are already supported? How flexible is the lambda-based exclusion system?

2. **Flows system:** How robust is the multi-page content handling? What are the current limitations?

3. **Spatial navigation limits:** What are the edge cases where the current spatial navigation approach breaks down?

4. **Performance characteristics:** Since lazy loading exists, what are the actual performance bottlenecks in large document processing?

5. **Layout analysis integration:** How well do YOLO/TATR results integrate with the spatial navigation system?

6. **OCR coordination:** Does the multi-engine OCR approach already handle script mixing and quality assessment?

### Key Realization
Natural PDF may already be more capable than I initially assessed. Many of my "enhancement suggestions" were reinventing existing wheels. The real gaps are likely more subtle - edge cases in existing systems rather than missing foundational capabilities.

### Refined Understanding of Actual Capabilities vs Gaps

#### ✅ **What Natural PDF Already Does Well**
1. **Exclusions**: Very flexible lambda-based system for filtering content
2. **Multi-page Content**: Flows system handles cross-page table continuation  
3. **Mixed Content**: Core design assumption - spatial navigation built for this
4. **Memory Management**: Lazy loading already implemented
5. **OCR Integration**: Multiple engines supported with flexible options
6. **Spatial Navigation**: Robust jQuery-like API for document traversal
7. **Layout Analysis Integration**: YOLO/TATR results work with spatial navigation

#### 🔧 **Actual Gaps Worth Addressing**
1. **Error Handling**: Spatial navigation fails ungracefully when elements missing
2. **Text Formatting Association**: Rect/line elements not linked to formatted text
3. **Multi-Engine OCR Workflows**: Need examples for detect-only + correction patterns
4. **Engine Comparison**: No easy way to compare OCR engine results
5. **Performance Boundaries**: Unknown limits for very large documents

#### 💡 **Revised Approach to "Bad PDF" Challenges**

**For Senate Expenditures (multi-line cells):**
- Use spatial navigation to extract table regions ✅
- Clean up multi-line cells in pandas post-processing ✅  
- NOT a Natural PDF problem - data structuring is out of scope

**For Georgia Bills (text formatting):**
- Use spatial navigation to find rect elements near text ✅
- Associate formatting via coordinate overlap analysis 🔧 (needs work)
- Create text elements with formatting attributes like bold/italic 🔧 (needs work)

**For Massive Puerto Rico Document:**
- Lazy loading already handles memory management ✅
- Use exclusions to skip administrative pages ✅
- Flows can handle multi-page tables ✅
- NOT a streaming problem - existing architecture adequate

**For Japanese Historical Document:**
- High-resolution OCR already supported ✅
- Multi-script handling via OCR engine selection ✅
- User should know script requirements going in ✅
- NOT a coordination problem - OCR engines handle this

**For Police Microscopic Font:**
- High-resolution OCR already supported ✅
- If OCR doesn't work, you can't recover zero signal ✅
- NOT an accessibility problem - data extraction only

### Focus Areas Based on Real Gaps
1. **Improve robustness** (error handling, edge cases)
2. **Text formatting detection** (common legal/legislative pattern)  
3. **OCR workflow examples** (multi-engine patterns)
4. **Performance characterization** (when we hit actual limits)

### Anti-Patterns to Avoid
- Suggesting features that spatial navigation already handles
- Focusing on application domains rather than structural patterns
- Assuming accessibility/compliance is in scope
- Reinventing memory management that lazy loading already provides