# natural-pdf Architecture Guide

This document describes the architectural patterns and design principles used in natural-pdf, established through systematic refactoring to improve modularity, testability, and maintainability.

## Table of Contents

- [Core Principles](#core-principles)
- [Service-First Architecture](#service-first-architecture)
- [Architectural Patterns](#architectural-patterns)
- [Implementation Examples](#implementation-examples)
- [Testing Approach](#testing-approach)
- [Decision Trees](#decision-trees)

---

## Core Principles

### 1. Service-First Design

**Principle**: Complex logic lives in services, not in host objects (`PDF`, `Page`, `Region`).

**Why**:
- Services are testable in isolation
- Host objects remain simple and focused on their core identity
- Logic can be reused across different hosts
- Clear separation of concerns

### 2. Thin Wrappers for Ergonomics

**Principle**: Host objects provide thin, ergonomic wrappers that delegate to services.

**Why**:
- Users get clean, chainable APIs (`page.classify()`, `page.extract()`)
- Implementation complexity hidden from users
- Easy to find where logic lives (in the service)

### 3. Text as Core Identity

**Principle**: `extract_text()` stays in hosts, using shared utilities.

**Why**:
- Text extraction defines what a Page/Region **is**
- Not a "feature" like tables or classification
- Already modular through utility functions

---

## Service-First Architecture

### The Pattern

```
┌─────────────────────────────────────────────────────────────┐
│  User calls host method                                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Page/Region/PDF (Host Object)                        │  │
│  │                                                        │  │
│  │  def feature(self, ...):                             │  │
│  │      """Ergonomic thin wrapper."""                   │  │
│  │      return self.services.capability.method(         │  │
│  │          self, ...                                    │  │
│  │      )                                                 │  │
│  └───────────────────┬──────────────────────────────────┘  │
│                      │ delegates                            │
│  ┌───────────────────▼──────────────────────────────────┐  │
│  │ Service (CapabilityService)                          │  │
│  │                                                        │  │
│  │  @register_delegate("capability", "method")          │  │
│  │  def method(self, host, ...):                        │  │
│  │      """Orchestrates logic, calls operations."""     │  │
│  │      # Complex implementation here                    │  │
│  │      return result                                    │  │
│  └───────────────────┬──────────────────────────────────┘  │
│                      │ uses (optional)                      │
│  ┌───────────────────▼──────────────────────────────────┐  │
│  │ Operations Module (capability/operations.py)         │  │
│  │                                                        │  │
│  │  def operation(data, ...):                           │  │
│  │      """Pure function doing actual work."""          │  │
│  │      return transformed_data                          │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Components

1. **Host Object** (`PDF`, `Page`, `Region`)
   - Provides user-facing API
   - Thin wrappers only (2-5 lines)
   - No business logic

2. **Service** (`services/capability_service.py`)
   - Orchestrates complex operations
   - Manages state/caching if needed
   - Registered via `@register_delegate`

3. **Operations Module** (optional: `capability/operations.py`)
   - Pure functions
   - Stateless, testable
   - Reusable utilities

4. **Provider** (optional: `capability/provider.py`)
   - Engine discovery/registration
   - Extensibility hook
   - Protocol-based

---

## Architectural Patterns

### Pattern 1: Thin Wrapper + Service

**Use when**: Feature has complex logic that should be testable.

**Structure**:
```python
# Host (Page)
def classify(self, labels, model=None, **kwargs):
    """Classify page content using ML models."""
    return self.services.classification.classify(
        self, labels=labels, model=model, **kwargs
    )

# Service (ClassificationService)
@register_delegate("classification", "classify")
def classify(self, host, labels, model=None, ...):
    # Complex implementation
    engine = get_classification_engine(...)
    result = engine.classify_item(...)
    return result
```

**Examples**:
- [Classification](#example-classification)
- [Structured Data Extraction](#example-structured-data)
- [Table Extraction](#example-tables)

---

### Pattern 2: Operations Module

**Use when**: Multiple hosts share complex utilities.

**Structure**:
```python
# operations.py
def filter_chars_spatially(
    char_dicts: List[Dict],
    exclusion_regions: List[Region],
    target_region: Optional[Region] = None,
) -> List[Dict]:
    """Filter character dicts by spatial constraints."""
    # Complex filtering logic
    return filtered_chars

# Host (Page)
def extract_text(self, ...):
    chars = self._element_mgr.chars
    filtered = filter_chars_spatially(chars, exclusions)
    return generate_text_layout(filtered)
```

**Examples**:
- [Text Utilities](#example-text-utilities)

**Key principle**: Operations are **pure functions** with no side effects.

---

### Pattern 3: Provider Pattern

**Use when**: Feature supports multiple engines/backends.

**Structure**:
```python
# provider.py
class CapabilityEngine:
    """Protocol for custom engines."""
    def execute(self, ...): ...

class _DefaultEngine(CapabilityEngine):
    def execute(self, ...):
        # Call operations.py functions
        return result

def get_engine(context, name: Optional[str] = None):
    """Get engine by name, fallback to default."""
    provider = get_provider()
    return provider.get_engine("capability", name or "default")

# Service
def method(self, host, ...):
    engine = get_engine(host._context)
    return engine.execute(...)
```

**Examples**:
- [Classification Engines](#example-classification)

**Key principle**: Providers enable **extensibility** without modifying core code.

---

### Pattern 4: Service with Rich Implementation

**Use when**: Service needs threading, progress tracking, or complex orchestration.

**Structure**:
```python
# Service
@register_delegate("text", "update_text")
def update_text(
    self, host, transform,
    max_workers: Optional[int] = None,
    show_progress: bool = False,
):
    elements = host.find_all(selector)

    # Complex threading logic
    if max_workers and max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Parallel processing
    else:
        # Sequential processing

    return host

# Host wrapper
def update_text(self, transform, show_progress=True, **kwargs):
    return self.services.text.update_text(
        self, transform, show_progress=show_progress, **kwargs
    )
```

**Examples**:
- [Text Update](#example-update-text)

**Key principle**: Host wrapper controls **defaults**, service implements **logic**.

---

## Implementation Examples

### Example: Classification

**Location**: `services/classification_service.py`, `classification/provider.py`, `classification/pipelines.py`

**Pattern**: Thin wrapper → Service → Provider → Operations

```python
# 1. Host wrapper (Page/Region)
def classify(self, labels, model=None, **kwargs):
    return self.services.classification.classify(
        self, labels=labels, model=model, **kwargs
    )

# 2. Service orchestration
@register_delegate("classification", "classify")
def classify(self, host, labels, model=None, ...):
    engine = get_classification_engine(host._context)
    content = self._get_content(host, using)
    result = engine.classify_item(
        item_content=content, labels=labels, model_id=model
    )
    host.analyses[key] = result
    return result

# 3. Provider discovery
def get_classification_engine(context, name=None):
    provider = get_provider()
    return provider.get_engine("classification", name or "default")

# 4. Operations (pipelines.py)
def classify_single(*, item_content, labels, model_id, ...):
    """Stateless classification function."""
    pipeline = _load_pipeline(model_id, using)
    result = pipeline(item_content, candidate_labels=labels)
    return ClassificationResult(...)
```

**Why this structure**:
- Service handles orchestration
- Provider allows custom engines
- Operations are testable in isolation

---

### Example: Structured Data

**Location**: `services/extraction_service.py`, `extraction/structured_ops.py`

**Pattern**: Thin wrapper → Service → Operations

```python
# 1. Host wrapper
def extract(self, schema, client=None, **kwargs):
    self.services.extraction.extract(
        self, schema=schema, client=client, **kwargs
    )
    return self

# 2. Service
@register_delegate("extraction", "extract")
def extract(self, host, schema, client, ...):
    content = host._get_extraction_content(using)
    result = extract_structured_data(
        content=content, schema=schema_model,
        client=client, prompt=prompt, using=using
    )
    host.analyses[key] = result
    return host

# 3. Operations (structured_ops.py)
def extract_structured_data(*, content, schema, client, ...):
    """Pure function for LLM extraction."""
    messages = _prepare_llm_messages(content, prompt, using, schema)
    completion = client.beta.chat.completions.parse(...)
    return StructuredDataResult(...)
```

**Why this structure**:
- No provider needed (client passed directly)
- Operations are LLM-agnostic
- Easy to test with mock client

---

### Example: Tables

**Location**: `services/table_service.py`

**Pattern**: Thin wrapper → Service (complex)

```python
# 1. Host wrapper (Page)
def extract_table(self, method=None, **kwargs):
    region = self._full_page_region()
    return self.services.table.extract_table(region, method=method, **kwargs)

# 2. Service (single file, 413 lines)
@register_delegate("table", "extract_table")
def extract_table(self, host, method=None, ...):
    # Auto-detect method if not specified
    if method is None:
        if cell_regions_exist:
            return TableResult(build_table_from_cells(...))
        structure_table = self._extract_table_from_structure(...)
        if structure_table:
            return structure_table

    # Provider-based extraction
    engine = resolve_table_engine(...)
    return run_table_engine(...)
```

**Why this structure**:
- Table logic already uses providers
- Complex enough to justify service
- FlowRegion handling in service

---

### Example: Text Utilities

**Location**: `text/operations.py`

**Pattern**: Shared operations (no service)

```python
# operations.py
def filter_chars_spatially(char_dicts, exclusion_regions, ...):
    """Spatial filtering of characters."""
    # ... complex polygon/bbox logic
    return filtered_chars

def generate_text_layout(char_dicts, bbox, **kwargs):
    """Generate text with layout preservation."""
    textmap = chars_to_textmap(char_dicts, **layout_kwargs)
    return textmap.as_string

def apply_bidi_processing(text: str) -> str:
    """Convert RTL text to logical order."""
    # ... BiDi algorithm
    return processed_text

# Host (Page/Region)
def extract_text(self, ...):
    # Host-specific element gathering
    chars = self._get_char_dicts()

    # Shared utilities
    filtered = filter_chars_spatially(chars, exclusions, self.bbox)
    text = generate_text_layout(filtered, self.bbox, **kwargs)
    if bidi:
        text = apply_bidi_processing(text)
    return text
```

**Why this structure**:
- Text extraction is core to Page/Region identity
- Utilities eliminate duplication
- Each host tailors its own orchestration

---

### Example: update_text()

**Location**: `services/text_service.py`

**Pattern**: Service with threading

```python
# Service
@register_delegate("text", "update_text")
def update_text(
    self, host, transform,
    max_workers: Optional[int] = None,
    show_progress: bool = False,
):
    elements = list(host.find_all(selector))

    # Helper for result application
    def _apply_result(element, corrected, error):
        if error:
            return False, True
        if corrected and corrected != element.text:
            element.text = corrected
            return True, False
        return False, False

    # Parallel or sequential
    if max_workers and max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for future in as_completed(futures):
                updated, errored = _apply_result(...)
    else:
        for element in elements:
            updated, errored = _apply_result(...)

    return host

# Host
def update_text(self, transform, show_progress=True, **kwargs):
    return self.services.text.update_text(
        self, transform, show_progress=show_progress, **kwargs
    )
```

**Why this structure**:
- Threading complexity in service (testable)
- Progress tracking centralized
- Host provides sensible default (`show_progress=True`)

---

 ### Example: ElementManager (Modular Orchestrator)

 **Location**: `core/element_manager.py`, `core/element_store.py`, `core/word_engine.py`, `core/decoration_detector.py`

 **Pattern**: Orchestrator + Specialized Modules

 ```python
 # 1. Orchestrator (ElementManager)
 class ElementManager:
     def __init__(self, page):
         self._store = ElementStore()
         self._word_engine = WordEngine(page, ...)
         self._decorations = DecorationDetector(page)
         self._ocr_converter = OCRConverter(page)

     def load_elements(self):
         # Coordination logic
         with self._store.transaction():
             chars = self._prepare_chars()
             self._decorations.annotate_chars(chars)
             words = self._word_engine.generate_words(chars, ...)
             self._store.replace({"words": words, "chars": chars})

 # 2. Storage Module (ElementStore)
 class ElementStore:
     """Thread-safe container with invalidation callbacks."""
     def transaction(self): ...
     def register_callback(self, kind, cb): ...

 # 3. Logic Modules (WordEngine, DecorationDetector)
class WordEngine:
    """Stateless word segmentation logic."""
    def generate_words(self, chars, ...): ...
```

**Shared helpers for other hosts**

- `Page._get_element_loader()` exposes the same loader instance the manager uses so
  Regions, collections, and importers can enrich raw character dictionaries without
  instantiating temporary `TextElement`s. Region-to-text conversions and manifest
  imports now call this helper before creating words.
- `Page._get_decoration_detector()` exposes the active `DecorationDetector`. Any host
  that synthesizes text elements (regions, manifest imports, etc.) runs the detector
  to annotate char dicts and propagate decoration flags to the resulting words.
  This keeps highlight/underline/strike metadata consistent regardless of where the
  text originated.

**Why this structure**:
- **Orchestrator**: Manages lifecycle and sequencing
- **Storage**: Handles concurrency and invalidation centrally
- **Modules**: Encapsulate domain logic (BiDi, decorations)
- **Testability**: Each module tested in isolation with real PDFs

 ---

## Testing Approach

### Testing Services

Services are the primary test target:

```python
def test_classification_service():
    # Create minimal context
    context = PDFContext()
    service = ClassificationService(context)

    # Create mock host
    class MockHost:
        def _get_classification_content(self, using):
            return "test page content"
        analyses = {}

    host = MockHost()

    # Test service directly
    result = service.classify(
        host, labels=["invoice", "receipt"], model="test-model"
    )

    assert result.scores
    assert "classification" in host.analyses
```

### Testing Operations

Operations are pure functions:

```python
def test_filter_chars_spatially():
    chars = [
        {"x0": 10, "top": 10, "x1": 20, "bottom": 20, "text": "a"},
        {"x0": 100, "top": 100, "x1": 110, "bottom": 110, "text": "b"},
    ]
    exclusion = Region(page, (95, 95, 115, 115))

    filtered = filter_chars_spatially(chars, [exclusion])

    assert len(filtered) == 1
    assert filtered[0]["text"] == "a"
```

### Testing Host Wrappers

Usually not needed (just delegation tests):

```python
def test_page_classify_delegates():
    page = PDF("test.pdf").pages[0]

    # Integration test - ensure delegation works
    result = page.classify(labels=["test"])

    assert isinstance(result, ClassificationResult)
```

---

## Decision Trees

### When to Create a Service?

```
Is the feature complex (>50 lines)?
├─ No → Keep inline in host
└─ Yes → Continue

Does it involve AI/ML models?
├─ Yes → Use Service + Provider pattern
└─ No → Continue

Is logic reused across Page/Region/PDF?
├─ Yes → Use Service pattern
└─ No → Consider keeping in host with helper methods
```

**Examples**:
- ✅ Classification: Complex + ML → Service + Provider
- ✅ Tables: Complex + Reused → Service
- ✅ Text update: Complex + Threading → Service
- ❌ Text extraction: Core identity → Operations only

---

### When to Create an Operations Module?

```
Is logic shared between Page and Region?
├─ Yes → Continue
└─ No → Keep in single location

Is the logic stateless (pure function)?
├─ Yes → Create operations.py
└─ No → Consider service instead

Is it >100 lines or >3 functions?
├─ Yes → Create dedicated module
└─ No → Keep in service helpers
```

**Examples**:
- ✅ text/operations.py: Shared + Stateless + Large
- ❌ One-off helper: Keep in service

---

### When to Use the Provider Pattern?

```
Does feature support multiple backends/engines?
├─ No → Don't use provider
└─ Yes → Continue

Do users need to register custom implementations?
├─ Yes → Use provider pattern
└─ No → Continue

Are there 2+ built-in engines?
├─ Yes → Use provider pattern
└─ No → Wait until there are
```

**Examples**:
- ✅ Classification: Multiple models (BART, CLIP, custom)
- ✅ Tables: Multiple engines (pdfplumber, TATR, custom)
- ❌ Extraction: Single LLM client (no provider)

---

## Migration Checklist

When extracting logic to a service:

### Planning
- [ ] Identify host method(s) to migrate
- [ ] Determine if operations module needed
- [ ] Determine if provider pattern needed
- [ ] Check for dependencies/side effects

### Implementation
- [ ] Create/update service file
- [ ] Move complex logic to service method
- [ ] Add `@register_delegate` decorator
- [ ] Create operations module (if needed)
- [ ] Create provider (if needed)
- [ ] Replace host method with thin wrapper

### Quality
- [ ] Service method has proper signature (`self, host, ...`)
- [ ] Host wrapper delegates all parameters
- [ ] No business logic in host wrapper
- [ ] Operations are pure functions (if used)
- [ ] Error handling preserved
- [ ] Logging preserved

### Testing
- [ ] Test service in isolation
- [ ] Test operations as pure functions
- [ ] Integration test via host method
- [ ] Test threading/async if applicable

### Documentation
- [ ] Update host method docstring
- [ ] Add service method docstring
- [ ] Note any breaking changes

---

## Anti-Patterns to Avoid

### ❌ Fat Wrappers

**Bad**:
```python
def classify(self, labels, model=None, **kwargs):
    # Validation logic
    if not labels:
        raise ValueError("labels required")

    # Pre-processing
    content = self.extract_text()

    # Actual call (buried)
    return self.services.classification.classify(...)
```

**Good**:
```python
def classify(self, labels, model=None, **kwargs):
    return self.services.classification.classify(
        self, labels=labels, model=model, **kwargs
    )
```

Put validation/pre-processing in the **service**, not wrapper.

---

### ❌ Service Calling Service Publicly

**Bad**:
```python
class ExtractionService:
    def extract(self, host, ...):
        # Calling other service publicly
        classification = host.services.classification.classify(...)
```

**Good**:
```python
class ExtractionService:
    def extract(self, host, ...):
        # Get content from host interface
        content = host._get_extraction_content(using)
```

Services should call **host interfaces**, not other services directly.

---

### ❌ Stateful Operations

**Bad**:
```python
# operations.py
_cache = {}

def process_text(text):
    if text in _cache:  # Stateful!
        return _cache[text]
    result = expensive_operation(text)
    _cache[text] = result
    return result
```

**Good**:
```python
# service.py
class TextService:
    def __init__(self, context):
        self._cache = {}  # State in service

    def process(self, host, text):
        # Use service state
        if text not in self._cache:
            self._cache[text] = _process_impl(text)
        return self._cache[text]

# operations.py
def _process_impl(text):
    # Pure function
    return result
```

State belongs in **services**, not operations.

---

## Summary

The natural-pdf architecture follows these key patterns:

1. **Service-First**: Complex logic in services, not hosts
2. **Thin Wrappers**: Hosts provide ergonomic API via delegation
3. **Operations Modules**: Shared utilities as pure functions
4. **Providers**: Extensibility for multi-engine features
5. **Core Identity**: Text extraction stays in hosts, uses utilities

This architecture provides:
- ✅ **Testability**: Services and operations tested in isolation
- ✅ **Modularity**: Clear separation of concerns
- ✅ **Extensibility**: Provider pattern for customization
- ✅ **Maintainability**: Logic lives in one place
- ✅ **Ergonomics**: Clean user-facing API preserved

When in doubt, follow the decision trees and look at existing examples.
