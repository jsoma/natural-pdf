# Natural PDF Architectural Code Review

## Current Status

**Phases 1-4: COMPLETE** | **All Priorities: COMPLETE**

### Quick Summary
- ✅ Architecture mapped (~201 files, 20 packages)
- ✅ Engine provider unified (OCR, QA, Layout now use EngineProvider)
- ✅ Pattern consistency audited (36 issues found)
- ✅ Critical/high fixes implemented (6 fixes)
- ✅ Core model deep dive - well-designed, no issues (3a)
- ✅ AI integrations deep dive - caching/error issues found (3b)
- ✅ Selector system deep dive - performance optimizations needed (3c)
- ✅ Error handling scan - patterns identified (4a)
- ✅ Config validation scan - patterns identified (4b)
- ✅ **Priority 1**: Paddle cache key fixed
- ✅ **Priority 2**: Exception hierarchy created (`natural_pdf/exceptions.py`)
- ✅ **Consensus review**: GPT-5.2, Gemini-3-pro, Opus 4.5 (2-1 for warn+auto-correct)
- ✅ **Priority 3**: Config validation - IMPLEMENTED (see details below)
- ✅ **Priority 4**: Performance - Selector cache & OR deduplication

---

## Phase 3: Deep Dives ✅

### Approach
Ran sequentially (findings build on each other):

### 3a. Core Model Deep Dive ✅
**Focus**: Page/Region/Element class hierarchy and interactions

**Findings:**
- **Inheritance**: Heavy mixin composition (DirectionalMixin, ServiceHostMixin, SelectorHostMixin, etc.)
- **Circular deps**: None at runtime - properly managed via TYPE_CHECKING and local imports
- **Spatial navigation**: Unified `_direction()` method with smart defaults (left/right match element height, above/below span full width)
- **Three-tier hierarchy**: Physical (Element/Region/Page) → Collection (ElementCollection) → Flow (FlowRegion)
- **Service resolution**: Clean DI via PDFContext → ServiceHostMixin
- **Memory optimization**: TextElement uses `_char_indices` instead of `_char_dicts` (50% savings)
- **Status**: Well-designed, no issues found

### 3b. AI Integrations Deep Dive ✅
**Focus**: How OCR engines, LLM providers, layout analyzers are abstracted

**Findings:**
- **Abstraction**: Clean, consistent pattern across 4+ OCR engines and 6+ layout detectors
- **EngineProvider**: Well-designed plugin system with thread-safe caching
- **Config flow**: Multi-layer resolution (explicit → context → global → default)
- **Options**: Type-safe dataclasses with `extra_args` for forward compatibility

**Issues found:**
| Issue | Severity | Status |
|-------|----------|--------|
| Paddle cache key missing model params | High | ✅ Fixed |
| No custom exception hierarchy | High | ✅ Fixed |
| Silent fallback to defaults if wrong options type | Medium | ✅ Fixed (Priority 3) |
| No input validation (confidence bounds, device) | Medium | ✅ Fixed (Priority 3) |
| No resource cleanup strategy (models stay in memory) | Medium | Defer |
| No timeout handling for model loads | Low | Defer |

### 3c. Selector System Deep Dive ✅
**Focus**: Query parsing, matching, performance

**Findings:**
- **Syntax**: 2,084 lines covering types, attributes, pseudo-classes, aggregates, relational
- **Parsing**: Sequential regex matching → AST dict, safe (no eval), supports OR operators
- **Matching**: Filter-based, composable, three-phase (element → aggregate → relational)
- **Extensibility**: 4-tier handler registry (@register_pseudo, @register_attribute, etc.)

**Issues found:**
| Issue | Severity | Status |
|-------|----------|--------|
| No indexes - full-scan linear matching | High | Defer (major refactor) |
| No selector caching - reparsed every query | Medium | ✅ Fixed (Priority 4) |
| Two-pass aggregates computed from full pool, not filtered | Medium | Defer |
| OR selector duplication - evaluates pool k times | Medium | ✅ Fixed (Priority 4) |
| Relational pseudo re-evaluates reference selector | Medium | Defer |
| Late error detection for regex/references | Low | Defer |

---

## Phase 4: Cross-cutting Concerns (Focused Scan Complete)

### 4a. Error Handling Patterns ✅
**Current state:**
- 5 custom exceptions exist but scattered (JudgeError, ClassificationError, etc.)
- No unified hierarchy - mixed inheritance (Exception vs RuntimeError)
- 257 bare `except Exception` blocks (too many)
- 56 instances of proper exception chaining (`raise X from e`)
- RuntimeError overused for generic failures

**Pattern to follow:** `natural_pdf/judge.py` and `search_service_protocol.py` have good examples

### 4b. Configuration Validation Patterns ✅
**Current state:**
- Dataclass-based (not Pydantic)
- Only 2 classes have `__post_init__()` validation
- Global `set_option()` has NO validation
- `isinstance()` checks happen late (at engine init), not at construction
- Silent fallback to defaults when wrong options type passed

**Pattern to follow:** `TextStyleOptions.__post_init__()` in `analyzers/text_options.py`

---

## Action Plan: Phase 3 Fixes

### Priority 1: Isolated Fixes (no design decisions needed)
| Fix | File | Status |
|-----|------|--------|
| ~~TATR cache key~~ | Already correct (includes both models) | ✅ N/A |
| ~~Docling cache key~~ | Already correct (hashes extra_args) | ✅ N/A |
| ~~Surya cache key~~ | Already correct (includes model_name) | ✅ N/A |
| **Paddle cache key** - was missing model params | `analyzers/layout/paddle.py` | ✅ Fixed |
| Selector parsing cache | `selectors/parser.py` | ⏳ Pending (Priority 4) |

### Priority 2: Exception Hierarchy ✅ (informed by 4a)
| Fix | Status |
|-----|--------|
| Create `NaturalPDFError` base | ✅ `natural_pdf/exceptions.py` |
| Create `OCRError(NaturalPDFError)` | ✅ For OCR failures |
| Create `LayoutError(NaturalPDFError)` | ✅ For layout detection failures |
| Create `SelectorError(NaturalPDFError)` | ✅ For selector parsing/matching failures |
| Migrate `JudgeError` | ✅ Now inherits from NaturalPDFError |
| Migrate `ClassificationError` | ✅ Now uses unified definition |
| Migrate `IndexConfigurationError/IndexExistsError` | ✅ Now inherit from SearchError |
| Migrate `HocrTransformError` | ✅ Now inherits from ExportError |

### Priority 3: Config Validation ✅ COMPLETE

#### Consensus Review Results (GPT-5.2, Gemini-3-pro, Opus 4.5)
- **Vote**: 2-1 for warn+auto-correct (GPT + Opus vs Gemini)
- **Decision**: Warn + auto-correct by default, matches library's "approachable" philosophy
- **Skip Pydantic**: Overkill for ~10 classes with basic range checks
- **Key insight**: "Natural PDF's design philosophy emphasizes accessibility and forgiving APIs"

#### Implementation Approach

**Behavior Rules:**
```python
# WARN + AUTO-CORRECT for recoverable issues:
if confidence > 1.0:
    logger.warning(f"[EasyOCROptions] confidence={confidence} > 1.0, using 1.0")
    self.confidence = 1.0

# RAISE for truly invalid states (no safe default):
if model_path and not Path(model_path).exists():
    raise InvalidOptionError(f"model_path '{model_path}' does not exist")
```

**Warning Format (standardized):**
```
[ClassName] field_name={original_value} {reason}, using {corrected_value}
```

#### Step 1: Create Validation Helpers ✅
**File**: `natural_pdf/utils/option_validation.py` (CREATED)

```python
def validate_confidence(value: float, field_name: str = "confidence") -> float:
    """Validate confidence is 0.0-1.0, warn and clamp if not."""

def validate_positive_int(value: int, field_name: str) -> int:
    """Validate value > 0, warn and use 1 if not."""

def validate_device(value: str) -> str:
    """Validate device is cpu/cuda/mps, warn and use 'cpu' if not."""

def coerce_to_float(value: Any, field_name: str) -> float:
    """Coerce string "0.5" to float, warn if coerced."""
```

#### Step 2: Add __post_init__() to OCR Options ✅
**File**: `natural_pdf/ocr/ocr_options.py` (UPDATED)

| Class | Validations |
|-------|-------------|
| `BaseOCROptions` | confidence 0-1, device validation |
| `EasyOCROptions` | + batch_size > 0 |
| `PaddleOCROptions` | + batch_size > 0 |
| `SuryaOCROptions` | + batch_size > 0 |
| `DoctrOCROptions` | + batch_size > 0 |

#### Step 3: Add __post_init__() to Layout Options ✅
**File**: `natural_pdf/analyzers/layout/layout_options.py` (UPDATED)

| Class | Validations |
|-------|-------------|
| `BaseLayoutOptions` | confidence 0-1, device validation |
| `YOLOLayoutOptions` | + image_size > 0 |
| `TATRLayoutOptions` | + max_detection_size > 0, max_structure_size > 0 |
| `PaddleLayoutOptions` | (many optional params - validate if set) |
| `SuryaLayoutOptions` | (minimal) |
| `DoclingLayoutOptions` | (minimal) |
| `GeminiLayoutOptions` | + model_name not empty |

#### Step 4: Log isinstance() Fallbacks ✅
**Files modified** (key engines updated with `validate_option_type`):
- `natural_pdf/ocr/engine_easyocr.py`
- `natural_pdf/ocr/engine_paddle.py`
- `natural_pdf/ocr/engine_surya.py`
- `natural_pdf/ocr/engine_doctr.py`
- `natural_pdf/analyzers/layout/yolo.py`
- `natural_pdf/analyzers/layout/tatr.py`
- `natural_pdf/analyzers/layout/paddle.py`
- `natural_pdf/analyzers/layout/surya.py`
- `natural_pdf/analyzers/layout/docling.py`
- `natural_pdf/analyzers/layout/gemini.py`

**Change pattern:**
```python
# BEFORE (silent):
if not isinstance(options, EasyOCROptions):
    options = EasyOCROptions()

# AFTER (warn):
if not isinstance(options, EasyOCROptions):
    logger.warning(
        f"[EasyOCREngine] Expected EasyOCROptions, got {type(options).__name__}. "
        "Using default EasyOCROptions."
    )
    options = EasyOCROptions()
```

#### Step 5: Validate global set_option() ✅
**File**: `natural_pdf/__init__.py` (UPDATED)

Add validation schema and check in `set_option()`:
```python
_OPTION_VALIDATORS = {
    "ocr.min_confidence": lambda v: 0.0 <= v <= 1.0,
    "layout.confidence": lambda v: 0.0 <= v <= 1.0,
    "image.resolution": lambda v: v > 0,
}
```

#### Step 6: Add Optional Strict Mode ✅
**Environment variable**: `NATURAL_PDF_STRICT=1`

When set, raises `InvalidOptionError` instead of warn+auto-correct.

#### Testing ✅
- Tests in `tests/test_option_validation.py` (41 tests, all passing)
- Tests each validation rule with valid/invalid inputs
- Tests warning messages are emitted
- Tests strict mode raises errors

### Priority 4: Performance ✅ COMPLETE
| Fix | Approach | Status |
|-----|----------|--------|
| Selector parsing cache | LRU cache on `parse_selector()` | ✅ Implemented |
| OR selector deduplication | Dedupe before evaluation | ✅ Implemented |

**Implementation Details:**
- Added LRU cache (256 entries) to `parse_selector()` in `natural_pdf/selectors/parser.py`
- Returns deep copies to prevent mutation of cached values
- Added `clear_selector_cache()` and `get_selector_cache_info()` utilities
- OR parts deduplicated while preserving order (first occurrence wins)
- Tests in `tests/test_selector_cache.py` (16 tests, all passing)

---

## Completed Work

### Phase 1: Architecture Mapping ✅
- Mapped ~201 Python files across ~20 packages
- Documented core structure: PDF → Page → Region → Element
- Identified service architecture: PDFContext → ServiceNamespace → 16+ Services

### Phase 1b: Engine Provider Unification ✅
- OCRFactory now delegates to EngineProvider
- Removed QA global singleton
- Removed Layout double-cache

### Phase 2: Pattern Consistency Audit ✅
Found 36 inconsistencies across OCR engines, elements, layout detectors, services.

### Phase 2b: Pattern Consistency Fixes ✅

| Fix | Status |
|-----|--------|
| Paddle is_available() returns bool | ✅ |
| Gemini is_available() checks openai | ✅ |
| DocTR stores dimensions in instance | ✅ |
| extract_text uses apply_exclusions | ✅ |
| Color conversion utility added | ✅ |
| Model return types documented | ✅ |
| Logger added to SuryaOCREngine | ✅ |

### Remaining Low Priority
- Type property unification (hardcoded vs dynamic) - minor, defer

---

## Reference

### Key Architecture
```
PDF → Page → Region → Element (hierarchy)
     ↓
  PDFContext → ServiceNamespace → 16+ Services
     ↓
  EngineProvider → OCR/Layout/Tables/Guides engines
```

### Model IDs for Consensus
- GPT 5.2: `gpt-5.2`
- Gemini 3 Pro: `gemini-3-pro-preview`
- Opus 4.5: `anthropic/claude-opus-4.5`
