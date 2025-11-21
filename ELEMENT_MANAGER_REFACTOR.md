# ElementManager Refactor – Final Plan

This document merges the peer-advisor analysis (“V2”) with the alternative execution plan so we have one definitive reference. V2 contributes the deep current-state understanding, compatibility constraints, and OCR questions; the alternative plan supplies the concrete architecture, transitional steps, and testing/concurrency design. Together they describe why we are refactoring, what we are building, and how we will ship it safely.

---

## 1. Current State Analysis (from V2)

### 1.1 Loading Pipeline (simplified)

```
ElementManager.load_elements():
 1. Guard: prevent re-entrant loads
 2. Load raw char dicts from pdfplumber (includes font data, bbox)
 3. Detect decorations on char dicts (strike, underline, highlight)
 4. Group chars into words (line grouping, direction detection, BiDi)
 5. Propagate decorations from chars → words
 6. Load other shapes (rects, lines, images)
 7. Instantiate TextElement/RectangleElement/… objects
 8. Populate internal caches
```

Key observations:
- Steps are sequential/interdependent (decorations must run before word extraction, etc.).
- The manager currently owns storage, concurrency guards, transformation logic, OCR mutation hooks, and the public API.
- Testing individual pieces is difficult because everything lives in one method.

### 1.2 OCR Open Questions

Before coding we must lock down answers for:
1. Do OCR-generated elements get decoration detection?
2. Does appending OCR text invalidate caches (words, lines, regions)?
3. Can OCR be invoked before the first `load_elements()`?
4. Can OCR be applied multiple times and still remain consistent?
5. How do OCR mutations interact with BiDi/word grouping?
6. Do OCR additions trigger downstream services (e.g., selectors)?

These questions are addressed in §2.4 via the callback/invalidation design, but they remain non-negotiable requirements.

### 1.3 Public API Compatibility Matrix

| API (ElementManager) | Current Behavior | Requirement |
|----------------------|------------------|-------------|
| `load_elements()` | Lazy load, re-entrant guard | Semantics unchanged |
| `get_chars()`, `get_words()`, `get_rects()` | Return cached lists | Same return types & caching |
| `get_elements(kind)` | General fetch | Same interface |
| `remove_elements(predicate)` | Mutates caches | Behavior preserved |
| `replace_words_with_chars()` | Rebuilds words | Same signature/side effects |
| `apply_ocr_results()` / `create_text_elements_from_ocr()` | Adds OCR text | Same inputs/outputs |
| `disable_text_sync()` | Context manager for locking | Available & functional |
| `element_sync()` / `sync_elements()` | Invalidation helpers | No contract change |
| `clear_cache()` | Wipes caches | Same behavior |
| `get_metadata()` (if any) | Accessors | Same |
| `is_loaded(kind)` | Flag check | Same |
| `refresh_elements(kind)` | Force reload | Same |
| `set_elements(kind, list)` | Manual overrides | Same |
| `iter_elements()` | Iteration view | Same |

No existing call sites (internal or external) may break during the transition. `ElementManager` remains the façade for all consumers.

---

## 2. Proposed Architecture (from ALTERNATIVE)

### 2.1 Modules & Responsibilities

```
ElementManager (orchestrator)
  ├── ElementStore        # storage, locks, invalidation callbacks, versioning
  ├── WordEngine          # char→word grouping, direction detection, BiDi helpers
  ├── DecorationDetector  # underline/strike/highlight detection + propagation
  └── OCRConverter        # OCR result → TextElement creation + cache integration
```

- **ElementManager**: Coordinates the loading pipeline, exposes the public API, and sequences calls between modules. It should not contain pdfplumber manipulation logic once the refactor is complete.
- **ElementStore**: Owns `_elements`, lazy-load flags, concurrency (re-entrant lock), version counters, and callback registration so services (OCR, selectors) can listen for invalidation events.
- **WordEngine**: Stateless module that accepts char dicts, performs direction detection and BiDi processing, and returns deterministic word lists.
- **DecorationDetector**: Stateless analyzer that marks char dicts and propagates decorations to words; pluggable so future services or custom strategies can replace it.
- **OCRConverter**: Converts OCR service output into TextElement/char entries, notifies the store via callbacks, and ensures caches remain consistent.

### 2.2 Concurrency & Lifecycle

- ElementStore exposes `with store.transaction(kind): ...` to guard mutations. The transaction acquires a re-entrant lock, ensures lazy-load flags stay consistent, and increments a version counter.
- ElementStore maintains callback hooks (`store.register_callback(event, fn)`) so services can respond to invalidations (e.g., OCR additions triggering selector cache purge).
- OCRService and other hosts interact with the manager; the manager delegates conversion to OCRConverter and then calls `store.invalidate(range)` which triggers registered callbacks.
- WordEngine and DecorationDetector operate on copies/snapshots; they do not mutate shared state and require no additional locking.

### 2.3 Transitional States (safe delegation)

| Step | Description | Public API Impact |
|------|-------------|-------------------|
| T1 | Introduce ElementStore; ElementManager proxies all storage/caching calls. | None |
| T2 | Extract WordEngine (chars→words) and have manager delegate inside loader. | None |
| T3 | Extract DecorationDetector, inject via manager (allow overrides). | None |
| T4 | Extract OCRConverter; OCRService + manager delegate conversions/invalidation. | None |
| T5 | Slim ElementManager (remove dead helpers, keep orchestration only). | None |

At every step the manager retains existing methods; internals switch to the new modules.

### 2.4 OCR Behavior (addresses V2 questions)

1. **Decoration detection for OCR elements?** – No re-run; OCRConverter tags elements as `source="ocr"` and the store notifies interested parties. DecorationDetector can optionally re-run if configured, thanks to pluggable strategy.
2. **Cache invalidation?** – Yes. OCRConverter calls `store.invalidate("words")` (and any other affected kinds). Callbacks handle downstream services.
3. **OCR before `load_elements()`?** – Allowed. Manager ensures `load_elements()` runs (via transaction) before inserting OCR output.
4. **Multiple OCR passes?** – Supported. Each call increments store version, appends new elements, and triggers callbacks. Consumers can inspect versions to decide whether to rebuild derived data.
5. **BiDi/word grouping interactions?** – OCRConverter relies on WordEngine for any grouping, so behavior matches native extraction.
6. **Downstream services?** – Register callbacks (`store.register_callback("words", selector.on_elements_changed)`), ensuring all consumers get notified when OCR modifies data.

### 2.5 Testing Strategy

**Fixtures (real PDFs, no mocks for spatial/text behavior):**
1. `pdfs/01-practice.pdf` – Baseline (mixed fonts, multi-column). Covers loader + word engine.
2. `pdfs/highlights-sample.pdf` – Underlines/strikethrough/highlights. Covers decoration detector.
3. `pdfs/rtl-text.pdf` – Arabic/Hebrew text for BiDi coverage (needs creation if absent).
4. `pdfs/ocr-mixed.pdf` – Pages with OCR output (needs creation if absent).

**Approach:**
- Snapshot testing: serialize element counts/ordering/bboxes to snapshots so diffs are obvious. Auto-update snapshots when intentionally changing behavior.
- `pytest -m realpdf` marker for heavy tests. Default unit suite keeps light mocks; nightly/CI pipeline runs full real-PDF tests.
- Compatibility suite (from V2) that exercises the 14 key methods against known outputs to ensure no regressions.

### 2.6 Behavioral Success Criteria

1. Parallel-safe loading (transactions + locks prevent races).
2. OCR invalidation via ElementStore callbacks (no stale caches).
3. WordEngine exposes deterministic API (`group(chars) -> List[TextElement]`).
4. Decoration detection is pluggable (strategy injection works).
5. ElementManager methods stay ≤ ~50 lines and contain orchestration only.
6. Performance regression ≤ 105% of current load time on `01-practice.pdf`.
7. Compatibility suite passes unchanged.

---

## 3. Execution Plan & Status

### 3.1 Timeline & Phases

- **Phase 0 – Infrastructure (✓ Completed):**
  - Real PDF fixtures checked in (`practice`, `shapes`, `arabic`, `needs_ocr`).
  - Snapshot harness + `pytest -m realpdf` suite in `tests/realpdf/`.
  - Benchmark script added (`tools/benchmarks/load_elements.py`); latest mean ≈0.051 s for `01-practice.pdf` (5 iterations).

- **Phase 1 – ElementStore (✓ Completed):**
  - Store owns locking, versioning, invalidation callbacks, and exposed setters.
  - ElementManager now delegates all cache mutations through the store.

- **Phase 2 – WordEngine (✓ Completed):**
  - Word grouping (tolerances, BiDi, per-line direction) lives in `core/word_engine.py`.
  - ElementManager passes creation/propagation callbacks into the engine.

- **Phase 3 – DecorationDetector (✓ Completed):**
  - Strikethrough/underline/highlight detection moved to `core/decoration_detector.py`.
  - Word-level propagation handled by the detector, not the manager.

- **Phase 4 – OCRConverter (✓ Completed):**
  - OCR payload processing extracted to `core/ocr_converter.py`.
  - `create_text_elements_from_ocr()` just calls the converter and updates the store.

- **Phase 5 – Cleanup & Docs (in progress):**
  - Expose ElementLoader/DecorationDetector through `Page` so Regions, collections, and
    manifest importers reuse the same enrichment logic when synthesizing text elements.
  - Update architecture docs, remove dead helpers, re-run compatibility + benchmark.

Total estimate: ~3 weeks after Phase 0 (similar to V2’s timeline but achievable with the concrete module split).

### 3.2 Deliverables / Checklist

- [x] Real PDF fixtures checked into repo (or scripts to generate).
- [x] Snapshot harness + `pytest -m realpdf` target.
- [x] ElementStore module with transaction API, callbacks, versioning.
- [x] WordEngine module + tests (RTL fixtures included).
- [x] DecorationDetector module + tests (highlight fixture).
- [x] OCRConverter module + updated OCRService integration.
- [x] ElementManager reduced to orchestration; methods updated.
- [x] Regions/collections/importers access shared ElementLoader + DecorationDetector for new text.
- [x] Compatibility suite updated/passing; performance benchmark script.
- [ ] Documentation updates (this plan, ARCHITECTURE.md, module READMEs).

### 3.3 Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Fixture creation stalls timeline | Allocate time in Phase 0; leverage existing datasets or scripts (ReportLab) to generate PDFs. |
| Partial migration breaks callers | Transitional plan keeps ElementManager APIs intact; compatibility suite run after each phase. |
| Performance regressions | Benchmark script per phase; limit regression to ≤5%. |
| OCR semantics drift | Callback-based invalidation guarantees consistent behavior; document behavior in OCRService guide. |
| Test suite slowdown | Gate heavy tests under `realpdf` marker; run nightly CI. |

---

## 4. Next Steps

1. Approve this merged plan.
2. Schedule/execute Phase 0 (fixtures, compatibility suite, benchmarks).
3. Proceed through Phases 1–5, checking the deliverables and success criteria at each gate.

With this combined document we retain V2’s rigor (clear requirements and compatibility) while executing with ALTERNATIVE’s practical architecture and testing strategy. ElementManager stays the ergonomic façade users rely on, but its internals become modular, testable, and resilient.***
