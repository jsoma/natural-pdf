# Architecture Overview

This guide explains how Natural PDF's service-oriented architecture is structured,
so contributors know where new code belongs and how to extend existing features.

## Core Principles

1. **Service-first design**: business logic lives in services, hosts stay thin.
2. **Reusable operations**: shared utilities (element loader, text operations) live in
   dedicated modules so Page/Region/Collection can reuse them.
3. **Orchestrators, not god objects**: components like `ElementManager` coordinate
   specialized modules rather than implementing logic inline.

## Host vs Service Responsibilities

| Host | Responsibility | Example |
|------|----------------|---------|
| `PDF` / `Page` / `Region` | Provide ergonomic APIs, maintain context | `page.extract_table()` delegates to TableService |
| Service (`services/*.py`) | Orchestrate complex logic, manage dependencies, register delegates | `ClassificationService.classify()` |
| Operations modules | Pure functions reused by multiple hosts/services | `text/operations.py` for layout filtering |

### Pattern: Thin Wrapper

```python
class Page(ServiceHostMixin):
    def classify(self, labels, model=None, **kwargs):
        """User-friendly API."""
        return self.services.classification.classify(
            self, labels=labels, model=model, **kwargs
        )
```

### Pattern: Service + Operations

```python
class ClassificationService:
    @register_delegate("classification", "classify")
    def classify(self, host, labels, model=None, **kwargs):
        engine = get_classification_engine(model)
        return engine.classify_item(host, labels, **kwargs)
```

Operations modules (e.g., `classification/pipelines.py`) contain pure functions that
implement the algorithm so they are easy to test and reuse.

## Element Loading Architecture

```
Page / Region / Collection
        │
        ▼
 ElementManager (orchestrator)
        │  ┌────────────────────────────┐
        │  │ ElementStore (storage +    │
        │  │ locks + callbacks)        │
        │  └────────────────────────────┘
        │          │
        │  ┌───────┴────────┐
        │  │ Specialized    │
        │  │ modules        │
        ▼  ▼                ▼
 ElementLoader   WordEngine   DecorationDetector
        │             │                 │
 OCRConverter  Text operations   Packaging/Regions reuse
```

* `ElementLoader` enriches raw pdfplumber char dicts with bold/italic/etc. Hosts that
  synthesize text (Regions, manifest import) resolve it via `page._get_element_loader()`.
* `DecorationDetector` marks underline/strike/highlights and propagates to words. It is
  shared the same way through `page._get_decoration_detector()`.
* `ElementStore` provides thread-safe storage with callbacks so services can react to
  invalidation events.

## When to Add a Service vs Operation

- **Service**: when the feature depends on host context, other services, or state
  (classification, structured extraction, rendering, OCR).
- **Operations module**: when logic is pure and shared between hosts (text layout
  filtering, structured extraction pipelines).
- **Host helper**: only for core identity operations (text extraction) that need to
  stay on the host for ergonomics but should rely on operations modules.

## Adding a New Capability

1. Decide whether it belongs in a service (most features) or an operations module.
2. Create `services/my_capability_service.py` with a class inheriting `BaseService`.
3. Register delegates with `@register_delegate("capability", "method")` so host
   wrappers are attached automatically when appropriate.
4. Implement heavy logic in a separate module (e.g., `capability/operations.py`).
5. Update the host wrapper to call `self.services.capability.method(self, ...)`.
6. Document the new capability in this guide and in feature-specific docs.

## Migrating Legacy Helpers

Older helper methods (e.g., `_create_char_elements`) have been removed. If you find code
that still needs similar behavior:

- Move the logic into `ElementLoader`, `DecorationDetector`, or a new operations module.
- Expose helpers through the host (like `_get_element_loader`) so Regions/Collections
  can reuse them instead of re-implementing.
- Prefer services over adding new manager methods.

## Testing Expectations

- Services: test in isolation by instantiating the service with a `PDFContext` and a
  mock host object.
- Operations modules: unit tests with pure inputs/outputs.
- Element loading: real-PDF tests under `pytest -m realpdf` cover `ElementManager`,
  loader, word engine, and decoration detector behavior.

## Additional Reading

- [`ARCHITECTURE.md`](../ARCHITECTURE.md) – internal design notes
- [`ELEMENT_MANAGER_REFACTOR.md`](../ELEMENT_MANAGER_REFACTOR.md) – detailed plan for
  the element loading refactor
- Service-specific guides (e.g., `docs/tables/`, `docs/ocr/`) for feature usage
