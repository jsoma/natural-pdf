## Composition Refactor Plan

### Why We’re Moving Away from Mixins
- Hundreds of methods are inherited via a deep MRO lattice (`Page`, `Region`, `FlowRegion`, collections, etc.), which hides dependencies, makes testing difficult, and couples every class to every capability.
- We need clearer boundaries so OCR, navigation, extraction, etc. can evolve independently (swappable implementations, better DI, easier mocking).
- Composition lets each capability live in a self‑contained service while the public API (`page.below()`, `region.apply_ocr()`) remains unchanged via generated delegates.

### Target Architecture
1. **Service Interfaces**
   - Define explicit service classes (e.g., `NavigationService`, `OCRService`, `ExtractionService`, `CollectionService`).
   - Each service consumes only the protocols it needs (`SupportsGeometry`, `SupportsSections`, `HasConfig`, etc.) plus config/registry references.
   - Host types (Element, Region, FlowRegion, Page, PageCollection, ElementCollection, etc.) wire the subset of services they require.

2. **Service Registry & Delegates**
   - Maintain a mapping from public method name → `(service_attr, service_method, host_types)`.
   - During class definition, auto‑generate delegates such that `Page.below` simply fetches `self._services.navigation` and calls `below(self, …)`.
   - Detect collisions early (e.g., two services claiming `extract_table`) to avoid silent overrides.
   - Because the delegates are real methods, IDE autocomplete and type checking continue working as before.

3. **Protocol Refinement**
   - Reuse existing protocols where possible, but consolidate variants that say the same thing (e.g., `SupportsSections` vs. other section traits).
   - Introduce new protocols if a service needs guarantees not currently captured (e.g., `HasFlowContext` for FlowRegion‑specific logic).

4. **Service Implementations**
   - First pass: move today’s mixin logic into the corresponding service with minimal changes (essentially a relocation).
   - Later iterations can split by host type if needed (e.g., `NavigationForRegion` vs. `NavigationForFlowRegion`) while exposing a unified service interface.

5. **Lifecycle & Context**
   - Hosts construct services lazily and cache them in `self._services`.
   - External dependencies (engine providers, configs) come from a context object so we can eventually remove singletons.
   - Allow injecting alternate service implementations for advanced scenarios/tests.

### Migration Strategy
1. **Inventory & Mapping**
    - List every mixin method and decide which service owns it.
    - Flag any method that is host‑specific or has conflicting semantics across hosts (requires per‑host service flavor).

2. **Implement Services Behind the Scenes**
   - For each host type, instantiate the necessary services (e.g., `Page` gets navigation, OCR, extraction; `ElementCollection` gets batch classification, export).
   - Generate delegates so public APIs continue to work.
   - Keep mixin classes temporarily, but deprecate them and have their methods call into the new services for backward compatibility during the transition.

3. **Testing & Rollout**
   - Add unit tests per service and integration tests per host to ensure parity.
   - Once parity is confirmed for a host type, drop the mixin inheritance and rely solely on services.
   - After all hosts migrate, remove the mixin modules entirely.

4. **Protocol Consolidation**
   - As part of each service migration, note duplicated or redundant protocols and merge them (e.g., unify “has pages” traits, collapse multiple text/section interfaces).
   - Update type hints so services rely on the consolidated protocols.

### Recent Decisions / Work in Flight
- **Exclusion ergonomics**: the service will fall back to creating regions from any object that exposes a bbox (using `extract_bbox`) if the host doesn’t implement `_element_to_region`. We’ll also consolidate the accepted-type documentation inside the service so callers get a single, authoritative error message.
- **Guides/table UX**: since every host now exposes `extract_table`, the guides helper will simply delegate to that method (or the table service) without special-casing pages vs. regions. This keeps the logic in one place and surfaces a single error when a host lacks table capability.
- **Shared docstrings for table helpers**: we’ll keep the canonical docstrings next to the shared helper functions (e.g., `table_methods.extract_table`). Host methods should point to those helpers or copy their `__doc__` so edits happen in one place. If we find ourselves touching the signatures often, we can generate the wrappers automatically from the delegate registry later.

### Open Questions / Items to Clarify
1. **Context Object Scope**
   - ✅ Decision: introduce `PDFContext` as part of this refactor. While we’re already touching every host to wire services, we’ll also thread a context that carries engine registries, config, caches, and shared resources. This lets us dismantle globals immediately instead of planning another disruptive sweep later.

2. **Service Granularity**
   - We want to avoid `isinstance` checks in the service layer. Plan:
     - For behaviors with overlapping semantics, define a shared protocol (`SupportsDirectionalOps`) and implement a single service that relies only on that protocol, optionally dispatching to host-specific helpers internally.
     - When hosts diverge significantly (e.g., `FlowRegion` vs. `Region`), keep the same service but branch via strategy hooks; only create separate service classes when overlap drops below ~50 % and the branching would dominate the implementation.
   - We’ll explore protocol-based dispatch where possible and only split when semantics differ materially.

3. **Custom Extensions**
   - Primary mechanism: **context injection**. `PDFContext` exposes factory hooks for each service so consumers can provide custom implementations per document/test without touching globals.
   - We can layer an optional registry later if we ever need process-wide plugin defaults, but it’s not required for the initial refactor.
   - Trait interfaces remain useful as documentation for what a service must implement.
   - Context defaults will eventually read from the existing global option objects so behavior remains consistent for users who don’t pass an explicit context, but explicit contexts remain the recommended path.

4. **Performance Costs**
   - Keep services internal (no public `.navigation` attributes) and favor small surface area.
   - Use the context to cache heavy services per document (e.g., OCR manager) and share them across hosts where appropriate (page, regions derived from that page).
   - Document the sharing strategy so we don’t regress performance when everything becomes compositional.

5. **Public API Exposure**
   - Decision: services remain internal for now. The generated delegates are the only supported surface. Once the architecture stabilizes we can revisit exposing service handles if there’s demand.

6. **Protocol Cleanup Plan**
   - “Everything is up for grabs” since we haven’t released the new version. We’ll merge redundant protocols first (e.g., consolidate section/text traits), then compose services around the cleaned-up interfaces.
   - After the merge, ensure each service documents which protocol it requires so future refactors don’t introduce accidental coupling.

7. **Delegate/Registry Interface**
   - Use decorator-based registration on service methods (e.g., `@register_delegate("navigation", "below")`) so the mapping from public API → service is centralized, self-documenting, and can detect collisions up front. Host classes consume this registry when generating delegates.

### Status & Next Steps

#### Completed Work
- `PDFContext` + `ServiceHostMixin` now back every host, and the decorator-based delegate registry drives all attached capabilities.
- Text/navigation delegates are wired for `PDF`, `Page`, `Region`, `Element`, and flow constructs; FlowRegion/Flow collections rely on NavigationService helpers instead of bespoke loops.
- Table extraction (including guides, FlowRegionCollection, explicit grid paths) runs entirely through `TableService`, so partial/full-guide workflows use the same implementation as raw regions.
- OCR, extraction, and classification are routed through their dedicated services with host-specific hooks only where rendering differs.
- QA was moved behind `QAService`, which manages blank results, normalization, and FlowRegion aggregation via `_qa_segments`. FlowRegionCollection now inherits `ServiceHostMixin`, exposes the QA hooks, and delegates `ask`/confidence ranking through the same service. Mixins only provide lightweight hooks (page numbers, normalization helpers).
- QA was moved behind `QAService`, which manages blank results, normalization, and FlowRegion aggregation via `_qa_segments`. FlowRegionCollection now inherits `ServiceHostMixin`, exposes the QA hooks, and delegates `ask`/confidence ranking through the same service. Mixins only provide lightweight hooks (page numbers, normalization helpers). Flow instances also bind their context and surface the QA hooks so `flow.ask()` goes through the same delegate path instead of a bespoke FlowRegion shim.
- Page collections and `PDF.ask` now ride through the same QA pipeline: `PageCollection` exposes the QA hooks/delegate, and the PDF extractive path simply instantiates a scoped collection so service logic (and confidence ranking) stays centralized while generative mode keeps its LLM-specific implementation.
- `PDFCollection` inherits `ServiceHostMixin`, wires the QA hooks, and forwards to `QAService` as well, so `pdfs.ask()` works just like `PageCollection.ask()` (segmenting by constituent pages) without bespoke loops.
- Guides go through `GuidesService`; `Page`, `Region`, and `FlowRegion` expose a `.guides()` helper that instantiates analyzer instances via the shared context instead of manual imports.
- Layout analysis is handled by `LayoutService`; `Page`, `Flow`, `PageCollection`, and `PDF` delegate `analyze_layout` via the service, which wraps `LayoutAnalyzer`, de-duplicates flow segments, and honors the existing caching semantics.
- Table extraction is entirely service-driven now; Region hosts attach the `"table"` capability directly, so the old `TableExtractionMixin` is no longer part of the capability bundle (it remains as a shim for third-party subclasses).
- Selector, single-page context, and table responsibilities are explicit on the host classes; the old `AnalysisHostMixin`/`TabularRegionMixin` shims have been removed.
- Exclusions now flow through `ExclusionService`, so `Page`/`Region`/`FlowRegion` share the same `add_exclusion`/evaluation logic and the old mixins have been removed from `AnalysisHostMixin`. FlowRegion performs local+constituent aggregation with service-provided helpers.
- `PDFContext` exposes per-capability options via `get_option`, so services (starting with selectors) no longer reach into host configs directly.
- Selector lookups are centralized in `SelectorService`; `Page`, `Region`, `Flow`, `FlowRegion`, `PDF`, `PageCollection`, `PDFCollection`, and individual `Element`s now delegate `find`/`find_all` through the service, which reuses the page-level query engine and host-specific hooks for overlap filtering, multi-page flows, and collections.
- Describe/inspect now run through `DescribeService` and are exposed via delegates on `Page`, `Region`, `Element`, `ElementCollection`, and `PageCollection`, allowing us to drop the old mixin inheritance.
- Visual search moved to `VisualSearchService`; `Page`, `PDF`, `PageCollection`, and `PDFCollection` attach a `"vision"` capability so `match_template`/`find_similar` call through the shared implementation rather than inheriting `VisualSearchMixin`.
- Shape detection now routes through `ShapeDetectionService`, with delegate hooks on `Page`, `Region`, `PDF`, `PageCollection`, and `PDFCollection`. The legacy mixin only provides helper implementations for the service proxy.
- Checkbox detection likewise runs through `CheckboxDetectionService`, and hosts previously inheriting `CheckboxDetectionMixin` now use the capability wiring instead of mixins.
- Structured extraction and classification both use their services exclusively. `Page`, `Region`, `PDF`, and `Element` attach the `extraction`/`classification` capabilities so the old mixins are no longer part of `AnalysisHostMixin`, and ElementCollection relies on the services (with the batch helper updated to check for the requisite hooks).
- Delegate registry tests plus targeted pytest suites (text update, navigation defaults, flow anchors, guides/tables, QA provider) are passing after each migration.

#### Upcoming Focus
1. **Analyzer Services**
   - Guides, layout, describe/inspect, visual search, tables, etc. are now service-driven. Future analyzer work is limited to new features rather than mixin rewrites.
2. **Flow/Collection Consistency**
   - Decide whether additional capabilities (tables/exclusions/text) should be exposed directly on flow/PDF collection hosts or continue forwarding through underlying regions.
3. **Protocol Cleanup**
   - After services own the remaining behaviors, collapse redundant protocols and document the exact contracts each service consumes.
4. **Docs & Tests**
   - Expand service-specific tests and update docs once the analyzer mixins are fully retired.

This roadmap keeps us marching toward a purely service-driven architecture—exclusions/selectors are next, followed by the remaining analyzer mixins and protocol cleanup.
