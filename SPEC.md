# Engine Provider & Plugin Spec

## Goal
Unify layout/OCR/table extraction engines (and future capabilities) behind a single provider/registry so core APIs stay simple, plugins can extend functionality, and legacy method strings like `extract_tables("lattice")` keep working.

## Core Concepts
- **Capability**: Named slot such as `layout`, `ocr`, `tables`, `selector.aggregate`. Each capability defines the method signature engines must implement (e.g., layout engines expose `detect(context, **opts)`).
- **EngineProvider**: Global registry/factory with methods like `register(capability, name, factory, metadata=None)` and `get(capability, context, name=None, **opts)`. Core systems call `provider.get(...)` exactly where substitution is needed.
- **Context**: The object invoking the engine (Page, Region, Flow, etc.). Methods set `context=self` automatically so users never pass it explicitly. Engines can walk up to parent PDF, read config, and even call other engines.
- **Entry Points**: Plugins declare entry points (eagerly loaded at import) under groups like `natural_pdf.engines.layout`. Each entry point calls `register(provider)` and can also patch public methods if the capability is non-standard (e.g., `describe_images`).

## Naming & Config
- Capability keys keep a single word for now (`layout`, `ocr`, `tables`). If we introduce non-engine features later we can prefix (`selector.aggregate`).
- Engine names match existing method strings (`"lattice"`, `"tatr"`, etc.). Built-ins register those aliases; plugins register additional names.
- Config precedence stays as today: per-call kwargs > page/PDF config (`get_config` up the context tree) > global options.

## Lifecycle
- Engines are cached globally per capability/name (no per-context duplication unless the engine itself chooses to). Provider handles lazy instantiation + reuse.
- Optional dependencies are treated lazily: engine factories import heavy libs on demand; provider loads all entry points eagerly so capabilities are discoverable.
- No special `EngineError` wrapper—engines raise normal exceptions. If a requested engine name isn’t registered, provider raises a clear error explaining how to install/enable it.

## Public API expectations
- Methods like `page.analyze_layout(engine="tatr")`, `page.extract_tables(method="lattice")`, `region.apply_ocr(engine="surya")` simply pass the name through to the provider. Legacy names remain supported by registering them as default engines.
- For optional features (e.g., `page.describe_images()`), plugins add the public method themselves and internally call the provider.
- Core objects (Page, Region, FlowRegion, Flow, PDF) will be wired up once so their built-in methods all route through `provider.get(...)`; plugins can still patch additional helper methods on any host after registering their engines if they need custom entry points.
- Built-in/legacy engines register themselves through the same provider so external plugins get identical treatment—no “special” branches inside Page/Region. All capability logic (including ours) flows through the registry, making swapping/extending trivial.

## Registration Helpers
- `natural_pdf.engine_registry` centralizes registration helpers so contributors don’t have to import individual provider modules. IntelliSense now surfaces the important helpers in one place.
- Available functions:
  - `register_engine(capability, name, factory, *, replace=True, metadata=None)` – lowest-level hook for any capability.
  - `register_table_engine(name, factory)` and `register_table_function(name, func)` – for full engines and simple callable-based table extractors. `natural_pdf.tables` re-exports both helpers. Use `register_structure_engine(name, factory)` for `tables.detect_structure`.
  - `register_guides_engine(name, factory)` – attaches to `guides.detect`; also re-exported via `natural_pdf.guides`.
  - `register_ocr_engine(name, factory, capabilities=("ocr", "ocr.apply", "ocr.extract"))` – registers an OCR engine across all OCR capabilities; re-exported from `natural_pdf.ocr`.
  - `register_layout_engine(name, factory)` – registers layout detectors; re-exported via `natural_pdf.analyzers.layout`.
  - `register_classification_engine(name, factory)` – attaches to the `classification` capability; re-exported from `natural_pdf.classification`.
  - `register_qa_engine(name, factory)` – wires engines into `qa.document`; re-exported from `natural_pdf.qa`.
  - `register_deskew_engine(name, factory, capabilities=("deskew", "deskew.detect", "deskew.apply"))` – registers a deskew engine across all deskew entry points; re-exported via `natural_pdf.deskew`.
  - `register_selector_engine(name, factory)` – registers selector providers (`selectors` capability); re-exported from `natural_pdf.selectors`.
- Example (functional table engine):
  ```python
  from natural_pdf.tables import register_table_function

  def table_delim(region, context=None, **kwargs):
      ...  # return TableResult or list of rows

  register_table_function("table_delim", table_delim)
  ```

## Capability Contracts
Each capability must document its method signature and return shape so Page/Region/Flow can treat every engine uniformly.

### Tables
- **Registration names**: Built-ins register `pdfplumber_auto`, `pdfplumber`, `stream`, `lattice`, `tatr`, `text`. Plugins can add new names; callers reference them via `method=...`/config.
- **Engine interface**: `extract_tables(*, context, region, table_settings=None, **kwargs) -> List[List[List[str]]]`. Engines receive the initiating context (Page/Region/etc.), plus extra kwargs the host forwards (`use_ocr`, `ocr_config`, `text_options`, `cell_extraction_func`, `show_progress`, `content_filter`, `apply_exclusions`, etc.). They return a list of tables (each table = list of rows = list of cell strings/None).
- **Single-table helpers**: `Region.extract_table` (and Page/Flow wrappers) always call the provider, then pick the “primary” table (biggest rows×cols). No engine-specific branches remain—`tatr` and `text` run via their provider registrations.
- **Settings handling**: Hosts pass a normalized copy of `table_settings` so engines are free to mutate without affecting callers.

### OCR
- **Separate capabilities**: We expose two related capabilities so engines can return distinct payloads without ambiguity:
  - `ocr.apply`: Mutates the host’s element store (e.g., adds OCR-derived `TextElement`s) and returns whatever metadata the engine deems useful (confidence summaries, stats, etc.).
  - `ocr.extract`: Returns OCR results without touching the element store (used by `extract_ocr_elements`/“preview” flows).
- **Engine interface**:
  - `apply_ocr(*, context, region_or_page, options=None, **kwargs) -> Any`
  - `extract_ocr(*, context, region_or_page, options=None, **kwargs) -> List[Dict[str, Any]] | List[List[Dict[str, Any]]]`
  Hosts pass along whatever supplemental kwargs they currently support (`resolution`, `languages`, `min_confidence`, `device`, etc.). Engines may ignore extras they don’t need.
- **Rendering responsibility**: Engines control how input data is prepared. We provide the context (Page/Region/etc.) plus any hints (requested DPI), but engines decide whether to render images themselves, request vector data, or operate on other modalities.
- **Concurrency**: Engine implementations own their locking/concurrency strategy. The provider simply hands out engine instances; if an engine needs per-device serialization, it must enforce that internally.

### Layout
- **Capability**: `layout`
- **Registration**: `natural_pdf.analyzers.layout.register_layout_engine("yolo", lambda **_: YOLODocLayoutDetector())`
- **Engine interface**: Engines implement the `LayoutDetector` protocol (see `natural_pdf.analyzers.layout.base`) and must expose methods like `detect(self, page, options=None)` returning layout elements. Hosts resolve engines via `provider.get("layout", ...)` and call `detector.detect(page, options=LayoutOptions(...))`.
- **Options**: Engines may define custom `LayoutOptions` dataclasses; callers pass them directly through when invoking `Flow.analyze_layout`/`Page.analyze_layout`.

### Deskew
- **Capabilities**: We expose `deskew.detect` (returns angle only) and `deskew.apply` (returns a deskewed image, optionally reporting the angle used). Both receive the same context+target data so engines can share logic. Legacy helpers (`detect_skew_angle`, `deskew`) simply call these capabilities via the provider.
- **Engine interface**:
  - `detect(*, context, target, resolution=72, grayscale=True, **kwargs) -> Optional[float]`
  - `apply(*, context, target, resolution=300, angle=None, detection_resolution=72, grayscale=True, **kwargs) -> DeskewApplyResult`
    where `DeskewApplyResult` at least includes the output `Image.Image` plus the angle actually used.
  Hosts forward kwargs such as `deskew_kwargs` (passed through to deterministic engines), and engines decide how to render/convert images (e.g., grayscale, numpy arrays) before running their detection logic.
- **Rendering & dependencies**: Engines control rendering through the supplied `target` (Page/Region/etc.) so different implementations can request the DPI they need. Optional dependencies (e.g., the `deskew` library) live inside the engine; if missing, the engine raises an informative ImportError.
- **Caching**: Hosts remain responsible for storing derived angles (e.g., Page caches `_skew_angle`). Engines do not persist state unless they choose to.
- **Registration**: `natural_pdf.deskew.register_deskew_engine("my-deskew", lambda **_: MyDeskewEngine())`.

### Classification
- **Capability**: `classification` engines receive content (text or image) plus label/mode metadata and return a `ClassificationResult`. The built-in engine wraps Hugging Face zero-shot pipelines via the service layer, and everything is routed through `provider.get('classification', …)` so external engines can drop in.
- **Engine interface**:
  - `infer_using(model_id: Optional[str], using: Optional[str]) -> str`
  - `default_model(using: str) -> str`
  - `classify_item(*, item_content, labels, model_id, using, min_confidence, multi_label, **kwargs) -> ClassificationResult`
  - `classify_batch(*, item_contents, labels, model_id, using, min_confidence, multi_label, batch_size, progress_bar, **kwargs) -> List[ClassificationResult]`
- **Hosts**: `Page`, `Region`, `PDF`, `ElementCollection`, etc. gather the appropriate content (text or rendered image) and call `run_classification_item/batch`. They no longer instantiate managers directly or reference `pdf.get_manager('classification')`.
- **Registration**: `natural_pdf.classification.register_classification_engine("my-classifier", lambda **_: MyClassificationEngine())`.

### Document QA
- **Capability**: `qa.document` engines accept a region plus question metadata and return one or more `QAResult` objects. Hosts (Page, Region, FlowRegion) simply call `run_document_qa` with their `_qa_target_region()` and let the engine decide how to render/crop.
- **Engine interface**: `ask_region(*, region, question, min_confidence, debug, **kwargs) -> QAResult | List[QAResult]`. Engines manage model loading internally, so swapping HF models or external services is just a matter of registering a new name.
- **Registration**: `natural_pdf.qa.register_qa_engine("my-qa", lambda **opts: MyQAEngine(**opts))`.

### Guides
- **Capability**: `guides.detect` engines produce horizontal/vertical guide coordinates. Hosts (GuidesList/Guides) specify the axis and method (`content`, `lines`, `whitespace`, etc.) plus contextual options (markers, alignment, thresholds). Engines return a coordinate list which the Guides object uses for storage/caching.
- **Engine interface**: `detect(*, axis, method, context, options) -> GuidesDetectionResult`. The default engine wraps the legacy `Guides.from_*` helpers so existing behaviors stay intact while enabling plugins to register new detectors or override built-ins.
- **Registration**: `natural_pdf.guides.register_guides_engine("my.guides", lambda **_: MyGuideEngine())`.

### Selectors
- **Capability**: `selectors`
- **Engine interface**: `query(*, context: SelectorContext, selector: str, options: SelectorOptions) -> SelectorResult`. Engines receive the host (Page/Region/etc.) via the `SelectorContext` wrapper and must return an `ElementCollection`. `SelectorOptions` includes parsed selector metadata plus tolerance/regex/case flags that mirror our native implementation.
- **Registration**: `natural_pdf.selectors.register_selector_engine("my.selector", lambda **_: MySelectorEngine())`.
- **Usage**: When `engine=` is passed to `find/find_all`, Page/Region resolves the requested selector engine and calls `query(...)`. Engines can inspect `context.parse(selector)` to reuse the built-in parser if desired.

## Plugin UX
- Implement the capability protocol (`detect`, `run`, etc.).
- Export `register(provider)` that calls `provider.register("layout", name, factory)`.
- Declare entry point in `pyproject.toml` so natural-pdf loads it at import time.
- If needed, patch objects with new methods during registration.

## Next Steps
1. Implement `EngineProvider` and load entry points eagerly.
2. Hook layout/OCR/table extraction into the provider while keeping legacy strings mapped to registered engines.
3. Document capability contracts (method signatures) so plugin authors know what to implement.
4. Provide tooling (e.g., `natural_pdf.list_engines("layout")`) to aid discovery/debugging.
