# Engines and Models

Natural PDF's core operations — selectors, spatial navigation, text extraction, pdfplumber tables, vector guides — are pure geometry. They download nothing and run anywhere. Everything heavier goes through the **engine registry**: a table of named engines grouped by *capability* (what job they do), so `apply_ocr(engine="doctr")` and `analyze_layout("tatr")` are the same gesture pointed at different capabilities.

## The registry

```python
import natural_pdf
from natural_pdf.engine_registry import list_engines

for capability, engines in sorted(list_engines().items()):
    print(f"{capability:15} {', '.join(engines)}")
```

Engines register when their capability's module loads, so the list grows as you touch features — the `tables` capability appears the first time table machinery runs:

```python
pdf = natural_pdf.PDF("pdfs/01-practice.pdf")
pdf.pages[0].extract_table()

print(", ".join(list_engines()["tables"]))
```

Registering an engine costs nothing — it's a name and a factory. Models load (and download) only when an engine actually runs. Third parties can add engines with the `register_ocr_engine` / `register_layout_engine` / `register_table_engine` (etc.) helpers in `natural_pdf.engine_registry`.

## What downloads, when, and where

Nothing downloads at import. The first *use* of a model-backed engine fetches its weights, then caches them:

- **Hugging Face-hosted models** (most of them) land in the HF hub cache — `~/.cache/huggingface/hub` by default; set `HF_HOME` to move it. Delete a `models--…` directory there and the next run re-downloads it.
- **RapidOCR** stores its ONNX models inside the installed `rapidocr` package (about 28 MB for the default detection/recognition pair).

Approximate on-disk cache sizes, measured on a working install — treat as ballpark, model repos change:

| First call | Model | Cache size |
|---|---|---|
| `analyze_layout()` (yolo, the default) | `juliozhao/DocLayout-YOLO-DocStructBench` | ~40 MB |
| `analyze_layout('tatr')` / `method='tatr'` tables | `microsoft/table-transformer-detection` + `…-structure-recognition-v1.1-all` | ~220 MB + ~110 MB |
| `analyze_layout('doclayout')` | `PaddlePaddle/PP-DocLayoutV3_safetensors` | ~45 MB |
| `apply_ocr()` (rapidocr, the default) | PP-OCRv4 ONNX models | ~28 MB |
| `page.ask(...)` / `extract(engine="doc_qa")` | `impira/layoutlm-document-qa` | ~490 MB |
| `classify(..., using="text")` | `facebook/bart-large-mnli` | ~1.5 GB |
| `classify(..., using="vision")` | `openai/clip-vit-base-patch16` | ~1.1 GB |
| `pdf.search(...)` | `sentence-transformers/all-MiniLM-L6-v2` | ~170 MB |
| `apply_ocr(engine="glm_ocr")` | `zai-org/GLM-OCR` (4-bit MLX variant on Apple Silicon) | ~2.5 GB full / less for 4-bit |
| `extract(..., engine="vlm")` with no model | `Qwen/Qwen3-VL-2B-Instruct` | multi-GB (2B-parameter VLM) |

The pattern to internalize: **selectors and geometry are free; the first OCR/layout/QA/classification call on a fresh machine will sit in a download.** In containers and CI, pre-warm or mount the HF cache.

## The capabilities

**OCR** — `page.apply_ocr(engine=...)`. Classic engines: `rapidocr` (default), `easyocr`, `surya`, `paddle`, `doctr`, `chandra2`; each has an install hint baked in (e.g. `pip install python-doctr`) that surfaces in the error when the dependency is missing. `paddlevl` auto-picks its variant per platform (MLX on Apple Silicon). VLM shorthands `glm_ocr`, `dots`, and `chandra` resolve a concrete local model for your platform, and `engine="vlm"` takes any `model=` and/or OpenAI-compatible `client=`.

**Layout** — `page.analyze_layout(engine)`: `yolo` (default), `tatr`, `paddle`, `surya`, `doclayout`, `vlm` (`gemini` is a deprecated alias for `vlm`). Each emits its own region-type vocabulary — see [Selectors](selectors.md).

**Tables** — `pdfplumber_auto`, `pdfplumber`, `stream`, `lattice` (no models, pure geometry), `tatr` (Table Transformer), `text`.

**Guides detection** — the built-in `lines` / `content` / `whitespace` / `headers` / `stripes` strategies behind `Guides.from_*` are themselves registered engines (`builtin.lines`, …), which is what makes them replaceable.

**Classification** — one `default` provider that routes `classify(labels, using="text")` to zero-shot BART-MNLI and `using="vision"` to CLIP; pass `model=` for anything else.

**Deskew** — three engines, no downloads (scipy/scikit-image math on the rendered page):

- `standard` / `projection` (same engine, two names): rotates a candidate sweep and scores horizontal-projection sharpness.
- `hough`: Canny edge detection plus a Hough line transform — sturdier when the page is mostly ruled lines or boxes rather than dense text. Tunable via `sigma`, `num_peaks`, `max_skew_deg`, `min_deviation_deg`.

```python skip=true
angle = page.detect_skew_angle(engine="hough")   # degrees of correction, 0.0 = straight
straightened = page.deskew(engine="hough")        # returns a PIL Image
```

(`engine="hough"` has existed since the deskew engines were internalized but this is its first appearance in the docs.)

**Extraction** — `page.extract(Schema)` / `page.ask(...)` resolve outside the registry, in `ExtractionService`: `doc_qa` (local LayoutLM, the default when you pass no client), `llm` (any OpenAI-compatible `client=`, required), and `vlm` (a local vision model through the shared VLM stack).

## Local vs. remote, and the `set_default_client()` contract

VLM-backed features can run two ways: **locally** (weights on your machine, pages never leave it) or **remotely** (page images sent to an OpenAI-compatible API). Which one you get follows one rule, implemented identically in OCR dispatch (`natural_pdf/ocr/unified_dispatch.py`) and extraction (`natural_pdf/services/extraction_service.py`):

```python skip=true
import natural_pdf
from openai import OpenAI

natural_pdf.set_default_client(OpenAI(), model="gpt-5-mini")

page.apply_ocr(engine="vlm")                          # → remote: default client fills the gap
page.extract(Schema, engine="vlm")                    # → remote: same rule

page.apply_ocr(engine="vlm", model="zai-org/GLM-OCR") # → LOCAL: explicit model wins
page.apply_ocr(engine="glm_ocr")                      # → LOCAL: shorthand resolves a local model
page.extract(Schema, engine="vlm", client=my_client)  # → remote, to *that* client
```

- **The default client applies only when you passed neither `model=` nor `client=`.** It's a fallback for the fully-unspecified call, nothing more.
- **An explicit `model=` never routes to the default client.** This is a deliberate privacy guarantee: naming a local model means your document images run in-process, even with a remote default configured. Internally the dispatch suppresses the default client for the whole call (`suppress_default_client()`), so nothing nested can backfill it either. The same guarantee covers the shorthand engines (`glm_ocr`, `dots`, `chandra`, `paddlevl`) — they resolve to a pinned local model. (Before 0.7 this leaked: a shared code path could backfill the global client and send pages to a remote endpoint despite an explicitly local model. That's fixed and now contractual.)
- **An explicit `client=` always wins** and is used as given; if you passed a client but no model, the default *model name* (not client) can fill in.

So the audit question "did this document leave the machine?" has a static answer: only if you passed a `client=`, or you called a VLM feature with no model and no client while a default client was set. Everything else — every named model, every shorthand engine, every classic OCR/layout/QA/classification engine — is local inference.
