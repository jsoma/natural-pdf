# Installation

Natural PDF pulls text, tables, and labeled values out of PDFs with CSS-like selectors and spatial navigation. This page covers what each install gets you and which operations download models.

## Requirements

- Python 3.10 or newer
- macOS, Linux, or Windows

## The base install

```console
pip install natural-pdf
```

The base install handles digital PDFs — files where the text is selectable, not a scan. It gives you:

- Open PDFs from a file path, URL, or bytes: `PDF("report.pdf")`
- Selectors: `page.find('text:contains("Total")')`, `page.find_all('text:bold')`
- Spatial navigation: `.right()`, `.below()`, `.above()`, `.left()`
- Text and table extraction: `page.extract_text()`, `page.extract_table().to_df()` (pandas is included)
- Page rendering: `page.show()` returns a PIL image you can `.save("page.png")`
- Exclusion zones for stripping headers and footers before extraction
- Checkbox detection (`page.detect_checkboxes()` — downloads a small model, see the callout below)
- LLM-based structured extraction against a remote OpenAI-compatible API (the `openai` client is included; you bring the API key)

Check that it worked:

```python
import natural_pdf

natural_pdf.__version__
```

If the import fails with `ModuleNotFoundError`, you probably installed into a different environment than the one running your script — run `pip show natural-pdf` and compare against `sys.executable`.

## The full install

```console
pip install "natural-pdf[all]"
```

`natural-pdf[all]` is the recommended install for document work. It bundles three extras — `ai`, `export`, and `quality`:

- **`ai`** — `rapidocr` (the default OCR engine), `torch`, `torchvision`, `transformers`, `sentence-transformers`, `timm`, `doclayout_yolo`, and (on Apple Silicon only) `mlx-vlm`. This unlocks `page.apply_ocr()`, local question answering with `page.ask()`, `page.classify()`, semantic search with `pdf.search()`, layout detection with `page.analyze_layout()`, and local VLM OCR/extraction.
- **`export`** — `pikepdf`, `img2pdf`, `openpyxl`, `jupytext`, `nbformat`. This unlocks searchable PDF output (`pdf.save_searchable()`) and Excel export.
- **`quality`** — `pyspellchecker`, `langdetect`. Garble-rate diagnostics for `to_llm()` output (detects OCR'd or corrupted text layers).

It is a large install — `torch` alone is over a gigabyte of wheels. If you only work with digital PDFs and never OCR, classify, or ask questions locally, the base install is enough.

## Feature-by-feature install matrix

| You want to | You call | Extra | Install |
|---|---|---|---|
| OCR a scanned page | `page.apply_ocr()` | `ai` | `pip install "natural-pdf[all]"` |
| Ask questions without an API key | `page.ask("What is the total?")` | `ai` | `pip install "natural-pdf[all]"` |
| Classify pages or regions | `page.classify(["invoice", "receipt"])` | `ai` | `pip install "natural-pdf[all]"` |
| Semantic search over pages | `pdf.search("hazardous materials")` | `ai` | `pip install "natural-pdf[all]"` |
| Detect layout regions | `page.analyze_layout("yolo")` | `ai` | `pip install "natural-pdf[all]"` |
| Run a local VLM (GLM-OCR) | `page.apply_ocr(engine="glm_ocr")` | `ai` | `pip install "natural-pdf[all]"` |
| Save a searchable PDF | `pdf.save_searchable("out.pdf")` | `export` | `pip install "natural-pdf[export]"` |
| Flag garbled text layers | `page.to_llm()` diagnostics | `quality` | `pip install "natural-pdf[quality]"` |
| Use PaddleOCR | `page.apply_ocr(engine="paddle")` | `paddle` (not in `all`) | `pip install "natural-pdf[paddle]"` |
| Use EasyOCR | `page.apply_ocr(engine="easyocr")` | none | `pip install easyocr` |
| Use Surya OCR | `page.apply_ocr(engine="surya")` | none | `pip install surya-ocr` |
| Use Doctr OCR | `page.apply_ocr(engine="doctr")` | none | `pip install python-doctr` |

`paddle` is deliberately not part of `all`: it pins `numpy` below 2.0 and ships its own runtime, so install it only if you need PaddleOCR.

!!! warning "These operations download models on first use"

    Installing `natural-pdf[all]` does **not** download models. The first *call* to each of these features does, into your Hugging Face cache, and the files are reused after that. Sizes below are approximate — check before running these on a metered connection or an offline machine.

    | Operation | Model | Approx. download |
    |---|---|---|
    | `page.ask(...)` / `page.extract(...)` with no LLM client (the `doc_qa` engine) | `impira/layoutlm-document-qa` | ~500 MB |
    | `page.classify(..., using="text")` | `facebook/bart-large-mnli` | ~1.6 GB |
    | `page.classify(..., using="vision")` | `openai/clip-vit-base-patch16` | ~600 MB |
    | VLM OCR / `to_markdown()` with GLM-OCR | `zai-org/GLM-OCR` (Apple Silicon gets the 4-bit `mlx-community/GLM-OCR-4bit` instead, which is smaller) | ~2 GB |
    | `pdf.search(...)` | `all-MiniLM-L6-v2` sentence embeddings | ~90 MB |
    | `page.analyze_layout("yolo")` | DocLayout-YOLO weights (`juliozhao/DocLayout-YOLO-DocStructBench`) | tens of MB |
    | `page.detect_checkboxes()` | YOLO12n ONNX (`wendys-llc/checkbox-detector`) | ~10 MB |

    The default OCR path is the exception: `page.apply_ocr()` uses RapidOCR, whose standard models ship inside the `rapidocr` package — no download on first use.

## Check your environment

Not sure what's installed or why an engine refuses to run?

```console
npdf doctor
```

It prints each dependency group with an OK/MISS status, the installed versions, and the exact `pip install` line for anything missing. Feature calls that hit a missing dependency raise with the same install hint.

## Next

- [Quickstart](quickstart.md) — open a PDF and extract a value and a table in 15 minutes.
- [Coming from pdfplumber](from-pdfplumber.md) — a task-by-task translation guide.
