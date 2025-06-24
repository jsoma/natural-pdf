# Bad-PDF Evaluation Toolkit

This folder contains small CLI utilities that drive the automated evaluation
harness for "bad" PDFs.

## 1. Analyse PDFs → `eval_results/<ID>`

```bash
# Analyse *every* submission listed in the default CSV
python -m tools.bad_pdf_eval.eval_suite \
       --csv bad-pdfs/Bad\ PDF\ Submission\ form_Submissions_2025-06-22.csv \
       --output-dir eval_results

# Common flags
--max-row 20      # debug – analyse first 20 rows of the CSV
--limit 10        # stop after processing 10 locally-found PDFs
```

What you get per submission:

```
 eval_results/<ID>/
   ├─ summary.json            # all metrics, page samples, thumbnails
   ├─ page_0001.jpg / .txt    # per-page images + describe/inspect snippets
   └─ …                       # any auxiliary artefacts
```

## 2. LLM enrichment (adds reasoning + code)

```bash
export OPENAI_API_KEY=sk-…     # required just once

# Enrich every summary that lacks thought_process/code_suggestion
python -m tools.bad_pdf_eval.llm_enrich --model o3

# Re-enrich a single PDF (overwrite existing fields)
python -m tools.bad_pdf_eval.llm_enrich \
       --submission ja6EqV1 \
       --force
```

Enrichment is idempotent – you can run it in parallel with the analyser; it will
skip any summaries that are still being written.

## 3. Export a CSV of the enrichment

```bash
python -m tools.bad_pdf_eval.export_enrichment_csv \
       --out eval_results/enrichment_summary.csv
```

## 4. Customising which pages are analysed

`eval_suite` picks three default pages (1-3) **plus** any page numbers it can
parse from the submission text (e.g. "page 30").  To override this behaviour,
open `tools/bad_pdf_eval/eval_suite.py` and tweak `DEFAULT_PAGES` or
`build_pages_list()`.

---
### Recap cheat-sheet files used by the LLM
* `LLM_NaturalPDF_CheatSheet.md` – API reference snippets
* `LLM_NaturalPDF_Workflows.md` – end-to-end examples

Both live in this folder so they travel with the code. 