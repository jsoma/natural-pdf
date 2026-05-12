# Workflow PDF Reuse: Pennsylvania Config Cache

Track: workflow-level PDF reuse

Hypothesis: Some benchmark-style workflows reopen the same PDF once per page.
Caching the `PDF` object inside the benchmark config estimates workflow-level
overhead and helps distinguish library cost from benchmark orchestration cost.

Primary case: `real:0500000US42001`.

Risk: low as workflow guidance, high as a generic production patch. This patch is
specific to one benchmark config and closes cached PDFs on context exit.
