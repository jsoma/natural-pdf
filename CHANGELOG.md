# Changelog

## 0.6.1 - 2026-04-09

Patch release focused on correctness, stability, and internal cleanup.

- Stabilized core state semantics around exclusions, cached pages, and partial-close behavior.
- Tightened selector execution and extraction service contracts for more consistent branch execution, dispatch, and mode resolution.
- Refactored guides generation and grid internals without changing the public guides/table API.
- Fixed release-blocking regressions in closest-string utilities, lazy page loading, and `to_llm()` garble-rate reporting.
- Added regression coverage around selectors, extraction wrappers, OCR/text updates, and lazy-page exclusion behavior.
