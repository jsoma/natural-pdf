# Progress Overview

## What We Accomplished
- Reworked QA asking APIs (`ask`, `ask_batch`) to support a new generative pathway with explicit `mode`, `llm_client`, and `prompt` parameters while keeping extractive behaviour intact.
- Added dedicated generative tests with a stubbed LLM client to validate the new signature and expected failures when a client is missing.
- Refactored visual search into `match_template`, introduced a deprecation wrapper for the old `find_similar`, and expanded coverage to confirm behaviour.
- Hardened table tooling: added `skiprows`, `dtype`, and `copy` to `TableResult.to_df`, tightened kwargs validation, and introduced focused tests.
- Standardised selector/text APIs across pages, regions, and collections to accept the same tolerance/reading-order options, updating docstrings for lint compliance.
- Ran Ruff over the touched modules, eliminating stale imports, unused vars, duplicated properties, and unsafe `except` blocks so our active surface is lint-clean.
- Added targeted documentation inside tests to ensure QA dependencies are skipped consistently using the new `_require_qa()` helper.

## Why These Changes Were Necessary
- QA modes lacked a clear path for LLM-backed responses; adding generative support closes that gap and aligns signatures with the mixin contract.
- Visual search naming was confusing and the old behaviour is under review; the rename/deprecation makes future replacements safer.
- Table conversions relied on blanket kwargs, making type checking noisy and surfacing runtime surprises; explicit parameters improve predictability.
- Selector helpers were inconsistent across entry points, causing both lint warnings and harder-to-reason behaviour; aligning signatures reduces future maintenance pain.
- The codebase had accumulated lint failures from earlier refactors; cleaning them now lets us trust future Ruff runs as a regression gate.
- Tests required repeated dependency guards, cluttering files and triggering lint warnings; the helper centralises that logic.

## Next Steps
- **Linting:** Run Ruff across the entire repository (not just the modified slices), address remaining warnings, and consider adding Ruff to the default CI lint session.
- **Testing speed:** Audit the test suite to identify slow cases; explore parametrisation cuts, fixture re-use, or selective skipping to bring runtime down while keeping coverage high.
