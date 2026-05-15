# LLM Agent Harness Plan

This folder tracks the next phase of `.to_llm()` and LLM-assisted Natural PDF extraction work.

The current benchmark answers a different question: can a model extract data directly from a PDF? The next useful question is: can a model learn to use `natural-pdf` to write deterministic, reproducible extraction code?

## Goal

Build a small harness that gives an LLM controlled ways to inspect a PDF, write `natural-pdf` code, run it, receive feedback, and iterate. Use the harness to compare prompt and tool designs before changing `.to_llm()` too aggressively.

## Immediate Workstreams

1. Improve `.to_llm()` as an exploration primitive, not a context dump.
2. Build an agent episode runner that executes candidate extraction code and grades outputs.
3. Create a compact synthetic and real-PDF task corpus.
4. Compare several guidance modes:
   - no `.to_llm()`, docs/search only
   - current `.to_llm()`
   - bounded `.to_llm()` with no suggestions
   - bounded `.to_llm()` with suggestions
   - narrow inspection tools
   - broad shell-style exploration
   - retrieved expert examples

## Documents

- [`to-llm-next.md`](to-llm-next.md): changes to make `.to_llm()` safer and more useful in the short term.
- [`agent-harness.md`](agent-harness.md): episode flow, tool surface, execution model, and artifacts.
- [`experiment-matrix.md`](experiment-matrix.md): guidance variants to compare.
- [`corpus-and-scoring.md`](corpus-and-scoring.md): task corpus design and grading metrics.
- [`roadmap.md`](roadmap.md): staged implementation plan.

## Working Principle

Do not assume that more structured tool use is better. Test it against simpler baselines such as docs plus `rg`, `find`, Python snippets, and direct calls to existing `natural-pdf` APIs. The harness should make it cheap to discover when a specialized tool helps and when it pushes the model into a brittle default mode.

## Related Local Material

- Existing `.to_llm()` design: [`docs/specs/to_llm_spec.md`](../../docs/specs/to_llm_spec.md)
- Existing `.to_llm()` implementation plan: [`docs/specs/to_llm_implementation_plan.md`](../../docs/specs/to_llm_implementation_plan.md)
- Current direct-extraction benchmark configs: [`benchmark/configs/`](../../benchmark/configs)
- Current performance harness: [`experiments/performance/`](../performance)
- Bad PDF submissions mirror: [`bad-pdfs/submissions/`](../../bad-pdfs/submissions)
