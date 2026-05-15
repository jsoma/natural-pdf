# Agent Harness Design

The harness should evaluate whether a model can produce reliable `natural-pdf` extraction code, not whether it can directly read a PDF.

## Episode Flow

1. Load task manifest.
2. Give the agent the PDF path, goal, expected output schema, and permitted resources.
3. Agent explores using one of the configured guidance modes.
4. Agent writes extraction code.
5. Harness runs the code in a subprocess.
6. Harness captures stdout, stderr, returned JSON/CSV, generated images, and runtime.
7. Harness grades the result.
8. If configured, the agent receives compact feedback and can revise.
9. Harness records all turns, commands, code, outputs, scores, and costs.

## Task Contract

Each task should define:

```yaml
id: practice_health_inspection
pdf: pdfs/01-practice.pdf
pages: [1]
goal: Extract the form fields and violations table.
output_format: json
expected: experiments/llm-agent-harness/tasks/practice_health_inspection/expected.json
tags:
  - label_value
  - ruled_table
  - checkboxes
allowed_guidance:
  - docs
  - to_llm
  - probes
  - screenshots
max_iterations: 4
```

## Agent Output Contract

The agent should submit one Python file with a function:

```python
def extract(pdf_path: str):
    ...
```

The function must return JSON-serializable data. This keeps grading independent of stdout formatting.

The harness can still capture stdout for debugging, but stdout should not be the primary result channel.

## Execution Model

Run code in a subprocess with:

- timeout
- clean temporary working directory
- repository import path
- fixed environment variables
- no network by default
- optional image artifact directory

Use the same runner for model-generated code and expert solutions so runtime and failures are comparable.

## Feedback Model

Feedback should be compact and actionable:

```text
Run failed:
  AttributeError: 'NoneType' object has no attribute 'right'
Likely cause:
  selector found no element: text:contains("Site:")
Available nearby text samples:
  "Site:", "Durham's Meatpacking", "Chicago, Ill."
```

For incorrect output:

```text
Score: 22/33 fields
Wrong:
  form_fields.site expected "Durham's Meatpacking", got "Durham's Meatpacking Chicago, Ill."
  violations[0].repeat expected "yes", got "no"
```

Avoid giving the full expert solution during normal evaluation.

## Tool Surface Variants

The harness should make tool access configurable so we can test whether specialized tools help.

### Minimal Shell

The agent can use ordinary code/search:

- `rg`
- `python`
- existing docs
- direct `natural-pdf` calls

This is the baseline for the "find + grep beats tools" concern.

### Broad NPDF Probe Tool

Single flexible probe:

```python
run_npdf_probe(pdf_path, code)
```

The model writes small snippets. This is close to normal coding and may avoid overly constraining the model.

### Narrow Inspection Tools

Purpose-built tools:

- `page_overview(page)`
- `selector_sample(selector)`
- `region_text(bbox)`
- `table_preview(strategy)`
- `ocr_status(page)`

These should be tested, not assumed better.

### Suggestion Tool

A tool that returns next possible probes:

```text
candidate probe: page.extract_table().to_df().head()
candidate probe: page.find_all("text:bold").extract_each_text()
candidate probe: page.find("text:contains('Site')").right(until="text").extract_text()
```

The grading should track whether the agent uses suggestions and whether suggestions improve outcome.

## Recorded Artifacts

For each episode, save:

- task metadata
- model and guidance config
- prompts/tool outputs
- submitted code per iteration
- subprocess result per iteration
- normalized extracted data
- score breakdown
- runtime and token estimates
- final failure reason if any

Recommended output path:

```text
experiments/llm-agent-harness/results/<timestamp>/<task>/<model>/<guidance>/
```

## Safety Checks

Generated code should not:

- modify repository files
- call network
- use shell commands outside the temporary directory
- read unrelated local files except docs/PDFs explicitly allowed by the task
- use API keys

For early local experiments, start with a best-effort subprocess guard and review artifacts manually before tightening isolation.
