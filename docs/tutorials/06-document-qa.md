# Document Question Answering (QA)

Sometimes, instead of searching for specific text patterns, you just want to ask the document a question directly. `natural-pdf` includes an extractive Question Answering feature.

"Extractive" means it finds the literal answer text within the document, rather than generating a new answer or summarizing.

Let's ask our `01-practice.pdf` a few questions.

```python
#%pip install "natural-pdf[ai]"  # DocumentQA relies on torch + transformers
```

```python
from natural_pdf import PDF

# Load the PDF and get the page
pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf")
page = pdf.pages[0]

# Ask about the date
question_1 = "What is the inspection date?"
answer_1 = page.ask(question_1)

# The result dictionary always contains:
#   question    - original question
#   answer      – extracted span (string, may be empty)
#   confidence  – model score 0–1
#   start / end – indices into page.words
#   found       – False if confidence < min_confidence
answer_1
```

```python
page.ask("What company was inspected?")
```

```python
page.ask( "What is statute 5.8.3 about?")
```

The results include the extracted `answer`, a `confidence` score (useful for filtering uncertain answers), the `page_num`, and the `source_elements`.

## Visualising Where the Answer Came From

You can manually access results sources through `answer['source_elements']` but it's much more fun to just use `.show()`.

```python
answer = page.ask("What is the inspection ID?")
answer.show()
```

## Asking an entire PDF

You don't need to select a single page to use `.ask`! It also works for entire PDFs, regions, anything.

```python
pdf.ask("What company was inspected?")
```

Notice that it collects the page number for later investigation.

## Collecting Results into a DataFrame

If you're asking multiple questions, it's often useful to collect the results into a pandas DataFrame. `page.ask` supports passing a **list of questions** directly. This is far faster than looping because the underlying model is invoked only once.

```python
from natural_pdf import PDF
import pandas as pd

pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf")
page = pdf.pages[0]

questions = [
    "What is the inspection date?",
    "What company was inspected?",
    "What is statute 5.8.3 about?",
    "How many violations were there in total?"
]

answers = page.ask(questions, min_confidence=0.2)

df = pd.json_normalize(answers)
df
```

`pd.json_normalize` flattens the list of answer dictionaries straight into a DataFrame, making it easy to inspect the questions, their extracted answers, and associated confidence scores.

## TODO

* Demonstrate passing `model="impira/layoutlm-document-qa"` to switch models.

## QA Model and Limitations

*   The QA system relies on underlying transformer models. Performance and confidence scores vary.
*   It works best for questions where the answer is explicitly stated. It cannot synthesize information or perform calculations (e.g., counting items might fail or return text containing a number rather than the count itself).
*   You can potentially specify different QA models via the `model=` argument in `page.ask()` if others are configured.