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
#   answer      – extracted span (string, may be empty)
#   confidence  – model score 0–1
#   start / end – indices into page.words
#   found       – False if confidence < min_confidence
answer_1
# ➜ {'answer': 'July 31, 2023', 'confidence': 0.82, 'start': 33, 'end': 36, 'found': True}
```

```python
# Ask about the company name
question_2 = "What company was inspected?"
answer_2 = page.ask(question_2)

# Display the answer dictionary
answer_2
```

```python
# Ask about specific content from the table
question_3 = "What is statute 5.8.3 about?"
answer_3 = page.ask(question_3)

# Display the answer
answer_3
```

The results include the extracted `answer`, a `confidence` score (useful for filtering uncertain answers), the `page_num`, and the `source_elements`.

## Visualising Where the Answer Came From

```python
from natural_pdf.elements.collections import ElementCollection

page.clear_highlights()

if answer_1["found"]:
    words = ElementCollection(page.words[answer_1["start"] : answer_1["end"] + 1])
    words.show(color="yellow", label=question_1)

page.to_image()
```

## Collecting Results into a DataFrame

If you're asking multiple questions, it's often useful to collect the results into a pandas DataFrame for easier analysis.

```python
from natural_pdf import PDF
import pandas as pd

pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf")
page = pdf.pages[0]

# List of questions to ask
questions = [
    "What is the inspection date?",
    "What company was inspected?",
    "What is statute 5.8.3 about?",
    "How many violations were there in total?" # This might be less reliable
]

# Collect answers for each question
results = []
for q in questions:
    ans = page.ask(q, min_confidence=0.2)
    ans["question"] = q
    results.append(ans)

cols = ["question", "answer", "confidence", "found"]
qa_df = pd.DataFrame(results)[cols]
qa_df
```

This shows how you can iterate through questions, collect the answer dictionaries, and then create a structured DataFrame, making it easy to review questions, answers, and their confidence levels together.

## TODO

* Demonstrate passing `model="impira/layoutlm-document-qa"` to switch models.
* Show multi-page QA: iterate over `pdf.pages` and add `page` column to the results.
* Add batch helper (`pdf.ask_many(questions)`) once implemented.

## Wish List

* Support for highlighting answer automatically via a `show_answer()` helper.
* Option to return bounding box coordinates directly (`bbox`) in the answer dict.
* Add `ElementCollection.to_dataframe()` for one-call DataFrame creation.

<div class="admonition note">
<p class="admonition-title">QA Model and Limitations</p>

    *   The QA system relies on underlying transformer models. Performance and confidence scores vary.
    *   It works best for questions where the answer is explicitly stated. It cannot synthesize information or perform calculations (e.g., counting items might fail or return text containing a number rather than the count itself).
    *   You can potentially specify different QA models via the `model=` argument in `page.ask()` if others are configured.
</div> 