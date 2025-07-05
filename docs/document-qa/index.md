# Document Question Answering

Natural PDF includes document QA functionality that allows you to ask natural language questions about your PDFs and get relevant answers. This feature uses LayoutLM models to understand both the text content and the visual layout of your documents.

## Setup

Let's start by loading a sample PDF to experiment with question answering.

```python
from natural_pdf import PDF

# Path to sample PDF
pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/0500000US42001.pdf")

# Display the first page
page = pdf.pages[0]
page.show()
```

## Basic Usage

Here's how to ask questions to a PDF page:

```python
# Ask a question about the entire document
page.ask("How many votes did Harris and Waltz get?")
```

```python
page.ask("Who got the most votes for Attorney General?")
```

```python
page.ask("Who was the Republican candidate for Attorney General?")
```

## Asking questions to part of a page questions

You can also ask questions to a specific *region of* a page*:

```python
# Get a specific page
region = page.find('text:contains("Attorney General")').below()
region.show()
```

```python
region.ask("How many write-in votes were cast?")
```

## Asking multiple questions

```python
import pandas as pd

questions = [
    "How many votes did Harris and Walz get?",
    "How many votes did Trump get?",
    "How many votes did Natural PDF get?",
    "What was the date of this form?"
]

# You can actually do this but with multiple questions
# in the model itself buuuut Natural PDF can'd do it yet
results = [page.ask(q) for q in questions]

df = pd.json_normalize(results)
df
```

## Visualizing where answers come from

Sometimes you'll want to see exactly where the model found an answer in your document. Maybe you're checking if it grabbed the right table cell, or you want to verify it didn't confuse similar-looking sections.

```python
result = page.ask("Who got the most votes for Attorney General?")

# See the answer
print(result.answer)  # "John Smith"

# Show exactly where it found that answer
result.show()
```

The `result.show()` method highlights the specific text elements the model used to answer your question - super helpful for debugging or when you need to double-check the results.

You can also access result data like a normal dictionary or use dot notation if you prefer:

```python
# Both of these work the same way
print(result["confidence"])  # 0.97
print(result.confidence)     # 0.97
```

If the model couldn't find a confident answer, `result.found` will be `False` and calling `result.show()` will let you know there's nothing to visualize.

## Next Steps

Now that you've learned about document QA, explore:

- [Element Selection](../element-selection/index.ipynb): Find specific elements to focus your questions.
- [Layout Analysis](../layout-analysis/index.ipynb): Automatically detect document structure.
- [Working with Regions](../regions/index.ipynb): Define custom areas for targeted questioning.
- [Text Extraction](../text-extraction/index.ipynb): Extract and preprocess text before QA.
