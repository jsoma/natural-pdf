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
df.insert(0, 'question', questions)
df
```

## Next Steps

Now that you've learned about document QA, explore:

- [Element Selection](../element-selection/index.ipynb): Find specific elements to focus your questions.
- [Layout Analysis](../layout-analysis/index.ipynb): Automatically detect document structure.
- [Working with Regions](../regions/index.ipynb): Define custom areas for targeted questioning.
- [Text Extraction](../text-extraction/index.ipynb): Extract and preprocess text before QA.