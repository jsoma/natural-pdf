# Document Question Answering (QA)

Sometimes, instead of searching for specific text patterns, you just want to ask the document a question directly. `natural-pdf` includes an extractive Question Answering feature.

"Extractive" means it finds the literal answer text within the document, rather than generating a new answer or summarizing.

Let's ask our `01-practice.pdf` a few questions.

```python
#%pip install "natural-pdf[all]"
```

```python
from natural_pdf import PDF

# Load the PDF and get the page
pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf")
page = pdf.pages[0]

# Ask about the date
question_1 = "What is the inspection date?"
answer_1 = page.ask(question_1)

# The result is a dictionary with the answer, confidence, etc.
answer_1
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
    answer_dict = page.ask(q)
    # Add the original question to the dictionary
    answer_dict['question'] = q
    results.append(answer_dict)

# Convert the list of dictionaries to a DataFrame
# We select only the most relevant columns here
df_results = pd.DataFrame(results)[['question', 'answer', 'confidence']]

# Display the DataFrame
df_results
```

This shows how you can iterate through questions, collect the answer dictionaries, and then create a structured DataFrame, making it easy to review questions, answers, and their confidence levels together.

<div class="admonition note">
<p class="admonition-title">QA Model and Limitations</p>

    *   The QA system relies on underlying transformer models. Performance and confidence scores vary.
    *   It works best for questions where the answer is explicitly stated. It cannot synthesize information or perform calculations (e.g., counting items might fail or return text containing a number rather than the count itself).
    *   You can potentially specify different QA models via the `model=` argument in `page.ask()` if others are configured.
</div> 