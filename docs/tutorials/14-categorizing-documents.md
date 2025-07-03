# Categorizing documents

When working with a collection of PDFs, you might need to automatically categorize pages of PDFs or entire collections of PDFs.

```python
#%pip install "natural-pdf[ai]"
```

```python
from natural_pdf import PDF

pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/cia-doc.pdf")
pdf.pages.show(columns=6)
```

## Vision classification

These pages are easily differentiable based on how they *look*, so we can most likely use a vision model to tell them apart.

```python
pdf.classify_pages(['diagram', 'text', 'invoice', 'blank'], using='vision')

for page in pdf.pages:
    print(f"Page {page.number} is {page.category} - {page.category_confidence:0.3}")
```

How did it do?

```python
(
    pdf.pages
    .filter(lambda page: page.category == 'diagram')
    .show(show_category=True)
)
```

Looks great! Note that **I had to play around with the categories a bit** before I got something that worked. Using "blank" doesn't ever show up, "invoice" did a lot better than "form," etc etc. It's pretty quick and easy to sanity check so you shouldn't have to suffer too much.

I can also save just those pages into a new PDF document.

```python tags=["skip-execution"]
(
    pdf.pages
    .filter(lambda page: page.category == 'diagram')
    .save_pdf("output.pdf", original=True)
)
```

## Text classification (default)

By default the search is done using **text**. It takes the text on the page and feeds it to the classifier along with the categories. Note that you might need to OCR your content first!

```python
pdf.classify_pages(['diagram', 'text', 'invoice', 'blank'], using='text')

for page in pdf.pages:
    print(f"Page {page.number} is {page.category} - {page.category_confidence:0.3}")
```

How does it compare to our vision option?

```python
pdf.pages.filter(lambda page: page.category == 'diagram').show(show_category=True)
```

Yes, you can notice that it's *wrong*, but more importantly **look at the confidence scores**. Low scores are your best clue that something might not be perfect (beyond manually checking things, of course).

If you're processing documents that are text-heavy you'll have much better luck with a text model as compared to a vision one.

## PDF classification

If you want to classify entire PDFs, the process is similar. The only gotcha is you can't use `using="vision"` with multi-page PDFs (yet?).

```python
import natural_pdf

pdf_paths = [
    "https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf",
    "https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/Atlanta_Public_Schools_GA_sample.pdf"
]

# Import your PDFs
pdfs = natural_pdf.PDFCollection(pdf_paths)

# Run your classification
pdfs.classify_all(['school', 'business'], using='text')
```
