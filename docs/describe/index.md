# Describe Functionality

The `describe()` and `inspect()` methods provide an easy way to understand the contents of your PDF elements without having to visualize them as images.

## Basic Usage

Get a summary of an entire page:

```python
from natural_pdf import PDF

pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf")
page = pdf.pages[0]

page.describe()
```

## Element collection summaries

You can describe element collections on a page with `.describe()`.

```python
# Describe all elements on the page
page.find_all('text').describe()
```

```python
# Describe all elements on the page
page.find_all('rect').describe()
```

## Inspecting lists of elements

For more detail, you can view specific details of element collections with `inspect()`.

```python
page.find_all('text').inspect()
```

```python
page.find_all('line').inspect()
```
