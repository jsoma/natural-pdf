# Loops, groups and repetitive tasks

Sometimes you need to do things again and again.

## Selecting things

Let's say we have a lot of pages that all look like this:

```python
from natural_pdf import PDF

# Path to sample PDF
pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/Atlanta_Public_Schools_GA_sample.pdf")

# Display the first page
page = pdf.pages[0]
page.show(width=500)
```

We can find all of the book titles by finding **(Removed:** on the page...

```python
page.find_all('text:contains("(Removed:")').show()
```

...but it's repeated on each following page, too!

```python
pdf.pages[1].find_all('text:contains("(Removed:")').show()
```

No problem, you can use `pdf.find_all` the same way to do with a single page - you just can't highlight them with `.show()` the same way.

```python
pdf.find_all('text:contains("(Removed:")')
```

You can see there are **37** across the entire PDF.

## Extracting data from elements

If you just want the text, `.extract_text()` will smush it all together, but you can also get it as a list.

```python
titles = pdf.find_all('text:contains("(Removed:")')

titles.extract_each_text()
```

You can also loop through them like a normal list...

```python
for title in titles[:10]:
    print(title.extract_text(), title.page.number)
```

...but you can also use `.apply` for a little functional-programming flavor.

```python
titles.apply(lambda title: {
    'title': title.extract_text(),
    'page': title.page.number
})
```

I think `.map` also works on that front?

## Filtering

You can also filter if you only want some of them. For example, maybe we weren't sure how to pick between the different **Removed:** text blocks.

```python
elements = page.find_all('text:contains("Removed:")')
elements.show()
```

We can filter for the ones that don't say "Copies Removed"

```python
titles = elements.filter(
    lambda element: 'Copies Removed' not in element.extract_text()
)
titles.show()
```
