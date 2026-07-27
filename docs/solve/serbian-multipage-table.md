---
fixture: https://github.com/jsoma/natural-pdf/raw/main/pdfs/serbia-zakon-o-naknadama-za-koriscenje-javnih.pdf
tier: nightly
thumbnail: 2
---

# A table that runs for eight pages of a Serbian law

This PDF is a Serbian regulatory document — 120 pages of it — collected for a cross-country study of industry policy. The data you want is one enormous fee table that runs from page 90 to page 97 with no boundary markers between pages, plus a math formula on page 98 that isn't text at all. Nothing about "the table" exists in the file; it's just eight pages of rows.

```python
from natural_pdf import PDF

pdf = PDF("https://github.com/jsoma/natural-pdf/raw/main/pdfs/serbia-zakon-o-naknadama-za-koriscenje-javnih.pdf")
pdf.pages[:8].show(cols=4)
```

## Find the pages by content, not page number

The submitter said "pages 90 to 97," but page numbers shift between document revisions. It's sturdier to say "between the page with this and the page with that." `pdf.find()` searches the whole document and every element knows its `.page`:

```python
first_page = pdf.find(text="Prilog 7.").page
last_page = pdf.find(text='VISINA NAKNADE ZA ZAGAĐENJE VODA').page
pages = pdf.pages[first_page.index:last_page.index+1]
pages.show(cols=4)
```

## One region across page boundaries

We want everything between "Tabela 4" and "Tabela 5". `multipage=True` lets `.below()` keep going past the end of the page — the result is a single region that spans however many pages it needs:

```python
region = (
    pages
    .find(text="Tabela 4")
    .below(
        until="text:contains(Tabela 5)",
        include_endpoint=False,
        multipage=True
    )
)
region.show(cols=4)
```

## Split into sections by category

The table is broken up by category headers labeled "RAZRED". `get_sections()` cuts the region at each one:

```python
sections = region.get_sections('text:contains(RAZRED)', include_boundaries='none')

sections.show(cols=4)
```

Some sections repeat the column headers and some don't — which is exactly what breaks naive extraction. Here's one that spans two pages *and* has headers:

```python
sections[7].show(cols=2)
```

Since it has headers, `.to_df()` just works — the page break in the middle doesn't matter, because the section is one region:

```python
sections[7].extract_table().to_df()
```

This next one has *no* header row:

```python
sections[5].show(cols=2)
```

Tell `to_df()` not to promote the first row, and name the columns yourself:

```python
df = sections[5].extract_table().to_df(header=False)
df.columns = ['Naziv proizvoda', 'Opis proizvoda', 'Jed. mere', 'Naknada u dinarima po jedinici mere']
df
```

Loop over `sections` with that if/else — headers or not — and you have the whole eight-page table.

## The formula on page 98

The math formula isn't text — it's an embedded image. Find its page by the text around it, then grab the image element:

```python
page = pdf.find(text="Obračun naknade za neposredno zagađenje voda").page
page.find("image").show()
```

Natural PDF won't convert the formula to LaTeX for you — that's a job for a math-OCR model. But pulling the image out means you can hand it to one, or to a human.
