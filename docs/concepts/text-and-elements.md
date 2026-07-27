---
fixture: pdfs/01-practice.pdf
---

# Text and Elements

A PDF does not contain words. It contains individual glyphs — characters placed at coordinates — and natural-pdf has to *decide* what counts as a word. Most text-finding surprises trace back to that decision, so it's worth knowing exactly how it works.

## How chars become words

When a page loads, the word engine takes the raw characters (via pdfplumber) and groups them:

1. Characters are bucketed into lines using a vertical tolerance (`y_tolerance`).
2. Within a line, consecutive characters join into one word while the horizontal gap between them stays under `x_tolerance`. A bigger gap starts a new word.
3. A word also splits — regardless of the gap — when the font changes. The default split attributes are `fontname`, `size`, `bold`, and `italic` (plus strikethrough/underline/highlight state), so **a bold label and its regular-weight value are always separate elements**, even when they're touching.
4. Blank characters are kept (`keep_blank_chars=True`), so a run of text with ordinary spaces can survive as a *single* word element containing spaces — that's why a whole phrase like "Jungle Health and Safety Inspection Service" is often one element. For PDFs that omit space characters entirely, a gap of at least 0.12× the font size gets a space injected (`space_gap_ratio`).

The tolerances scale with the text by default (`auto_text_tolerance=True`): `x_tolerance` = 0.35 × the page's median character size, `y_tolerance` = 0.6 × median size, and an `x_tolerance_ratio` of 0.35 additionally scales the gap threshold to each character's own font size, so 9 pt body text and 15 pt headlines each get sensible thresholds. If a page has no measurable sizes, both fall back to 3.0 pt.

You can see the grouping — and the font-based splits — directly:

```python
from natural_pdf import PDF

pdf = PDF("pdfs/01-practice.pdf")
page = pdf.pages[0]

for word in page.find_all('text')[:5]:
    print(f"{word.text!r:45} bold={word.bold} font={word.fontname}")
```

`Site:` and `Durham’s Meatpacking` sit on the same visual line, but the bold label split into its own element.

## Why `:contains()` misses text that `extract_text()` shows

This is the most common problem people report with the library ([#4](https://github.com/jsoma/natural-pdf/issues/4), [#5](https://github.com/jsoma/natural-pdf/issues/5)), and LLM agents hit it constantly, so here it is bluntly:

**`:contains()` tests each element's text separately. `extract_text()` assembles text across elements. A phrase that spans an element boundary exists in the second and not the first.**

```python
"Site: Durham" in page.extract_text()
```

```python
print(page.find('text:contains("Site: Durham")'))   # spans two elements
```

`extract_text()` happily prints `Site: Durham’s Meatpacking` because it stitches the label element and the value element together with a space. But no single element contains that string — `Site:` is one element (bold) and `Durham’s Meatpacking` is another (regular) — and `:contains()` is a per-element substring check (`natural_pdf/selectors/_clauses.py`). There is no cross-element phrase matching in the selector engine.

What to do instead:

- **Anchor on the shortest distinctive piece** that lives inside one element: `text:contains("Site:")`, then navigate — `label.right()`, `label.below()` — to get the rest. This is the intended pattern.
- **Search the assembled text** when you just need to know whether a phrase exists: `"Site: Durham" in page.extract_text()` or a regex over it.
- **Inspect the actual elements** when a selector mysteriously fails: `page.find_all('text:contains("Durham")').inspect()` shows you where the element boundaries really are.

The same boundary logic explains the reverse surprise: `text:contains("Health and Safety")` *does* match here, because that whole line was kept as one element (spaces and all). Whether a phrase matches depends on how the words got grouped — which is exactly why the grouping rules above matter.

## What `text_tolerance=` actually affects

`PDF("file.pdf", text_tolerance={...})` changes how characters group into word *elements at load time*. It is not a search option and has no effect on selectors after elements exist. Recognized keys: `x_tolerance`, `x_tolerance_ratio`, `y_tolerance`, `y_tolerance_ratio`, `keep_blank_chars`, `space_gap_ratio`; unknown keys are ignored. Pass `auto_text_tolerance=False` to turn off the font-size scaling and use pdfplumber-style fixed defaults.

Raise `x_tolerance` and gappy letter-spaced text merges into fewer, longer elements; lower it and words split apart. If `:contains()` keeps missing phrases in a specific document, re-loading with a larger `x_tolerance` is sometimes the honest fix — but check `keep_blank_chars` grouping with `.inspect()` before tuning blind.

## `:ocr()` — matching garbled text

OCR output (and some born-digital PDFs with broken encodings) contains confusable characters: `l`/`1`/`I`, `O`/`0`, curly vs. straight quotes. `:contains()` does exact substring matching, so `“Durham’s”` with a curly apostrophe won't match a search for `"Durham's"`:

```python
print(page.find("""text:contains("Durham's Meatpacking")"""))   # curly vs straight apostrophe
```

```python
match = page.find("""text:ocr("Durham's Meatpacking")""")
match.text
```

`:ocr()` scores candidates with an OCR-confusion-aware similarity (`natural_pdf/selectors/ocr_match.py`): confusable characters count as near-matches, and the query is matched as a substring window, so `:ocr()` finds everything `:contains()` finds plus the garbled variants. The default acceptance threshold is 0.75; tune it inline with `text:ocr("Total@0.6")`. Results come back sorted by score, best first. It's still per-element — `:ocr()` does not fix the element-boundary problem above.

## Never hand-join element text

Element `.text` values carry no spacing information about their neighbors. Joining them yourself reconstructs the boundary problem in reverse:

```python
row = page.find_all('text')[2:5]
print("".join(el.text for el in row))     # wrong: words fuse together
print(row.extract_text())                 # right: assembled with spacing
```

`ElementCollection.extract_text()` runs the same assembly used for page/region extraction — reading order, word separators, exclusions — so the output matches what `page.extract_text()` would show for those elements. The rule: **element `.text` is for inspecting one element; any time two or more elements become a string, go through `.extract_text()`.**
