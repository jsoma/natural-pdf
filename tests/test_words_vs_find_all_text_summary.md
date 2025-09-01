# Difference between page.words and page.find_all('text')

## Summary

While both `page.words` and `page.find_all('text')` return word-level text elements from a PDF page, they have important differences:

### 1. Return Types
- **`page.words`**: Returns a plain Python `list` of `TextElement` objects
- **`page.find_all('text')`**: Returns an `ElementCollection` object (which has additional methods like `.below()`, `.above()`, etc.)

### 2. What They Match
- **`page.words`**: Returns only elements with `type='word'`
- **`page.find_all('text')`**: According to the selector parser, matches elements with `type` in `['text', 'char', 'word']`
  - In practice, for the test PDF, both returned 14 word elements (no 'text' type elements were found)
  - The parser is more flexible and would include 'text' and 'char' type elements if they existed at the word level

### 3. Exclusions Handling
- **`page.words`**: Does NOT apply exclusions - always returns all word elements
- **`page.find_all('text')`**: Applies exclusions by default (can be disabled with `apply_exclusions=False`)

### 4. Usage Recommendations

Use `page.words` when:
- You need quick access to all word elements
- You don't need exclusion filtering
- You don't need the additional methods provided by ElementCollection
- You're doing simple iteration or counting

Use `page.find_all('text')` when:
- You need to respect exclusion zones
- You want to chain spatial methods (`.below()`, `.above()`, etc.)
- You need more complex filtering or selection capabilities
- You want consistent behavior with other find operations

### Example Code
```python
# Direct access to words - no exclusions applied
words = page.words  # Returns: list

# Flexible text selection with exclusions
text_elements = page.find_all('text')  # Returns: ElementCollection

# Skip exclusions if needed
text_no_exclusions = page.find_all('text', apply_exclusions=False)

# Chain operations (only possible with find_all)
header_content = page.find_all('text:contains("Header")').below()
```

### Technical Details
- The 'text' selector in `find_all()` is handled by the selector parser in `natural_pdf/selectors/parser.py`
- It specifically checks for elements where `el.type in ['text', 'char', 'word']`
- The `page.words` property directly accesses `self._element_mgr.words` which is populated during element loading
- Element types in test PDF: 14 words, 59 chars, 0 text elements
