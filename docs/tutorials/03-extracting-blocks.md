# Extracting Text Blocks

Often, you need a specific section, like a paragraph between two headings. You can find a starting element and select everything below it until an ending element.

Let's extract the "Summary" section from `01-practice.pdf`. It starts after "Summary:" and ends before the thick horizontal line.

```python
#%pip install "natural-pdf[all]"
```


```python
from natural_pdf import PDF

# Load the PDF and get the page
pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf")
page = pdf.pages[0]

# Find the starting element ("Summary:")
start_marker = page.find('text:contains("Summary:")')

# Select elements below the start_marker, stopping *before*
# the thick horizontal line (a line with height > 1).
summary_elements = start_marker.below(
    include_element=True, # Include the "Summary:" text itself
    until="line[height > 1]"
)

# Visualize the elements found in this block
summary_elements.highlight(color="lightgreen", label="Summary Block")

# Extract and display the text from the collection of summary elements
summary_elements.extract_text()

```

```python
# Display the page image to see the visualization
page.to_image()
```

This selects the elements using `.below(until=...)` and extracts their text. The second code block displays the page image with the visualized section.

<div class="admonition note">
<p class="admonition-title">Selector Specificity</p>

    We used `line[height > 1]` to find the thick horizontal line. You might need to adjust selectors based on the specific PDF structure. Inspecting element properties can help you find reliable start and end markers.
</div>
