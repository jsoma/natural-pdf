# Semantic Search Across Multiple Documents

When working with a collection of PDFs, you might need to find information relevant to a specific query across all documents, not just within a single one. This tutorial demonstrates how to perform semantic search over a `PDFCollection`.

```python
#%pip install "natural-pdf[all]"
#%pip install "natural-pdf[search]"  # Ensure search dependencies are installed
```

```python
import logging
import natural_pdf

# Optional: Configure logging to see progress
natural_pdf.configure_logging(level=logging.INFO)

# Define the paths to your PDF files
pdf_paths = [
    "https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf",
    "https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/Atlanta_Public_Schools_GA_sample.pdf"
    # Add more PDF paths as needed
]

# Create a PDFCollection
collection = natural_pdf.PDFCollection(pdf_paths)
print(f"Created collection with {len(collection.pdfs)} PDFs.")
```

## Initializing the Search Index

Before performing a search, you need to initialize the search capabilities for the collection. This involves processing the documents and building an index.

```python
# Initialize search. 'index=True' builds the index immediately.
# This might take some time depending on the number and size of PDFs.
collection.init_search(index=True) 
print("Search index initialized.")
```

## Performing a Semantic Search

Once the index is ready, you can use the `find_relevant()` method to search for content semantically related to your query.

```python
# Perform a search query
query = "american president"
results = collection.find_relevant(query)

print(f"Found {len(results)} results for '{query}':")
```

## Understanding Search Results

The `find_relevant()` method returns a list of dictionaries, each representing a relevant text chunk found in one of the PDFs. Each result includes:

*   `pdf_path`: The path to the PDF document where the result was found.
*   `page_number`: The page number within the PDF.
*   `score`: A relevance score (higher means more relevant).
*   `content_snippet`: A snippet of the text chunk that matched the query.

```python
# Process and display the results
if results:
    for i, result in enumerate(results):
        print(f"  {i+1}. PDF: {result['pdf_path']}")
        print(f"     Page: {result['page_number']} (Score: {result['score']:.4f})")
        # Display a snippet of the content
        snippet = result.get('content_snippet', '')
        print(f"     Snippet: {snippet}...") 
else:
    print("  No relevant results found.")

# You can access the full content if needed via the result object, 
# though 'content_snippet' is usually sufficient for display.
```

Semantic search allows you to efficiently query large sets of documents to find the most relevant information without needing exact keyword matches, leveraging the meaning and context of your query. 