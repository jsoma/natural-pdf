# Document Question Answering

Natural PDF includes document QA functionality that allows you to ask natural language questions about your PDFs and get relevant answers. This feature uses LayoutLM models to understand both the text content and the visual layout of your documents.

## Basic Usage

Here's how to ask questions to a PDF document:

```python
from natural_pdf import PDF

# Open a PDF
pdf = PDF('document.pdf')

# Ask a question about the entire document
result = pdf.ask("What is the total revenue reported?")

# Get the answer
if result.get("found", False):
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Found on page: {result.get('page_num', 0)}")
else:
    print("No answer found.")
```

## Asking Questions to Specific Pages

You can also ask questions to a specific page:

```python
# Get a page
page = pdf.pages[0]

# Ask a question just about this page
result = page.ask("Who is the CEO?")

# Get the answer
if result.get("found", False):
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']:.2f}")
    
    # Access the source elements that contain the answer
    if "source_elements" in result:
        # Highlight the answer
        for element in result["source_elements"]:
            element.highlight(color=(1, 0.5, 0, 0.3))
        
        # Save the highlighted image
        page.save_image("answer_highlighted.png")
```

## Asking Questions to Regions

You can even ask questions to specific regions of a page:

```python
# Find a region
title = page.find('text:contains("Financial Report")')
financial_section = title.below(height=300)

# Ask a question just about this region
result = financial_section.ask("What was the net profit?")

# Get the answer
if result.get("found", False):
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']:.2f}")
    
    # Highlight the region and the answer
    financial_section.highlight(color=(0, 0, 1, 0.2), label="Financial Section")
    
    if "source_elements" in result:
        for element in result["source_elements"]:
            element.highlight(color=(1, 0, 0, 0.3), label="Answer")
    
    # Save the highlighted image
    page.save_image("region_answer.png")
```

## Controlling Model and Parameters

You can control which model is used and set various parameters:

```python
# Specify a different model
result = pdf.ask(
    "What was the company's revenue in 2022?",
    model="microsoft/layoutlmv3-large"
)

# Set a higher confidence threshold for more reliable answers
result = pdf.ask(
    "What year was the company founded?", 
    min_confidence=0.8  # Only accept answers with 80%+ confidence
)

# Control image resolution for better accuracy
result = page.ask(
    "What is the CEO's name?",
    resolution=300  # Higher resolution for better text recognition
)

# Set a maximum length for the answer
result = page.ask(
    "Summarize the business outlook",
    max_answer_length=50  # Keep the answer concise
)

# Specify the language for non-English documents
result = pdf.ask(
    "Quels sont les revenus totaux?",
    language="fr"  # Use French for the question and answer
)
```

## Handling OCR Documents

Document QA works with both native text PDF documents and scanned documents requiring OCR:

```python
# For a scanned document, enable OCR
pdf = PDF('scanned_document.pdf', ocr=True)

# Ask a question - OCR will be automatically applied if needed
result = pdf.ask("What is the date of the report?")

# You can explicitly control OCR behavior
result = page.ask(
    "Who signed the document?",
    use_ocr=True,  # Force OCR even if there's native text
    ocr_languages=["en"]  # Specify OCR language
)
```

For best results with scanned documents, you might need to fine-tune OCR settings. See our [OCR Challenges and Solutions](../explanations/ocr-challenges.md) guide for detailed advice on improving OCR quality before applying Document QA.

## Debugging and Troubleshooting

If you're having trouble with document QA, you can enable debugging to see more details:

```python
# Enable debug mode to save intermediate files
result = page.ask(
    "What is the company's mission statement?",
    debug=True  # Will save images and word boxes to the output directory
)
```

This will save:
- The input image
- The extracted word boxes
- A visualization of the word boxes
- The raw result from the model

You can also specify a custom debug directory:

```python
result = page.ask(
    "What is the company's mission statement?",
    debug=True,
    debug_dir="qa_debug"  # Save debug files to this directory
)
```

## Handling Complex Documents

For complex documents, you might want to break them down into focused questions:

```python
# First find the relevant section
financial_section = pdf.find('text:contains("Financial Results")').below(height=500)

# Then ask specific questions about that section
profit_result = financial_section.ask("What was the net profit?")
revenue_result = financial_section.ask("What was the total revenue?")
growth_result = financial_section.ask("What was the year-over-year growth?")

# Combine the answers
if profit_result.get("found") and revenue_result.get("found"):
    print(f"Net profit: {profit_result['answer']}")
    print(f"Total revenue: {revenue_result['answer']}")
    if growth_result.get("found"):
        print(f"Growth: {growth_result['answer']}")
        
    # Highlight all answers with different colors
    if "source_elements" in profit_result:
        for elem in profit_result["source_elements"]:
            elem.highlight(color=(1, 0, 0, 0.3), label="Profit")
            
    if "source_elements" in revenue_result:
        for elem in revenue_result["source_elements"]:
            elem.highlight(color=(0, 1, 0, 0.3), label="Revenue")
            
    if "source_elements" in growth_result:
        for elem in growth_result["source_elements"]:
            elem.highlight(color=(0, 0, 1, 0.3), label="Growth")
            
    # Save the visualization
    financial_section.page.save_image("financial_answers.png", labels=True)
```

## Preprocessing Documents for Better Results

For best results, you might want to prepare your documents:

```python
# Remove headers and footers
pdf.add_exclusion(
    lambda page: page.find('text:contains("Confidential")').above(),
    label="header"
)
pdf.add_exclusion(
    lambda page: page.find('text:contains("Page")').below(),
    label="footer"
)

# Find the main content area and focus questions there
for page in pdf.pages:
    # Apply layout analysis to find the main content
    page.analyze_layout()
    
    # Get text regions
    text_regions = page.find_all('region[type=plain-text]')
    
    # Ask questions about the main content
    for region in text_regions:
        result = region.ask("What is the main topic discussed?")
        if result.get("found", False) and result.get("confidence", 0) > 0.7:
            print(f"Page {page.page_number}, Topic: {result['answer']}")
```

## A Complete Document QA Example

Here's a complete example that walks through the document QA process:

```python
from natural_pdf import PDF
import os

# Create output directory if it doesn't exist
os.makedirs("qa_results", exist_ok=True)

# Open a PDF
pdf = PDF('annual_report.pdf')

# First, ask a general question to find the most relevant page
result = pdf.ask("Where is the financial summary?")
if result.get("found", False):
    print(f"Financial summary is on page {result.get('page_num', 0)}")
    page = pdf.pages[result.get('page_num', 0) - 1]  # Convert to 0-indexed
else:
    # Default to first page if not found
    page = pdf.pages[0]
    print("Financial summary location not found, using first page")

# Apply layout analysis to find regions
page.analyze_layout()

# Find the table regions
tables = page.find_all('region[type=table]')
print(f"Found {len(tables)} tables on page {page.page_number}")

# Ask questions about each table
for i, table in enumerate(tables):
    # Extract table data
    table_data = table.extract_table()
    
    # Highlight the table
    table.highlight(color=(0, 0, 1, 0.2), label=f"Table {i+1}")
    
    # Ask questions about the table
    revenue_result = table.ask("What was the total revenue?")
    profit_result = table.ask("What was the net profit?")
    
    print(f"\nTable {i+1}:")
    if revenue_result.get("found", False):
        print(f"  Revenue: {revenue_result['answer']} (confidence: {revenue_result['confidence']:.2f})")
        # Highlight the answer
        if "source_elements" in revenue_result:
            for elem in revenue_result["source_elements"]:
                elem.highlight(color=(0, 1, 0, 0.3), label="Revenue")
    
    if profit_result.get("found", False):
        print(f"  Net Profit: {profit_result['answer']} (confidence: {profit_result['confidence']:.2f})")
        # Highlight the answer
        if "source_elements" in profit_result:
            for elem in profit_result["source_elements"]:
                elem.highlight(color=(1, 0, 0, 0.3), label="Profit")
                
# Save the highlighted page
page.save_image("qa_results/financial_qa.png", labels=True)

# Now ask about textual content
text_regions = page.find_all('region[type=plain-text]')
for i, region in enumerate(text_regions):
    # Ask about this text region
    summary_result = region.ask("What does this section discuss?")
    
    if summary_result.get("found", False) and summary_result.get("confidence", 0) > 0.6:
        print(f"\nText region {i+1}:")
        print(f"  Topic: {summary_result['answer']}")
        
        # Highlight the region
        region.highlight(color=(0.5, 0.5, 0, 0.2), label=f"Text {i+1}: {summary_result['answer'][:20]}")

# Save the final highlighted page with all answers
page.save_image("qa_results/page_qa_analysis.png", labels=True)

# Ask broader questions about the document
broad_questions = [
    "What was the company's overall performance?",
    "What are the main business risks mentioned?",
    "What is the outlook for next year?"
]

print("\nDocument-wide insights:")
for question in broad_questions:
    result = pdf.ask(question)
    if result.get("found", False):
        print(f"\nQ: {question}")
        print(f"A: {result['answer']}")
        print(f"   (confidence: {result['confidence']:.2f}, page: {result.get('page_num', 0)})")
```

## How It Works

Document QA in Natural PDF:

1. Extracts text elements and their positions from the PDF
2. Renders a high-resolution image of the page or region
3. Passes the image, text positions, and your question to a LayoutLM model
4. Returns the answer with confidence score and source elements

The use of layout-aware models (like LayoutLM) allows the system to understand both the textual content and the spatial layout of the document, making it more effective than text-only QA systems for documents with complex layouts.

### Word Box Processing

For technical users, here's how the system processes word boxes:

```python
# The document QA system converts text elements to word boxes like this:
word_boxes = [
    {
        "text": "Revenue",
        "box": [100, 200, 150, 220],  # [x0, y0, x1, y1] in pixel coordinates
        "page_num": 1,
        "confidence": 1.0  # For native text (non-OCR)
    },
    # More word boxes...
]

# These are then processed by the LayoutLM model along with the page image
```

## Technical Considerations

When using document QA, consider these technical details:

- **Memory Usage**: Processing large documents or high-resolution images requires significant memory
- **Speed**: The first question may be slower as models are loaded; subsequent questions are faster
- **Model Selection**: Larger models (like layoutlmv3-large) are more accurate but slower
- **OCR Quality**: The quality of OCR text extraction directly impacts QA accuracy
- **Image Resolution**: Higher resolution improves accuracy but consumes more memory
- **Answer Confidence**: Always check the confidence score; lower scores may indicate uncertain answers

## Next Steps

Now that you know how to use document QA, you might want to explore:

- [OCR](../ocr/index.md) for working with scanned documents first
- [OCR Challenges and Solutions](../explanations/ocr-challenges.md) for improving OCR quality
- [Layout Analysis](../layout-analysis/index.md) for identifying regions to ask questions about
- [Visual Debugging](../visual-debugging/index.md) to visualize results
- [Why Document QA is Better than LLMs](../explanations/ocr-challenges.md#why-document-qa-can-be-better-than-just-using-an-llm) for advantages over general LLMs