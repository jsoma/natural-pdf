# OCR Fine-tuning

While the built-in OCR engines (EasyOCR, PaddleOCR, Surya) offer good general performance, you might encounter situations where their accuracy isn't sufficient for your specific needs. This is often the case with:

*   **Unique Fonts:** Documents using unusual or stylized fonts.
*   **Specific Languages:** Languages or scripts not perfectly covered by the default models.
*   **Low Quality Scans:** Noisy or degraded document images.
*   **Specialized Layouts:** Text within complex tables, forms, or unusual arrangements.

Fine-tuning allows you to adapt a pre-trained OCR recognition model to your specific data, significantly improving its accuracy on documents similar to those used for training.

## Why Fine-tune?

-   **Higher Accuracy:** Achieve better text extraction results on your specific document types.
-   **Adaptability:** Train the model to recognize domain-specific terms, symbols, or layouts.
-   **Reduced Errors:** Minimize downstream errors in data extraction and processing pipelines.

## Strategy: Detect + LLM Correct + Export

Training an OCR model requires accurate ground truth: images of text snippets paired with their correct transcriptions. Manually creating this data is tedious. A powerful alternative leverages the strengths of different models:

1.  **Detect Text Regions:** Use a robust local OCR engine (like Surya or PaddleOCR) primarily for its *detection* capabilities (`detect_only=True`). This identifies the *locations* of text on the page, even if the initial *recognition* isn't perfect. You can combine this with layout analysis or region selections (`.region()`, `.below()`, `.add_exclusion()`) to focus on the specific areas you care about.
2.  **Correct with LLM:** For each detected text region, send the image snippet to a powerful Large Language Model (LLM) with multimodal capabilities (like GPT-4o, Claude 3.5 Sonnet/Haiku) using the `direct_ocr_llm` utility. The LLM performs high-accuracy OCR on the snippet, providing a "ground truth" transcription.
3.  **Export for Fine-tuning:** Use the `PaddleOCRRecognitionExporter` to package the original image snippets (from step 1) along with their corresponding LLM-generated text labels (from step 2) into the specific format required by PaddleOCR for fine-tuning its *recognition* model.

This approach combines the efficient spatial detection of local models with the superior text recognition of large generative models to create a high-quality fine-tuning dataset with minimal manual effort.

## Example: Fine-tuning for Greek Spreadsheet Text

Let's walk through an example of preparing data to fine-tune PaddleOCR for text from a scanned Greek spreadsheet, adapting the process described above.

```python
# --- 1. Setup and Load PDF ---
from natural_pdf import PDF
from natural_pdf.ocr.utils import direct_ocr_llm
from natural_pdf.exporters import PaddleOCRRecognitionExporter
import openai # Or your preferred LLM client library
import os

# Ensure your LLM API key is set (using environment variables is recommended)
# os.environ["OPENAI_API_KEY"] = "sk-..." 
# os.environ["ANTHROPIC_API_KEY"] = "sk-..." 

# pdf_path = "path/to/your/document.pdf" 
pdf_path = "pdfs/hidden/the-bad-one.pdf" # Replace with your PDF path
pdf = PDF(pdf_path)

# --- 2. (Optional) Exclude Irrelevant Areas ---
# If the document has consistent headers, footers, or margins you want to ignore
# Use exclusions *before* detection
pdf.add_exclusion(lambda page: page.region(right=45)) # Exclude left margin/line numbers
pdf.add_exclusion(lambda page: page.region(left=500)) # Exclude right margin

# --- 3. Detect Text Regions ---
# Use a good detection engine. Surya is often robust for line detection.
# We only want the bounding boxes, not the initial (potentially inaccurate) OCR text.
print("Detecting text regions...")
# Process only a subset of pages for demonstration if needed
for page in pdf.pages[:10]:
    # Use a moderate resolution for detection; higher res used for LLM correction later
    page.apply_ocr(engine='surya', resolution=120, detect_only=True) 
print(f"Detection complete for {num_pages_to_process} pages.")

# (Optional) Visualize detected boxes on a sample page
# pdf.pages[9].find_all('text[source=ocr]').show() 

# --- 4. Correct with LLM ---
# Configure your LLM client (example using OpenAI client, adaptable for others)
# For Anthropic: client = openai.OpenAI(base_url="https://api.anthropic.com/v1/", api_key=os.environ.get("ANTHROPIC_API_KEY"))
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY")) 

# Craft a clear prompt for the LLM
# Be as specific as possible! If it's in a specific language, what kinds
# of characters, etc.
prompt = """OCR this image patch. Return only the exact text content visible in the image. 
Preserve original spelling, capitalization, punctuation, and symbols. 
Do not add any explanatory text, translations, comments, or quotation marks around the result.
The text is likely from a Greek document, potentially a spreadsheet, containing Modern Greek words or numbers."""

# Define the correction function using direct_ocr_llm
def correct_text_region(region):
    # Use a high resolution for the LLM call for best accuracy
    return direct_ocr_llm(
        region, 
        client, 
        prompt=prompt, 
        resolution=300, 
        # model="claude-3-5-sonnet-20240620" # Example Anthropic model
        model="gpt-4o-mini" # Example OpenAI model
    )

# Apply the correction function to the detected text regions
print("Applying LLM correction to detected regions...")
for page in pdf.pages[:num_pages_to_process]:
    # This finds elements added by apply_ocr and passes their regions to 'correct_text_region'
    # The returned text from the LLM replaces the original OCR text for these elements
    # The source attribute is updated (e.g., to 'ocr-llm-corrected')
    page.correct_ocr(correct_text_region) 
print("LLM correction complete.")

# --- 5. Export for PaddleOCR Fine-tuning ---
print("Configuring exporter...")
exporter = PaddleOCRRecognitionExporter(
    # Select all of the non-blank OCR text
    # Hopefully it's all been LLM-corrected! 
    selector="text[source^=ocr][text!='']", 
    resolution=300,     # Resolution for the exported image crops
    padding=2,          # Add slight padding around text boxes
    split_ratio=0.9,    # 90% for training, 10% for validation
    random_seed=42,     # For reproducible train/val split
    include_guide=True  # Include the Colab fine-tuning notebook
)

# Define the output directory
output_directory = "./my_paddleocr_finetune_data"
print(f"Exporting data to {output_directory}...")

# Run the export process
exporter.export(pdf, output_directory)

print("Export complete.")
print(f"Dataset ready for fine-tuning in: {output_directory}")
print(f"Next step: Upload '{os.path.join(output_directory, 'fine_tune_paddleocr.ipynb')}' and the rest of the contents to Google Colab.")

# --- Cleanup ---
pdf.close() 
```

## Running the Fine-tuning

The `PaddleOCRRecognitionExporter` automatically includes a Jupyter Notebook (`fine_tune_paddleocr.ipynb`) in the output directory. This notebook is pre-configured to guide you through the fine-tuning process on Google Colab (which offers free GPU access):

1.  **Upload:** Upload the entire output directory (e.g., `my_paddleocr_finetune_data`) to your Google Drive or directly to your Colab instance.
2.  **Open Notebook:** Open the `fine_tune_paddleocr.ipynb` notebook in Google Colab.
3.  **Set Runtime:** Ensure the Colab runtime is set to use a GPU (Runtime -> Change runtime type -> GPU).
4.  **Run Cells:** Execute the cells in the notebook sequentially. It will:
    *   Install necessary libraries (PaddlePaddle, PaddleOCR).
    *   Point the training configuration to your uploaded dataset (`images/`, `train.txt`, `val.txt`, `dict.txt`).
    *   Download a pre-trained PaddleOCR model (usually a multilingual one).
    *   Start the fine-tuning process using your data.
    *   Save the fine-tuned model checkpoints.
    *   Export the best model into an "inference format" suitable for use with `natural-pdf`.
5.  **Download Model:** Download the resulting `inference_model` directory from Colab.

## Using the Fine-tuned Model

Once you have the `inference_model` directory, you can instruct `natural-pdf` to use it for OCR:

```python
from natural_pdf import PDF
from natural_pdf.ocr import PaddleOCROptions

# Path to the directory you downloaded from Colab
finetuned_model_dir = "/path/to/your/downloaded/inference_model" 

# Specify the path in PaddleOCROptions
paddle_opts = PaddleOCROptions(
    rec_model_dir=finetuned_model_dir,
    rec_char_dict_path=os.path.join(finetuned_model_dir, 'your_dict.txt') # Or wherever your dict is
    use_gpu=True # If using GPU locally
)

pdf = PDF("another-similar-document.pdf")
page = pdf.pages[0]

# Apply OCR using your fine-tuned model
ocr_elements = page.apply_ocr(engine='paddle', options=paddle_opts)

# Extract text using the improved results
text = page.extract_text() 
print(text)

pdf.close()
```

By following this process, you can significantly enhance OCR performance on your specific documents using the power of fine-tuning. 