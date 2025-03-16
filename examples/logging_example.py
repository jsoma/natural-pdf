"""
Example demonstrating the logging system in Natural PDF.
"""
import os
import sys
import logging
from pathlib import Path

# Add the project directory to the path to import the library
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from natural_pdf import configure_logging, PDF

def main():
    # Basic setup with INFO level
    print("=== Configuring logging at INFO level ===")
    configure_logging(level=logging.INFO)
    
    # Create a PDF
    pdf_path = "./pdfs/01-practice.pdf"
    if not os.path.exists(pdf_path):
        # Try another file if the first one doesn't exist
        pdf_path = list(Path("./pdfs").glob("*.pdf"))[0]
    
    print(f"\nLoading PDF with standard logging: {pdf_path}")
    pdf = PDF(pdf_path)
    
    # OCR with default settings (should log OCR engine initialization)
    print("\nExtracting text with OCR")
    text = pdf.pages[0].extract_text(ocr=True)
    print(f"Extracted {len(text)} characters")
    
    # Switch to DEBUG level
    print("\n=== Configuring logging at DEBUG level ===")
    configure_logging(level=logging.DEBUG)
    
    # Try layout detection (generates more detailed logs)
    print("\nRunning layout detection with DEBUG logging")
    regions = pdf.pages[0].analyze_layout(
        model="paddle",
        model_params={"detect_text": True, "verbose": True}
    )
    print(f"Found {len(regions)} regions")
    
    # Try logging to a file
    print("\n=== Logging to a file ===")
    log_file = os.path.join("output", "natural_pdf.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Create a file handler with custom formatter
    file_handler = logging.FileHandler(log_file, mode='w')  # 'w' mode to overwrite any existing file
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Configure logging with the file handler
    configure_logging(level=logging.DEBUG, handler=file_handler)
    
    # Force a few log events
    logger = logging.getLogger("natural_pdf")
    logger.debug("This is a debug message written to the log file")
    logger.info("This is an info message written to the log file")
    logger.warning("This is a warning message written to the log file")
    
    # Run another OCR operation to log to the file
    print(f"Running OCR with logging to {log_file}")
    text = pdf.pages[0].extract_text(ocr=True)
    
    print(f"\nDone! Check {log_file} for detailed logs.")

if __name__ == "__main__":
    main()