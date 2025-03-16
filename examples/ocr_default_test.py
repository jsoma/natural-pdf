"""
Test to ensure OCR is disabled by default.
"""
import os
import sys

# Add the parent directory to the path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from natural_pdf import PDF

def test_ocr_default():
    """Test that OCR is disabled by default but can be enabled explicitly."""
    # Use the scanned PDF for testing OCR
    pdf_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'pdfs', 'needs-ocr.pdf'))
    
    if not os.path.exists(pdf_path):
        # Fall back to a different PDF
        pdf_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'pdfs', 'HARRY ROQUE_redacted.pdf'))
        
    if not os.path.exists(pdf_path):
        print("No suitable PDF file found for OCR testing. Please provide a scanned PDF file.")
        return
    
    print(f"Testing with PDF: {pdf_path}")
    
    # Test 1: OCR should be OFF by default
    print("\nTEST 1: Default Behavior (OCR should be OFF)")
    print("-" * 60)
    
    with PDF(pdf_path) as pdf:
        # Print initial OCR config
        print(f"Initial OCR config: {pdf._ocr_config}")
        print(f"OCR enabled? {pdf._ocr_config.get('enabled', False)}")
        
        # Extract text without OCR
        page = pdf.pages[0]
        text = page.extract_text()
        
        print(f"Extracted {len(text)} characters without explicit OCR")
        print(f"First 100 chars: {text[:100]}...")
    
    # Test 2: Explicit OCR enable via constructor
    print("\nTEST 2: Explicit OCR Enable via Constructor")
    print("-" * 60)
    
    with PDF(pdf_path, ocr=True) as pdf:
        # Print OCR config
        print(f"OCR config: {pdf._ocr_config}")
        print(f"OCR enabled? {pdf._ocr_config.get('enabled', False)}")
        
        # Extract text with OCR
        page = pdf.pages[0]
        text = page.extract_text()
        
        print(f"Extracted {len(text)} characters with OCR enabled in constructor")
        print(f"First 100 chars: {text[:100]}...")
    
    # Test 3: Explicit OCR enable via extract_text parameter
    print("\nTEST 3: Explicit OCR Enable via extract_text parameter")
    print("-" * 60)
    
    with PDF(pdf_path) as pdf:
        # Print initial OCR config
        print(f"Initial OCR config: {pdf._ocr_config}")
        print(f"OCR enabled? {pdf._ocr_config.get('enabled', False)}")
        
        # Extract text with OCR parameter
        page = pdf.pages[0]
        text = page.extract_text(ocr=True)
        
        print(f"Extracted {len(text)} characters with OCR enabled in extract_text")
        print(f"First 100 chars: {text[:100]}...")
    
    # Test 4: OCR via with_ocr builder
    print("\nTEST 4: OCR via with_ocr builder")
    print("-" * 60)
    
    with PDF(pdf_path) as pdf:
        # Configure OCR with builder
        pdf.with_ocr(enabled=True, languages=["en"])
        
        # Print updated OCR config
        print(f"Updated OCR config: {pdf._ocr_config}")
        print(f"OCR enabled? {pdf._ocr_config.get('enabled', False)}")
        
        # Extract text with OCR configured via builder
        page = pdf.pages[0]
        text = page.extract_text()
        
        print(f"Extracted {len(text)} characters with OCR enabled via builder")
        print(f"First 100 chars: {text[:100]}...")

if __name__ == "__main__":
    test_ocr_default()