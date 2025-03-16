from setuptools import setup, find_packages
from datetime import datetime

# Create version using CalVer: YY.MM.DD.build
now = datetime.now()
build = 2  # Increment this for each release on the same day
version = f"{now.year % 100:02d}.{now.month:02d}.{now.day:02d}.{build}"

setup(
    name="natural-pdf",
    version=version,
    packages=find_packages(),
    package_data={
        "natural_pdf.templates": ["*.html"],
    },
    install_requires=[
        "pdfplumber>=0.7.0",       # Base PDF parsing
        "Pillow>=8.0.0",           # Image processing
        "colour>=0.1.5",           # Color name/hex/RGB conversion for selectors
        "numpy>=1.20.0",           # Required for image processing
        "urllib3>=1.26.0",         # For handling URL downloads
        # The following can be moved to extras to reduce install size
        "doclayout_yolo>=0.0.3",   # YOLO model for document layout detection
        "torch>=2.0.0",            # Required for AI models
        "torchvision>=0.15.0",     # Required for AI models
        "transformers>=4.30.0",    # Used for TATR and document QA
        "huggingface_hub>=0.19.0", # For downloading models
    ],
    extras_require={
        # Optional dependencies for specific features
        "easyocr": ["easyocr>=1.7.0"],  # OCR using EasyOCR engine
        "paddle": ["paddlepaddle>=2.5.0", "paddleocr>=2.7.0"],  # OCR using PaddleOCR engine
        "qa": [],  # Document QA already uses transformers from install_requires 
        "core": [  # Minimal install without AI models
            "pdfplumber>=0.7.0", 
            "Pillow>=8.0.0", 
            "colour>=0.1.5", 
            "numpy>=1.20.0"
        ],
        "ai": [  # Just the AI models
            "doclayout_yolo>=0.0.3",
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "transformers>=4.30.0",
            "huggingface_hub>=0.19.0",
        ],
        "all": [
            "easyocr>=1.7.0",
            "paddlepaddle>=2.5.0", 
            "paddleocr>=2.7.0"
        ],  # Everything
    },
    author="Jonathan Soma",
    author_email="jonathan.soma@gmail.com",
    description="A more intuitive interface for working with PDFs",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jsoma/natural-pdf",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)