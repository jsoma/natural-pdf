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
        # Core AI libraries, potentially used by multiple optional features
        "torch>=2.0.0",            # Required for AI models
        "torchvision>=0.15.0",     # Required for AI models
        "transformers>=4.30.0",    # Used for TATR and document QA
        "huggingface_hub>=0.19.0", # For downloading models
    ],
    extras_require={
        # Optional dependencies for specific features/engines (latest versions will be installed)
        "easyocr": ["easyocr"],  # OCR using EasyOCR engine
        "paddle": ["paddlepaddle", "paddleocr"],  # OCR using PaddleOCR engine
        "layout_yolo": ["doclayout_yolo"], # Layout detection using doclayout_yolo
        "surya": ["surya-ocr"], # OCR using Surya
        # QA feature uses 'transformers' from install_requires, so no extra deps needed here unless others are added
        "qa": [], 
        "all": [ # Installs all optional engines/features (latest versions)
            "easyocr",
            "paddlepaddle", 
            "paddleocr",
            "doclayout_yolo",
            "surya-ocr", 
            # Removed "surya-detection",
        ],
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