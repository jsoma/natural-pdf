#!/bin/bash
set -e

# Print start message
echo "=== Starting test install for natural_pdf with all extras ==="

echo "Python version: $(python --version)"
echo "pip version: $(pip --version)"
echo "uv version: $(uv --version)"

# Create and activate a fresh venv using uv
echo
echo "--- Creating and activating fresh virtual environment (.testenv) ---"
uv venv .testenv
source .testenv/bin/activate

# Install the package with all extras
echo
echo "--- Installing natural_pdf with all extras ---"
uv pip install -v ".[all]"

# List installed packages
echo
echo "--- Installed packages ---"
pip list

# Test import, print version and import path
echo
echo "--- Profiling import times with python -X importtime ---"
python -X importtime -c "import natural_pdf" > importtime_output.txt 2>&1
echo "Import time profile saved to importtime_output.txt"

echo
echo "--- Testing import and version ---"
python -c "import natural_pdf; print('Import succeeded! Version:', natural_pdf.__version__); print('Imported from:', natural_pdf.__file__)"

# Deactivate and clean up
echo
echo "--- Deactivating and removing test environment ---"
deactivate
rm -rf .testenv

echo

# Success message
echo "✅ Install and import test succeeded!" 