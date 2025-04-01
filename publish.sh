#!/bin/bash
set -e

# --- Publishing Script for natural-pdf ---
# Expected Workflow:
# 1. Ensure your working directory is clean.
# 2. Create a Git tag for the release: git tag vX.Y.Z
# 3. Run this script: ./publish.sh
# 4. If publish is successful, push the tag: git push origin vX.Y.Z
# ----------------------------------------

# --- Check for Git Tag --- 
# Check if the current commit HEAD is tagged. setuptools-scm uses this tag.
# Redirect stderr to /dev/null to avoid printing git errors if no tag is found.
if ! git describe --exact-match --tags HEAD 2> /dev/null; then
    echo "Error: HEAD commit is not tagged. Publishing requires a Git tag (e.g.,
 vX.Y.Z)."
    echo "Please tag the commit you want to release and try again."
    exit 1
fi
TAGGED_VERSION=$(git describe --exact-match --tags HEAD)
echo "Found Git tag: $TAGGED_VERSION. Proceeding with publish..."
# ------------------------

# Clean previous builds
rm -rf build/ dist/ *.egg-info

# Ensure build tools are installed/updated
python -m pip install --upgrade pip build twine

# Build the package
# setuptools-scm will automatically determine the version from the latest Git tag
echo "Building package..."
python -m build

# Show the files that will be published
echo "\nFiles generated in dist/:"
ls -l dist/

# Verify the package distribution files
echo "\nVerifying distribution files..."
python -m twine check dist/*

# Ask for confirmation before publishing
read -p "\nDo you want to publish to PyPI? (y/n) " -n 1 -r
echo # Move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
    # Publish to PyPI
    echo "\nPublishing to PyPI..."
    python -m twine upload dist/*
    
    # Version extraction and tagging removed - tagging should be done *before* running this script.
    echo "\nPackage published!"
    echo "Remember to push the tag used for this release (e.g., git push origin vX.Y.Z)"
else
    echo "\nPublishing canceled."
fi