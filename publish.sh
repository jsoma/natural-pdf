#!/bin/bash
set -e

# Clean previous builds
rm -rf build/ dist/ *.egg-info

# Build the package
python -m pip install --upgrade pip
python -m pip install --upgrade build twine
python -m build

# Show the files that will be published
echo "Files to be published:"
ls -l dist/

# Verify the package
python -m twine check dist/*

# Ask for confirmation before publishing
read -p "Do you want to publish to PyPI? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    # Publish to PyPI
    python -m twine upload dist/*
    
    # Extract the version from a wheel filename in dist/
    WHEEL_FILE=$(ls dist/*.whl | head -1)
    VERSION=$(basename $WHEEL_FILE | cut -d'-' -f2)
    
    # Create a git tag
    git tag -a "v$VERSION" -m "Release $VERSION"
    
    echo "Package v$VERSION published and tagged!"
    echo "Don't forget to push the tag with: git push origin v$VERSION"
else
    echo "Publishing canceled."
fi