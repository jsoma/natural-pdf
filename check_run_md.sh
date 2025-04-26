#!/bin/bash

# check_run_md.sh - Convert markdown tutorials to notebooks and execute them

if [ $# -ne 1 ]; then
  echo "Usage: $0 <path-to-markdown-file>"
  exit 1
fi

MARKDOWN_FILE=$1
NOTEBOOK_FILE="${MARKDOWN_FILE%.md}.ipynb"
KERNEL_NAME="natural-pdf"

echo "Converting $MARKDOWN_FILE to notebook..."
# Jupytext will now automatically add tags based on markdown metadata
jupytext --to ipynb "$MARKDOWN_FILE" || { echo "Conversion failed"; exit 1; }

echo "Patching notebook $NOTEBOOK_FILE with kernel $KERNEL_NAME..."
python3 - <<EOF
import nbformat
nb = nbformat.read("$NOTEBOOK_FILE", as_version=4)
nb.metadata["kernelspec"] = {
    "name": "$KERNEL_NAME",
    "display_name": "Python ($KERNEL_NAME)",
    "language": "python"
}
nbformat.write(nb, "$NOTEBOOK_FILE")
EOF


echo "Executing notebook $NOTEBOOK_FILE..."
jupyter execute "$NOTEBOOK_FILE" --inplace --ExecutePreprocessor.kernel_name=natural-pdf  || { echo "Execution failed"; exit 1; }

echo "Success! Notebook executed and results saved to $NOTEBOOK_FILE"