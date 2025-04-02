#!/bin/bash

# check_run_md.sh - Convert markdown tutorials to notebooks and execute them

if [ $# -ne 1 ]; then
  echo "Usage: $0 <path-to-markdown-file>"
  exit 1
fi

MARKDOWN_FILE=$1
NOTEBOOK_FILE="${MARKDOWN_FILE%.md}.ipynb"

echo "Converting $MARKDOWN_FILE to notebook..."
# Jupytext will now automatically add tags based on markdown metadata
jupytext --to ipynb "$MARKDOWN_FILE" || { echo "Conversion failed"; exit 1; }

echo "Executing notebook $NOTEBOOK_FILE..."
jupyter execute "$NOTEBOOK_FILE" --inplace || { echo "Execution failed"; exit 1; }

echo "Success! Notebook executed and results saved to $NOTEBOOK_FILE"