#!/bin/bash

# run_all_tutorials.sh - Run all markdown tutorials in the docs/tutorials directory

TUTORIALS_DIR="docs/tutorials"
SCRIPT_DIR="$(dirname "$0")"

# Check if check_run_md.sh exists
if [ ! -f "$SCRIPT_DIR/check_run_md.sh" ]; then
  echo "Error: check_run_md.sh not found in $SCRIPT_DIR"
  exit 1
fi

# Find all markdown files in the tutorials directory
echo "Running all tutorials in $TUTORIALS_DIR..."
for md_file in "$TUTORIALS_DIR"/*.md; do
  if [ -f "$md_file" ]; then
    echo "----------------------------------------"
    echo "Running tutorial: $md_file"
    echo "----------------------------------------"
    "$SCRIPT_DIR/check_run_md.sh" "$md_file"
    # Check if the last command failed
    if [ $? -ne 0 ]; then
      echo "----------------------------------------"
      echo "Failed to run tutorial: $md_file"
      echo "Stopping execution of remaining tutorials"
      exit 1
    fi
    echo "----------------------------------------"
  fi
done 