"""
Shared prompt components for benchmark configurations.

Import these constants and combine them in your prompts.
"""

# Instruction to copy values exactly as they appear
EXACT_COPY = """IMPORTANT: Copy all values EXACTLY as they appear in the document.
Do not correct any spelling, numbers, or formatting."""

# Instruction for proper CSV formatting
CSV_QUOTING = """Use proper CSV quoting: wrap fields in double quotes if they contain commas, newlines, or quotes."""

# Common endings
CSV_ONLY = """Output ONLY the CSV, no explanation."""
JSON_ONLY = """Output ONLY the JSON, no explanation."""
