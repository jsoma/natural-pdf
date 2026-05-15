# Natural PDF Agent Instructions

## Command Policy

- Use `uv run ...` for all Python project commands.
- Do not invoke raw `python`, `pytest`, `black`, `isort`, `mypy`, or `nox`.
- Examples:
  - `uv run python -m pytest tests/test_ocr_cache.py`
  - `uv run pytest tests/ -x`
  - `uv run black --check natural_pdf tests`
  - `uv run nox -s lint`

## Local Context

- Package manager: `uv`
- Virtual environment: `.venv`
- Temp files go in `temp/`.
- Test files go in `tests/`.
- Prefer existing PDFs in `pdfs/` for tests.
