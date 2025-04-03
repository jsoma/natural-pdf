# execute_notebooks.py

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import yaml
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, Future, as_completed
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import nbformat
from jupytext import jupytext
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

# --- Configuration ---
DOCS_DIR = Path("docs")
CACHE_FILE = Path(".notebook_cache.json")
# Add relative paths or glob patterns from DOCS_DIR, e.g., 'api/', '**/_*.md'
EXCLUDE_PATTERNS = [
    "installation/index.md",
    "ocr/index.md",
    "explanations"
    "api/index.md",
]
MAX_WORKERS = os.cpu_count()

# --- Logging Setup ---
# Logger name remains the same
logger = logging.getLogger("notebook_executor")
# Default level set high; will be adjusted by command-line args
logger.setLevel(logging.WARNING)
logger.propagate = False
# Ensure handlers are not duplicated if script is re-run/imported
if not logger.handlers:
    console_handler = logging.StreamHandler(sys.stdout) # Use stdout for logs
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

# Type alias for cache entries
CacheEntry = Dict[str, Any]

# --- Cache Handling ---

def load_cache() -> Dict[str, CacheEntry]:
    """Loads the cache file if it exists."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r") as f:
                cache_data = json.load(f)
                return {k: v for k, v in cache_data.items() if isinstance(v, dict)}
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Cache file {CACHE_FILE} is corrupted or unreadable ({e}). Starting fresh.")
            return {}
    return {}

def save_cache(cache_data: Dict[str, CacheEntry]):
    """Saves the cache data to the file."""
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache_data, f, indent=2, sort_keys=True)
    except IOError as e:
        logger.error(f"Failed to save cache file {CACHE_FILE}: {e}")

# --- File Processing ---

def calculate_hash(file_path: Path) -> str:
    """Calculates the SHA256 hash of a file's content."""
    hasher = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except IOError as e:
        # Log error here as it might not be caught elsewhere if hash fails early
        logger.error(f"Could not read file {file_path} for hashing: {e}")
        return ""

def is_excluded(file_path: Path, base_dir: Path, exclude_patterns: List[str]) -> bool:
    """Checks if a file path matches any exclude patterns."""
    relative_path = file_path.relative_to(base_dir)
    # Only exclude index.md if it's in the root directory
    if str(file_path) == "docs/index.md":
        return True
    if 'explanations' in str(file_path):
        return True
    for pattern in exclude_patterns:
        if pattern in str(file_path):
            return True
    return False

def find_markdown_files(base_dir: Path, exclude_patterns: List[str]) -> List[Path]:
    """Finds all .md files in the base directory, excluding specified patterns."""
    if not base_dir.is_dir():
        logger.error(f"Docs directory not found: {base_dir}")
        return []

    md_files = []
    for md_file in base_dir.rglob("*.md"):
        if not is_excluded(md_file, base_dir, exclude_patterns):
            md_files.append(md_file)
        else:
            # Use logger configured in main process
            logger.debug(f"Excluding {md_file} due to exclude patterns.")

    return md_files


# --- Worker Function ---
def process_notebook(md_file_path_str: str, log_level: int) -> Dict[str, Any]:
    """
    Processes a single markdown file: converts, executes, saves ipynb.
    Designed to be run in a separate process. Log level is passed explicitly.

    Returns:
        A dictionary containing: 'file_id', 'status', 'error_message', 'new_hash', 'elapsed'.
        Status: 'success', 'failed'.
    """
    # Setup logger *within* the worker process
    worker_logger = logging.getLogger(f"notebook_worker_{os.getpid()}")
    worker_logger.setLevel(log_level)
    if not worker_logger.handlers:
        # Log to stderr from workers to potentially separate from main stdout
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(f"%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        worker_logger.addHandler(handler)
        # Add file handler if specified globally (needs path passed or global access)
        # For simplicity, file logging handled centrally in main process based on results

    md_file_path = Path(md_file_path_str)
    file_id = str(md_file_path.relative_to(DOCS_DIR)) # Relative path as ID
    ipynb_file_path = md_file_path.with_suffix(".ipynb")
    # REMOVED: worker_logger.info(f"Processing: {file_id}") - Progress bar handles this

    start_time = time.monotonic()
    current_hash = calculate_hash(md_file_path) # Hash calculated here now
    error_message = None
    notebook = None
    status = 'failed' # Default

    if not current_hash:
        error_message = "Could not read file for hashing."
        worker_logger.error(f"{file_id}: {error_message}")
        elapsed = time.monotonic() - start_time
        return {
            "file_id": file_id, "status": "failed", "error_message": error_message,
            "new_hash": None, "elapsed": elapsed
        }

    try:
        md_content = md_file_path.read_text(encoding='utf-8')
        notebook = jupytext.reads(md_content, fmt='md')

        cwd = md_file_path.parent
        client = NotebookClient(notebook, timeout=600, kernel_name='python3', resources={'metadata': {'path': str(cwd)}})
        client.execute() # Modifies 'notebook' object

        status = 'success'
        worker_logger.debug(f"{file_id}: Execution successful.") # Success is DEBUG

    except CellExecutionError as e:
        status = 'failed'
        # Format error nicely from nbclient if possible
        error_message = f"Cell execution failed (see notebook output for details):\n{e}"
        worker_logger.error(f"{file_id}: {error_message}", exc_info=False) # Log error at ERROR

    except Exception as e:
        status = 'failed'
        error_message = f"Unexpected error during processing: {e}"
        worker_logger.error(f"{file_id}: {error_message}", exc_info=True) # Log full traceback for unexpected
        notebook = None # Don't save if conversion/unexpected error

    # Write executed notebook (if available), even on CellExecutionError
    if notebook is not None:
        try:
            with open(ipynb_file_path, "w", encoding='utf-8') as f:
                nbformat.write(notebook, f)
            worker_logger.debug(f"{file_id}: Saved notebook to {ipynb_file_path.relative_to(DOCS_DIR)}") # Save is DEBUG
        except Exception as e:
            status = 'failed' # Overwrite status if save fails
            error_message = f"Failed to write notebook file {ipynb_file_path}: {e}"
            worker_logger.error(f"{file_id}: {error_message}", exc_info=True)

    elapsed = time.monotonic() - start_time
    # REMOVED: worker_logger.info(f"{file_id}: Finished successfully in {elapsed:.2f}s.") - Progress bar handles this

    return {
        "file_id": file_id, "status": status, "error_message": error_message,
        "new_hash": current_hash, "elapsed": elapsed
    }


# --- Main Orchestration ---
def setup_logging(level: int, log_file: Optional[str]):
    """Configures the main logger level and optionally adds a file handler."""
    logger.setLevel(level)
    # Configure console handler level (already added)
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(level)

    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, mode='a') # Append mode
            file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - [%(name)s] - %(message)s")
            file_handler.setFormatter(file_formatter)
            # File handler logs everything from DEBUG level upwards
            file_handler.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)
            logger.info(f"Logging detailed output to {log_file}")
        except Exception as e:
            logger.error(f"Failed to set up log file handler for {log_file}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Execute documentation notebooks.")
    parser.add_argument(
        "-f", "--force", action="store_true",
        help="Force re-execution of all notebooks, ignoring cache."
    )
    parser.add_argument(
        "--workers", type=int, default=MAX_WORKERS,
        help=f"Number of parallel workers (default: {MAX_WORKERS})"
    )
    parser.add_argument(
        "--log-file", type=str, default=None,
        help="Path to a file for detailed logging (DEBUG level)."
    )
    # Verbosity control
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        "-q", "--quiet", action="store_true",
        help="Show only warnings, errors, progress bar, and final summary."
    )
    verbosity_group.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show detailed debug information (includes worker logs)."
    )

    args = parser.parse_args()

    # Determine log level
    log_level = logging.INFO
    if args.quiet:
        log_level = logging.WARNING
    elif args.verbose:
        log_level = logging.DEBUG

    # Setup logging based on args
    setup_logging(log_level, args.log_file)

    logger.info("Starting notebook execution process...")
    start_time = time.monotonic()

    cache = load_cache()
    processed_files = 0
    skipped_files = 0
    failed_files_count = 0
    files_to_process_paths: List[str] = []
    all_md_files = find_markdown_files(DOCS_DIR, EXCLUDE_PATTERNS)

    logger.info(f"Found {len(all_md_files)} markdown files in {DOCS_DIR}.")

    # --- Determine files to process ---
    skipped_cached_files: List[str] = [] # Initialize list for skipped files
    for md_file in all_md_files:
        file_id = str(md_file.relative_to(DOCS_DIR))
        cache_entry = cache.get(file_id)
        # Perform hashing only if needed for comparison or processing
        if not args.force and cache_entry and cache_entry.get("status") == "success":
            current_hash = calculate_hash(md_file)
            if not current_hash:
                logger.warning(f"{file_id}: Could not hash, will re-run.")
                files_to_process_paths.append(str(md_file))
            elif cache_entry.get("hash") == current_hash:
                logger.debug(f"Skipping cached and unchanged: {file_id}")
                skipped_files += 1
                skipped_cached_files.append(file_id) # Add skipped file_id
            else:
                logger.info(f"Detected change in: {file_id}")
                files_to_process_paths.append(str(md_file))
        else:
            if args.force:
                 logger.info(f"Forcing re-run for: {file_id}")
            elif not cache_entry:
                 logger.info(f"Detected new file: {file_id}")
            elif cache_entry.get("status") != "success":
                 logger.info(f"Retrying previously failed/incomplete file: {file_id}")
            # Hash is needed for the worker, calculate if not skipped
            current_hash = calculate_hash(md_file)
            if not current_hash:
                 logger.warning(f"{file_id}: Could not hash, skipping.")
                 continue # Cannot process if unreadable
            files_to_process_paths.append(str(md_file))

    # --- Execute in Parallel with tqdm ---
    results: List[Dict[str, Any]] = []
    failed_details: List[Tuple[str, str]] = [] # Store (file_id, error_message)
    successful_files: List[str] = [] # Initialize list for successful files
    successful_count = 0
    # Use a dictionary to map futures back to their input paths if needed for error reporting on crash
    future_to_path: Dict[Future, str] = {}

    if not files_to_process_paths:
        logger.info("No markdown files need processing.")
    else:
        num_workers = min(args.workers, len(files_to_process_paths), os.cpu_count() or 1) # Ensure at least 1
        logger.info(f"Processing {len(files_to_process_paths)} files using {num_workers} workers...")

        # Disable tqdm progress bar in quiet mode unless specifically requested?
        # For now, let's show it always, as it's minimal output.
        # Use `disable=args.quiet` in tqdm call if you want to hide it in quiet mode.
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_path = {
                executor.submit(process_notebook, path_str, log_level): path_str
                for path_str in files_to_process_paths
            }

            # Process results as they complete, updating tqdm progress bar
            for future in tqdm(as_completed(future_to_path), total=len(files_to_process_paths), desc="Executing notebooks", unit="file"):
                path_str = future_to_path[future]
                file_id = str(Path(path_str).relative_to(DOCS_DIR))
                try:
                    result = future.result()
                    results.append(result) # Collect full results if needed later

                    # Process status for summary immediately
                    status = result["status"]
                    new_hash = result["new_hash"]
                    error_message = result.get("error_message")
                    elapsed = result.get("elapsed", 0)

                    processed_files += 1
                    # Update cache
                    cache_update_data = {"hash": new_hash, "status": status, "error": error_message}
                    if new_hash: # Only update cache if hashing was successful
                        cache[file_id] = cache_update_data
                    else:
                        logger.warning(f"Did not update cache for {file_id} due to earlier hashing error.")

                    if status == 'success':
                        successful_count += 1
                        successful_files.append(file_id) # Add successful file_id
                    else: # failed or unknown status
                        failed_files_count += 1
                        # Use a more concise error if available, else the full one
                        short_error = (error_message.splitlines()[0] if error_message else "Unknown error")
                        failed_details.append((file_id, short_error))
                        # Detailed error might already be logged by worker or logged to file

                except Exception as e:
                    # Critical errors in the worker process itself (crash)
                    logger.critical(f"CRITICAL error processing {file_id}: Worker crashed: {e}", exc_info=True)
                    # Add to failed list immediately
                    failed_files_count += 1
                    processed_files += 1 # It was attempted
                    failed_details.append((file_id, f"Worker process crashed: {e}"))
                    # Update cache to reflect failure
                    cache[file_id] = {"hash": calculate_hash(Path(path_str)), "status": "failed", "error": f"Worker process crashed: {e}"}

        save_cache(cache)

    # --- Final Report ---
    total_time = time.monotonic() - start_time
    # Use print for the final summary to ensure it appears cleanly after tqdm/logs
    print("\n--- Execution Summary ---")
    print(f"Total files found: {len(all_md_files)}")
    print(f"Processed: {processed_files}")
    print(f"Skipped (cached): {skipped_files}")
    print(f"Successful: {successful_count}")
    print(f"Failed: {failed_files_count}")
    if failed_details:
        print("\n--- Failed Notebooks ---")
        for file_id, error in failed_details:
            print(f"  - {file_id}: {error}")
            # Optionally print full error here if not too long, or direct to log file
            cache_entry = cache.get(file_id)
            full_error = cache_entry.get("error") if cache_entry else "Error details not in cache."
            # Limit printing very long errors to console summary
            if full_error and len(full_error) > 200 :
                 print(f"    (Full error details in notebook output or log file if enabled)")
            elif full_error and error != full_error: # Print if different from short error shown
                 print(f"    Details: {full_error}")

    print(f"\nTotal execution time: {total_time:.2f}s")
    print("-------------------------")

    # --- Output Successful Files in YAML ---
    completed_files = sorted(list(set(successful_files + skipped_cached_files)))

    if completed_files:
        print("--- Completed Files (Processed or Cached) (YAML) ---")
        print(yaml.dump({'completed_files': completed_files}, default_flow_style=False))
        print("-------------------------------------------------")

    return 1 if failed_files_count > 0 else 0

if __name__ == "__main__":
    # Make sure tqdm doesn't interfere with tracebacks on crash
    try:
        exit_code = main()
        exit(exit_code)
    except Exception as e:
        logger.exception("Unhandled exception during script execution.")
        exit(1)
