import re
import ssl
import urllib.request
from pathlib import Path
from typing import Optional

from rich.console import Console

ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # project root
BAD_PDF_DIR = ROOT_DIR / "bad_pdf_analysis"
SUBMISSIONS_DIR = ROOT_DIR / "bad-pdfs" / "submissions"

console = Console()


def slugify(value: str, max_length: int = 50) -> str:
    """Make a filesystem-safe filename from arbitrary text."""
    value = re.sub(r"[^\w\-\. ]+", "_", value)
    value = value.strip().replace(" ", "_")
    return value[:max_length]


def _search_directory(directory: Path, pattern: str, predicate) -> Optional[Path]:
    """Utility: recursively search *directory* using glob *pattern*; return first match passing *predicate*."""
    for p in directory.glob(pattern):
        try:
            if predicate(p):
                return p
        except Exception:
            # Just in case of any weird permission/path errors – skip
            continue
    return None


def find_local_pdf(submission_id: str, pdf_url: Optional[str] = None) -> Optional[Path]:
    """Return the local path to the PDF for *submission_id*.

    Search strategy (in order):
    1. Inside ``bad_pdf_analysis`` where early analyses live – matching *submission_id* in filename.
    2. Inside ``bad-pdfs/submissions`` where raw downloads reside – matching *submission_id*.
    3. If *pdf_url* is supplied, also try the basename of the URL in ``bad-pdfs/submissions``.
    4. As a last resort – try to download the PDF to the submissions folder
    """

    submission_id_lower = submission_id.lower()

    # 1) Search the processed-analysis folder (legacy path)
    path = _search_directory(
        BAD_PDF_DIR,
        f"**/{submission_id}*.pdf",
        lambda p: submission_id_lower in p.stem.lower(),
    )
    if path:
        return path

    # 2) Search the raw submissions folder by id substring
    path = _search_directory(
        SUBMISSIONS_DIR,
        f"**/*{submission_id}*.pdf",
        lambda p: submission_id_lower in p.stem.lower(),
    )
    if path:
        return path

    # 3) Use basename from URL, if provided
    if pdf_url:
        # Extract filename portion before any query string
        from urllib.parse import urlparse

        parsed = urlparse(pdf_url)
        filename = Path(parsed.path).name
        if filename:
            candidate = SUBMISSIONS_DIR / filename
            if candidate.exists():
                return candidate
            # fallback: case-insensitive glob match on stem
            stem = Path(filename).stem.lower()
            path = _search_directory(
                SUBMISSIONS_DIR,
                f"**/{stem}*.pdf",
                lambda p: stem in p.stem.lower(),
            )
            if path:
                return path

    # 4) As a last resort – try to download the PDF to the submissions folder
    if pdf_url:
        try:
            SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
            from urllib.parse import urlparse

            parsed = urlparse(pdf_url)
            filename = Path(parsed.path).name or f"{submission_id}.pdf"
            # Sanitise filename a bit – avoid query strings leaking in
            filename = slugify(filename, max_length=100)
            if not filename.lower().endswith(".pdf"):
                filename += ".pdf"

            dest_path = SUBMISSIONS_DIR / filename
            if not dest_path.exists():
                # Retrieve the file (no progress bar – keep it simple and robust)
                try:
                    # Some hosts reject default Python user-agent; set one.
                    req = urllib.request.Request(pdf_url, headers={"User-Agent": "Mozilla/5.0"})
                    # Disable SSL verification edge-cases the storage host sometimes triggers
                    ctx = ssl.create_default_context()
                    with urllib.request.urlopen(req, context=ctx, timeout=30) as resp, open(
                        dest_path, "wb"
                    ) as f:
                        f.write(resp.read())
                except Exception as e:
                    # Fallback: try requests if available (venv usually has it)
                    try:
                        import requests

                        r = requests.get(pdf_url, timeout=30)
                        r.raise_for_status()
                        with open(dest_path, "wb") as f:
                            f.write(r.content)
                    except Exception as e2:
                        console.print(f"[red]Download failed for {submission_id}: {e2}")
                        return None
            if dest_path.exists():
                return dest_path
        except Exception:
            # Networking problems, permissions, etc. – silently give up; caller will log
            return None

    # None found
    return None
