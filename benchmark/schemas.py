"""
Benchmark Output Schemas

Defines the JSON schemas for benchmark output files:
- meta.json: Run metadata and configuration
- ground_truth.json: Natural PDF extraction results
- llm_results/*.json: LLM extraction results
- comparison.json: Comparison results between ground truth and LLM
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional


@dataclass
class TrapDefinition:
    """Definition of a trap in the benchmark PDF."""

    name: str
    description: str
    page: int
    expected_value: str
    trap_for: str  # What LLMs commonly output instead
    category: str  # "ocr_confusion", "normalization", "autocomplete"
    field_path: Optional[str] = None  # JSON path to extract this value
    bbox: Optional[tuple[float, float, float, float]] = None


@dataclass
class PageGroundTruth:
    """Ground truth data for a single page."""

    page_number: int
    data: list[dict[str, Any]]  # Extracted rows/records
    traps: list[TrapDefinition] = field(default_factory=list)
    extraction_time_ms: int = 0


@dataclass
class GroundTruth:
    """Complete ground truth for a PDF."""

    pdf_name: str
    pdf_path: str
    total_pages: int
    extracted_at: str
    extraction_method: str = "natural_pdf"
    pages: list[PageGroundTruth] = field(default_factory=list)
    aggregate_data: list[dict[str, Any]] = field(default_factory=list)
    schema_version: str = "1.0"
    is_trap: bool = False  # True if this is a trap PDF
    data_format: Literal["tabular", "structured"] = "structured"  # Data structure type
    total_extraction_time_ms: int = 0  # Total time for Natural PDF extraction across all pages

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "pdf_name": self.pdf_name,
            "pdf_path": self.pdf_path,
            "total_pages": self.total_pages,
            "extracted_at": self.extracted_at,
            "extraction_method": self.extraction_method,
            "schema_version": self.schema_version,
            "is_trap": self.is_trap,
            "data_format": self.data_format,
            "total_extraction_time_ms": self.total_extraction_time_ms,
            "pages": [
                {
                    "page_number": p.page_number,
                    "data": p.data,
                    "traps": [asdict(t) for t in p.traps],
                    "extraction_time_ms": p.extraction_time_ms,
                }
                for p in self.pages
            ],
            "aggregate_data": self.aggregate_data,
        }

    def save(self, path: str | Path) -> None:
        """Save ground truth to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "GroundTruth":
        """Load ground truth from JSON file."""
        with open(path) as f:
            data = json.load(f)

        pages = []
        for p in data.get("pages", []):
            traps = [TrapDefinition(**t) for t in p.get("traps", [])]
            pages.append(
                PageGroundTruth(
                    page_number=p["page_number"],
                    data=p["data"],
                    traps=traps,
                    extraction_time_ms=p.get("extraction_time_ms", 0),
                )
            )

        return cls(
            pdf_name=data["pdf_name"],
            pdf_path=data["pdf_path"],
            total_pages=data["total_pages"],
            extracted_at=data["extracted_at"],
            extraction_method=data.get("extraction_method", "natural_pdf"),
            pages=pages,
            aggregate_data=data.get("aggregate_data", []),
            schema_version=data.get("schema_version", "1.0"),
            is_trap=data.get("is_trap", False),
            data_format=data.get("data_format", "structured"),
            total_extraction_time_ms=data["total_extraction_time_ms"],
        )


@dataclass
class LLMResponse:
    """Raw response from an LLM API call."""

    page_number: int
    prompt: str
    raw_response: str
    parsed_data: list[dict[str, Any]]
    tokens_input: int = 0
    tokens_output: int = 0
    latency_ms: int = 0
    cached: bool = False


@dataclass
class TrapResult:
    """Result of checking a single trap."""

    trap_name: str
    expected: str
    actual: str
    passed: bool
    category: str
    page: int = 1
    description: str = ""


@dataclass
class LLMResult:
    """Complete LLM evaluation result for a PDF."""

    pdf_name: str
    model: str
    provider: str  # "openai", "google", "openrouter"
    evaluated_at: str
    prompt_template: str
    total_pages: int
    responses: list[LLMResponse] = field(default_factory=list)
    trap_results: list[TrapResult] = field(default_factory=list)
    accuracy_score: float = 0.0
    total_tokens: int = 0
    schema_version: str = "1.0"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "pdf_name": self.pdf_name,
            "model": self.model,
            "provider": self.provider,
            "evaluated_at": self.evaluated_at,
            "prompt_template": self.prompt_template,
            "total_pages": self.total_pages,
            "schema_version": self.schema_version,
            "responses": [
                {
                    "page_number": r.page_number,
                    "prompt": r.prompt,
                    "raw_response": r.raw_response,
                    "parsed_data": r.parsed_data,
                    "tokens_input": r.tokens_input,
                    "tokens_output": r.tokens_output,
                    "latency_ms": r.latency_ms,
                    "cached": r.cached,
                }
                for r in self.responses
            ],
            "trap_results": [asdict(t) for t in self.trap_results],
            "accuracy_score": self.accuracy_score,
            "total_tokens": self.total_tokens,
        }

    def save(self, path: str | Path) -> None:
        """Save LLM result to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "LLMResult":
        """Load LLM result from JSON file."""
        with open(path) as f:
            data = json.load(f)

        responses = [
            LLMResponse(
                page_number=r["page_number"],
                prompt=r["prompt"],
                raw_response=r["raw_response"],
                parsed_data=r["parsed_data"],
                tokens_input=r.get("tokens_input", 0),
                tokens_output=r.get("tokens_output", 0),
                latency_ms=r.get("latency_ms", 0),
                cached=r.get("cached", False),
            )
            for r in data.get("responses", [])
        ]

        trap_results = [TrapResult(**t) for t in data.get("trap_results", [])]

        return cls(
            pdf_name=data["pdf_name"],
            model=data["model"],
            provider=data["provider"],
            evaluated_at=data["evaluated_at"],
            prompt_template=data["prompt_template"],
            total_pages=data["total_pages"],
            responses=responses,
            trap_results=trap_results,
            accuracy_score=data.get("accuracy_score", 0.0),
            total_tokens=data.get("total_tokens", 0),
            schema_version=data.get("schema_version", "1.0"),
        )


@dataclass
class ModelSummary:
    """Summary stats for a single model."""

    model: str
    provider: str
    pdfs_evaluated: int
    total_traps: int
    traps_passed: int
    accuracy: float


@dataclass
class PDFSummary:
    """Summary stats for a single PDF."""

    pdf_name: str
    total_pages: int
    total_traps: int
    models_evaluated: list[str]
    best_model: str
    best_accuracy: float


@dataclass
class BenchmarkMeta:
    """Metadata for a benchmark run."""

    created_at: str
    updated_at: str
    benchmark_version: str = "1.0"
    natural_pdf_version: str = ""
    pdfs: list[PDFSummary] = field(default_factory=list)
    models: list[ModelSummary] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    progress: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "benchmark_version": self.benchmark_version,
            "natural_pdf_version": self.natural_pdf_version,
            "pdfs": [asdict(p) for p in self.pdfs],
            "models": [asdict(m) for m in self.models],
            "config": self.config,
            "progress": self.progress,
        }

    def save(self, path: str | Path) -> None:
        """Save metadata to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkMeta":
        """Load metadata from JSON file."""
        with open(path) as f:
            data = json.load(f)

        pdfs = [PDFSummary(**p) for p in data.get("pdfs", [])]
        models = [ModelSummary(**m) for m in data.get("models", [])]

        return cls(
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            benchmark_version=data.get("benchmark_version", "1.0"),
            natural_pdf_version=data.get("natural_pdf_version", ""),
            pdfs=pdfs,
            models=models,
            config=data.get("config", {}),
            progress=data.get("progress", {}),
        )


class BenchmarkOutput:
    """
    Manages the benchmark output directory structure.

    Structure:
        benchmark_output/
        ├── meta.json                    # Run metadata
        ├── index.html                   # Dashboard
        ├── atlanta_schools/
        │   ├── ground_truth.json
        │   ├── ground_truth.csv
        │   ├── llm_results/
        │   │   ├── gpt-5.2.json
        │   │   └── gemini-3-pro.json
        │   └── report_data.json
        └── ...
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def pdf_dir(self, pdf_name: str) -> Path:
        """Get directory for a specific PDF."""
        # Normalize name: remove .pdf, replace spaces
        clean_name = pdf_name.replace(".pdf", "").replace(" ", "_").lower()
        path = self.output_dir / clean_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def llm_results_dir(self, pdf_name: str) -> Path:
        """Get LLM results directory for a PDF."""
        path = self.pdf_dir(pdf_name) / "llm_results"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def pages_dir(self, pdf_name: str) -> Path:
        """Get directory for page images."""
        path = self.pdf_dir(pdf_name) / "pages"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def page_image_path(self, pdf_name: str, page_number: int) -> Path:
        """Get path for a specific page image."""
        return self.pages_dir(pdf_name) / f"page_{page_number:03d}.png"

    def ground_truth_json_path(self, pdf_name: str) -> Path:
        """Get path for ground truth JSON."""
        return self.pdf_dir(pdf_name) / "ground_truth.json"

    def ground_truth_csv_path(self, pdf_name: str) -> Path:
        """Get path for ground truth CSV."""
        return self.pdf_dir(pdf_name) / "ground_truth.csv"

    def llm_result_path(self, pdf_name: str, model: str) -> Path:
        """Get path for LLM result JSON."""
        # Normalize model name for filename
        clean_model = model.replace("/", "_").replace(":", "_").replace("@", "_")
        return self.llm_results_dir(pdf_name) / f"{clean_model}.json"

    def report_path(self, pdf_name: str) -> Path:
        """Get path for HTML report."""
        return self.pdf_dir(pdf_name) / "report.html"

    def meta_path(self) -> Path:
        """Get path for meta.json."""
        return self.output_dir / "meta.json"

    def index_path(self) -> Path:
        """Get path for index.html."""
        return self.output_dir / "index.html"

    def cache_dir(self) -> Path:
        """Get cache directory for LLM responses."""
        path = self.output_dir / "llm-cache"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def list_pdfs(self) -> list[str]:
        """List all PDFs that have been prepared."""
        pdfs = []
        for path in self.output_dir.iterdir():
            if path.is_dir() and not path.name.startswith("."):
                if (path / "ground_truth.json").exists():
                    pdfs.append(path.name)
        return sorted(pdfs)

    def list_models(self, pdf_name: str) -> list[str]:
        """List all models that have results for a PDF."""
        results_dir = self.llm_results_dir(pdf_name)
        models = []
        for path in results_dir.glob("*.json"):
            models.append(path.stem)
        return sorted(models)

    def get_or_create_meta(self) -> BenchmarkMeta:
        """Get existing meta or create new one."""
        meta_path = self.meta_path()
        if meta_path.exists():
            return BenchmarkMeta.load(meta_path)

        now = datetime.now().isoformat()
        return BenchmarkMeta(
            created_at=now,
            updated_at=now,
        )

    def save_meta(self, meta: BenchmarkMeta) -> None:
        """Save metadata."""
        meta.updated_at = datetime.now().isoformat()
        meta.save(self.meta_path())
