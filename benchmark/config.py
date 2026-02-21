"""
Benchmark Configuration

Supports configuration via:
1. benchmark.yaml or benchmark.json in working directory
2. ~/.natural-pdf/benchmark.yaml
3. Environment variables

Configuration options:
- API keys for providers
- Rate limits
- Default models
- Output paths
- Cost limits
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    """Configuration for a single provider."""

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    rate_limit_per_minute: int = 30
    max_retries: int = 3
    timeout: float = 120.0


@dataclass
class BenchmarkConfig:
    """Complete benchmark configuration."""

    # Output settings
    output_dir: str = "benchmark_output"
    cache_enabled: bool = False  # Disabled by default for accurate benchmarking

    # Provider configs
    openai: ProviderConfig = field(default_factory=ProviderConfig)
    google: ProviderConfig = field(default_factory=ProviderConfig)
    openrouter: ProviderConfig = field(default_factory=ProviderConfig)
    local: ProviderConfig = field(default_factory=ProviderConfig)

    # Default models
    default_models: list[str] = field(
        default_factory=lambda: [
            "gpt-5.2",
            # "gpt-5-mini",
            # "gpt-5-nano",
            "gemini-2.5-flash",
            "gemini-3-flash-preview",
            "gemini-3-pro-preview",
            "claude-haiku-4.5",
            "claude-sonnet-4.5",
            "claude-opus-4.5",
        ]
    )

    # Evaluation settings
    max_concurrent: int = 12  # Run all models in parallel (rate limiting is per-provider)
    max_pages_per_pdf: Optional[int] = None

    # Report settings
    open_report_after_generation: bool = True

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "BenchmarkConfig":
        """
        Load configuration from file or defaults.

        Search order:
        1. Explicit config_path if provided
        2. benchmark.yaml in current directory
        3. benchmark.json in current directory
        4. ~/.natural-pdf/benchmark.yaml
        5. Environment variables only
        """
        config_data = {}

        # Find config file
        search_paths = []
        if config_path:
            search_paths.append(Path(config_path))
        search_paths.extend(
            [
                Path("benchmark.yaml"),
                Path("benchmark.json"),
                Path.home() / ".natural-pdf" / "benchmark.yaml",
                Path.home() / ".natural-pdf" / "benchmark.json",
            ]
        )

        for path in search_paths:
            if path.exists():
                config_data = cls._load_file(path)
                break

        # Create config with file data
        config = cls._from_dict(config_data)

        # Override with environment variables
        config = cls._apply_env_overrides(config)

        return config

    @classmethod
    def _load_file(cls, path: Path) -> dict:
        """Load config from file."""
        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml

                with open(path) as f:
                    return yaml.safe_load(f) or {}
            except ImportError:
                # Fall back to JSON-style YAML parsing
                pass

        with open(path) as f:
            return json.load(f)

    @classmethod
    def _from_dict(cls, data: dict) -> "BenchmarkConfig":
        """Create config from dictionary."""
        config = cls()

        # Simple fields
        if "output_dir" in data:
            config.output_dir = data["output_dir"]
        if "cache_enabled" in data:
            config.cache_enabled = data["cache_enabled"]
        if "default_models" in data:
            config.default_models = data["default_models"]
        if "max_concurrent" in data:
            config.max_concurrent = data["max_concurrent"]
        if "max_pages_per_pdf" in data:
            config.max_pages_per_pdf = data["max_pages_per_pdf"]
        if "open_report_after_generation" in data:
            config.open_report_after_generation = data["open_report_after_generation"]

        # Provider configs - use dataclass field defaults to avoid duplication
        for provider_name in ["openai", "google", "openrouter", "local"]:
            if provider_name in data:
                provider_data = data[provider_name]
                # Build kwargs only for keys that are actually present in the config
                kwargs = {}
                if "api_key" in provider_data:
                    kwargs["api_key"] = provider_data["api_key"]
                if "base_url" in provider_data:
                    kwargs["base_url"] = provider_data["base_url"]
                if "rate_limit_per_minute" in provider_data:
                    kwargs["rate_limit_per_minute"] = provider_data["rate_limit_per_minute"]
                if "max_retries" in provider_data:
                    kwargs["max_retries"] = provider_data["max_retries"]
                if "timeout" in provider_data:
                    kwargs["timeout"] = provider_data["timeout"]
                provider_config = ProviderConfig(**kwargs)
                setattr(config, provider_name, provider_config)

        return config

    @classmethod
    def _apply_env_overrides(cls, config: "BenchmarkConfig") -> "BenchmarkConfig":
        """Apply environment variable overrides."""
        # API keys
        if os.environ.get("OPENAI_API_KEY"):
            config.openai.api_key = os.environ["OPENAI_API_KEY"]
        if os.environ.get("GOOGLE_API_KEY"):
            config.google.api_key = os.environ["GOOGLE_API_KEY"]
        if os.environ.get("OPENROUTER_API_KEY"):
            config.openrouter.api_key = os.environ["OPENROUTER_API_KEY"]
        if os.environ.get("LOCAL_MODEL_BASE_URL"):
            config.local.base_url = os.environ["LOCAL_MODEL_BASE_URL"]

        # Other overrides
        if os.environ.get("BENCHMARK_OUTPUT_DIR"):
            config.output_dir = os.environ["BENCHMARK_OUTPUT_DIR"]
        if os.environ.get("BENCHMARK_MAX_PAGES"):
            config.max_pages_per_pdf = int(os.environ["BENCHMARK_MAX_PAGES"])

        return config

    def get_provider_config(self, provider: str) -> ProviderConfig:
        """Get config for a specific provider."""
        return getattr(self, provider, ProviderConfig())

    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a provider."""
        return self.get_provider_config(provider).api_key

    def to_dict(self) -> dict:
        """Convert to dictionary for saving."""
        return {
            "output_dir": self.output_dir,
            "cache_enabled": self.cache_enabled,
            "default_models": self.default_models,
            "max_concurrent": self.max_concurrent,
            "max_pages_per_pdf": self.max_pages_per_pdf,
            "open_report_after_generation": self.open_report_after_generation,
            "openai": {
                "rate_limit_per_minute": self.openai.rate_limit_per_minute,
                "max_retries": self.openai.max_retries,
                "timeout": self.openai.timeout,
            },
            "google": {
                "rate_limit_per_minute": self.google.rate_limit_per_minute,
                "max_retries": self.google.max_retries,
                "timeout": self.google.timeout,
            },
            "openrouter": {
                "rate_limit_per_minute": self.openrouter.rate_limit_per_minute,
                "max_retries": self.openrouter.max_retries,
                "timeout": self.openrouter.timeout,
            },
            "local": {
                "base_url": self.local.base_url,
                "rate_limit_per_minute": self.local.rate_limit_per_minute,
                "max_retries": self.local.max_retries,
                "timeout": self.local.timeout,
            },
        }

    def save(self, path: str | Path) -> None:
        """Save configuration to file."""
        path = Path(path)
        data = self.to_dict()

        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml

                with open(path, "w") as f:
                    yaml.dump(data, f, default_flow_style=False)
                return
            except ImportError:
                pass

        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def create_sample_config(path: str = "benchmark.json") -> None:
    """Create a sample configuration file."""
    sample = {
        "output_dir": "benchmark_output",
        "cache_enabled": True,
        "default_models": ["gpt-4o", "gemini-1.5-pro"],
        "max_concurrent": 3,
        "max_pages_per_pdf": None,
        "open_report_after_generation": True,
        "openai": {
            "api_key": "sk-... (or set OPENAI_API_KEY env var)",
            "rate_limit_per_minute": 30,
            "max_retries": 3,
            "timeout": 120.0,
        },
        "google": {
            "api_key": "(or set GOOGLE_API_KEY env var)",
            "rate_limit_per_minute": 30,
            "max_retries": 3,
            "timeout": 120.0,
        },
        "openrouter": {
            "api_key": "(or set OPENROUTER_API_KEY env var)",
            "rate_limit_per_minute": 30,
            "max_retries": 3,
            "timeout": 120.0,
        },
        "local": {
            "base_url": "http://localhost:11434/v1 (or set LOCAL_MODEL_BASE_URL env var)",
            "rate_limit_per_minute": 30,
            "max_retries": 3,
            "timeout": 120.0,
        },
    }

    with open(path, "w") as f:
        json.dump(sample, f, indent=2)

    logger.info(f"Sample config created: {path}")
