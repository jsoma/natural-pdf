import json
from pathlib import Path
from typing import Any, Dict

from rich.console import Console

console = Console()


def save_json(data: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def log_section(title: str):
    console.rule(f"[bold cyan]{title}")
