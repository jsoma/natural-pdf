"""
Live progress display for benchmark evaluation.

Shows a table of PDFs and models with their status and scores that updates in real-time.
Uses the 'rich' library for terminal rendering.
"""

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text


class Status(Enum):
    WAITING = "waiting"
    PREPARING = "preparing"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


# Spinner frames for animation
SPINNER_FRAMES = ["◐", "◓", "◑", "◒"]
COMPLETED_BLOCK = "●"
ERROR_BLOCK = "✗"
WAITING_BLOCK = "○"
PREPARING_BLOCK = "◎"


@dataclass
class TaskProgress:
    """Progress state for a single PDF/model combination."""

    pdf_name: str
    model: str
    status: Status = Status.WAITING
    score: Optional[float] = None
    error: Optional[str] = None
    time_seconds: Optional[float] = None


class ProgressDisplay:
    """
    Live terminal display for benchmark progress.

    Shows only active tasks (up to max_display), replacing completed ones
    as new tasks start. Includes a summary of total progress.

    Example display:

    Progress: 5/24 completed
    PDF              Model                     Status       Score    Time
    ─────────────────────────────────────────────────────────────────────
    01-practice      ◐ gpt-4o                  running      -        -
    atlanta_schools  ◎ gemini-2.5-flash        preparing    -        -
    """

    def __init__(self, pdfs: list[str], models: list[str], max_display: int = 20):
        """
        Initialize progress display.

        Args:
            pdfs: List of PDF names to process
            models: List of model names to evaluate
            max_display: Maximum number of tasks to show at once
        """
        self.pdfs = pdfs
        self.models = models
        self.max_display = max_display
        self.tasks: dict[tuple[str, str], TaskProgress] = {}
        self._active_order: list[tuple[str, str]] = []  # Track order tasks became active

        # Initialize all tasks as waiting
        for pdf in pdfs:
            for model in models:
                key = (pdf, model)
                self.tasks[key] = TaskProgress(pdf_name=pdf, model=model)

        self._lock = threading.Lock()
        self._live: Optional[Live] = None
        self._console = Console()
        self._frame_index = 0
        self._stop_animation = False
        self._animation_thread: Optional[threading.Thread] = None

    def _get_spinner_frame(self) -> str:
        """Get current spinner frame."""
        return SPINNER_FRAMES[self._frame_index % len(SPINNER_FRAMES)]

    def _get_status_indicator(self, status: Status) -> tuple[str, str]:
        """Get status indicator character and style."""
        if status == Status.WAITING:
            return WAITING_BLOCK, "dim"
        elif status == Status.PREPARING:
            return PREPARING_BLOCK, "blue"
        elif status == Status.RUNNING:
            return self._get_spinner_frame(), "yellow"
        elif status == Status.COMPLETED:
            return COMPLETED_BLOCK, "green"
        elif status == Status.ERROR:
            return ERROR_BLOCK, "red"
        return "?", ""

    def _build_table(self) -> Table:
        """Build the progress table showing only active tasks."""
        # Count completed and total
        total_tasks = len(self.tasks)
        completed = sum(
            1 for t in self.tasks.values() if t.status in (Status.COMPLETED, Status.ERROR)
        )
        running = sum(1 for t in self.tasks.values() if t.status == Status.RUNNING)

        # Build header with progress info
        table = Table(
            show_header=True,
            header_style="bold",
            box=None,
            title=f"[bold]Progress: {completed}/{total_tasks} completed[/bold]  [dim]({running} running)[/dim]",
        )
        table.add_column("PDF", width=25)
        table.add_column("Model", width=30)
        table.add_column("Status", width=14)
        table.add_column("Score", width=8, justify="right")
        table.add_column("Time", width=8, justify="right")

        # Get tasks to display: active ones (running/preparing) first, then recent from active_order
        tasks_to_show: list[tuple[str, str]] = []

        # First add all currently running/preparing tasks
        for key, task in self.tasks.items():
            if task.status in (Status.RUNNING, Status.PREPARING):
                if key not in tasks_to_show:
                    tasks_to_show.append(key)

        # Then add from active_order (recently completed) to fill up to max_display
        for key in reversed(self._active_order):
            if len(tasks_to_show) >= self.max_display:
                break
            if key not in tasks_to_show:
                tasks_to_show.append(key)

        # Limit to max_display
        tasks_to_show = tasks_to_show[: self.max_display]

        for key in tasks_to_show:
            task = self.tasks[key]
            pdf, model = key

            # Truncate long names
            pdf_display = pdf[:23] + ".." if len(pdf) > 25 else pdf
            model_display = model[:28] + ".." if len(model) > 30 else model

            # Status indicator (spinner/block) + text combined
            indicator, indicator_style = self._get_status_indicator(task.status)
            status_str, status_style = self._format_status(task.status)
            status_combined = f"[{indicator_style}]{indicator}[/{indicator_style}] [{status_style}]{status_str}[/{status_style}]"

            # Score
            if task.score is not None:
                if task.score < 0:
                    score_str = "N/A"
                    score_style = "dim"
                else:
                    score_str = f"{task.score * 100:.0f}%"
                    score_style = self._score_style(task.score)
            else:
                score_str = "-"
                score_style = "dim"

            # Time
            if task.time_seconds is not None:
                time_str = f"{task.time_seconds:.1f}s"
            else:
                time_str = "-"

            table.add_row(
                pdf_display,
                model_display,
                status_combined,
                f"[{score_style}]{score_str}[/{score_style}]",
                f"[dim]{time_str}[/dim]",
            )

        return table

    def _animate(self) -> None:
        """Animation thread - updates spinner frames."""
        while not self._stop_animation:
            time.sleep(0.1)  # 10 FPS
            with self._lock:
                self._frame_index += 1
                if self._live:
                    self._live.update(self._build_table())

    def start(self) -> None:
        """Initialize the display."""
        self._live = Live(
            self._build_table(),
            console=self._console,
            refresh_per_second=10,
            transient=False,
        )
        self._live.start()

        # Start animation thread
        self._stop_animation = False
        self._animation_thread = threading.Thread(target=self._animate, daemon=True)
        self._animation_thread.start()

    def update(
        self,
        pdf_name: str,
        model: str,
        status: Status,
        score: Optional[float] = None,
        error: Optional[str] = None,
        time_seconds: Optional[float] = None,
    ) -> None:
        """Update a task's status."""
        with self._lock:
            key = (pdf_name, model)
            if key in self.tasks:
                old_status = self.tasks[key].status
                self.tasks[key].status = status
                if score is not None:
                    self.tasks[key].score = score
                if error is not None:
                    self.tasks[key].error = error
                if time_seconds is not None:
                    self.tasks[key].time_seconds = time_seconds

                # Track when tasks become active (running/preparing)
                if status in (Status.RUNNING, Status.PREPARING) and old_status == Status.WAITING:
                    if key not in self._active_order:
                        self._active_order.append(key)

            if self._live:
                self._live.update(self._build_table())

    def update_pdf(self, pdf_name: str, status: Status) -> None:
        """Update status for all models of a PDF (used for preparing stage)."""
        with self._lock:
            for model in self.models:
                key = (pdf_name, model)
                if key in self.tasks:
                    # Only update if still waiting
                    if self.tasks[key].status == Status.WAITING:
                        self.tasks[key].status = status
                        # Track when tasks become active
                        if status in (Status.RUNNING, Status.PREPARING):
                            if key not in self._active_order:
                                self._active_order.append(key)

            if self._live:
                self._live.update(self._build_table())

    def finish(self) -> None:
        """Clean up and show final state."""
        self._stop_animation = True
        if self._animation_thread:
            self._animation_thread.join(timeout=0.5)
        if self._live:
            self._live.stop()

    def _format_status(self, status: Status) -> tuple[str, str]:
        """Format status with appropriate style."""
        if status == Status.WAITING:
            return "waiting", "dim"
        elif status == Status.PREPARING:
            return "preparing", "blue"
        elif status == Status.RUNNING:
            return "running", "yellow"
        elif status == Status.COMPLETED:
            return "completed", "green"
        elif status == Status.ERROR:
            return "error", "red"
        return str(status.value), ""

    def _score_style(self, score: float) -> str:
        """Get style for score."""
        if score >= 0.9:
            return "green"
        elif score >= 0.7:
            return "yellow"
        else:
            return "red"


# Keep backward compatibility for single-PDF usage
class SinglePDFProgressDisplay(ProgressDisplay):
    """Simplified progress display for single PDF evaluation."""

    def __init__(self, models: list[str], pdf_name: str = ""):
        super().__init__([pdf_name or "PDF"], models)
        self._pdf_name = pdf_name or "PDF"

    def _build_table(self) -> Table:
        """Build the progress table (single PDF, no PDF column)."""
        table = Table(show_header=True, header_style="bold", box=None)
        table.add_column("Model", width=40)
        table.add_column("Status", width=14)
        table.add_column("Score", width=8, justify="right")
        table.add_column("Time", width=8, justify="right")

        for model in self.models:
            task = self.tasks[(self._pdf_name, model)]

            # Truncate long model names
            model_display = model[:38] + ".." if len(model) > 40 else model

            # Status indicator (spinner/block) + text combined
            indicator, indicator_style = self._get_status_indicator(task.status)
            status_str, status_style = self._format_status(task.status)
            status_combined = f"[{indicator_style}]{indicator}[/{indicator_style}] [{status_style}]{status_str}[/{status_style}]"

            # Score
            if task.score is not None:
                if task.score < 0:
                    score_str = "N/A"
                    score_style = "dim"
                else:
                    score_str = f"{task.score * 100:.0f}%"
                    score_style = self._score_style(task.score)
            else:
                score_str = "-"
                score_style = "dim"

            # Time
            if task.time_seconds is not None:
                time_str = f"{task.time_seconds:.1f}s"
            else:
                time_str = "-"

            table.add_row(
                model_display,
                status_combined,
                f"[{score_style}]{score_str}[/{score_style}]",
                f"[dim]{time_str}[/dim]",
            )

        return table

    def update(
        self,
        model: str,
        status: Status,
        score: Optional[float] = None,
        error: Optional[str] = None,
        time_seconds: Optional[float] = None,
    ) -> None:
        """Update a model's status (simplified interface for single PDF)."""
        super().update(self._pdf_name, model, status, score, error, time_seconds)
