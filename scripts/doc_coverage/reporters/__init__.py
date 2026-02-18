"""Report generators for documentation coverage results."""

from .html_report import write_html
from .json_report import write_json
from .terminal import print_report

__all__ = ["print_report", "write_json", "write_html"]
