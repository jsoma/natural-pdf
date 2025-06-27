"""
Summary objects for describe functionality.
"""

from typing import Any, Dict, List, Union


class ElementSummary:
    """
    Container for element summary data with markdown rendering.

    Automatically renders as markdown in Jupyter notebooks and provides
    access to underlying data as dictionaries.
    """

    def __init__(self, data: Dict[str, Any], title: str = "Summary"):
        """
        Initialize summary with data and optional title.

        Args:
            data: Dictionary containing summary sections
            title: Title for the summary display
        """
        self.data = data
        self.title = title

    def __str__(self) -> str:
        """String representation as markdown."""
        return self._to_markdown()

    def __repr__(self) -> str:
        """Repr as markdown for better display."""
        return self._to_markdown()

    def _repr_markdown_(self) -> str:
        """Jupyter notebook markdown rendering."""
        return self._to_markdown()

    def to_dict(self) -> Dict[str, Any]:
        """Return underlying data as dictionary."""
        return self.data.copy()

    def _to_markdown(self) -> str:
        """Convert data to markdown format."""
        lines = [f"## {self.title}", ""]

        for section_name, section_data in self.data.items():
            lines.extend(self._format_section(section_name, section_data))
            lines.append("")  # Empty line between sections

        return "\n".join(lines).rstrip()

    def _format_section(self, name: str, data: Any) -> List[str]:
        """Format a single section as markdown."""
        # Use bold text instead of headers for more compact display
        section_title = name.replace("_", " ").title()

        if isinstance(data, dict):
            lines = [f"**{section_title}**:", ""]
            lines.extend(self._format_dict(data, indent=""))
        elif isinstance(data, list):
            lines = [f"**{section_title}**: {', '.join(str(item) for item in data)}"]
        else:
            lines = [f"**{section_title}**: {data}"]

        return lines

    def _format_dict(self, data: Dict[str, Any], indent: str = "") -> List[str]:
        """Format dictionary as markdown list."""
        lines = []

        for key, value in data.items():
            key_display = key.replace("_", " ")

            if isinstance(value, dict):
                # Nested dict - always format as list items
                lines.append(f"{indent}- **{key_display}**:")
                for subkey, subvalue in value.items():
                    subkey_display = subkey.replace("_", " ")
                    if isinstance(subvalue, dict):
                        # Another level of nesting
                        lines.append(f"{indent}  - **{subkey_display}**:")
                        for subsubkey, subsubvalue in subvalue.items():
                            subsubkey_display = subsubkey.replace("_", " ")
                            lines.append(f"{indent}    - {subsubkey_display}: {subsubvalue}")
                    else:
                        lines.append(f"{indent}  - {subkey_display}: {subvalue}")
            elif isinstance(value, list):
                if len(value) <= 5:
                    value_str = ", ".join(str(v) for v in value)
                    lines.append(f"{indent}- **{key_display}**: {value_str}")
                else:
                    lines.append(f"{indent}- **{key_display}**: {len(value)} items")
            else:
                lines.append(f"{indent}- **{key_display}**: {value}")

        return lines

    def _format_list(self, data: List[Any]) -> List[str]:
        """Format list as markdown."""
        lines = []
        for item in data:
            if isinstance(item, dict):
                # Could be table rows
                lines.append(f"- {item}")
            else:
                lines.append(f"- {item}")
        return lines

    def _format_horizontal_table(self, title: str, data: Dict[str, Any]) -> List[str]:
        """Format dict as horizontal table."""
        headers = list(data.keys())
        values = list(data.values())

        # Create table
        header_row = "| " + " | ".join(headers) + " |"
        separator = "|" + "|".join("------" for _ in headers) + "|"
        value_row = "| " + " | ".join(str(v) for v in values) + " |"

        return [f"- **{title}**:", "", header_row, separator, value_row, ""]

    # Added for better VS Code and other frontends support
    def _repr_html_(self) -> str:  # type: ignore
        """Return HTML representation so rich rendering works in more frontends.

        Many notebook frontends (including VS Code) give priority to the
        ``_repr_html_`` method over Markdown. When available, we convert the
        generated Markdown to HTML using the *markdown* library. If the
        library is not installed we simply wrap the Markdown in a ``<pre>``
        block so that at least the plain-text representation is visible.
        """
        md_source = self._to_markdown()
        try:
            import markdown as _markdown  # pylint: disable=import-error

            # Convert markdown to HTML. We explicitly enable tables so the
            # element and inspection summaries render nicely.
            return _markdown.markdown(md_source, extensions=["tables"])
        except Exception:  # noqa: BLE001, broad-except
            # Fallback: present the Markdown as-is inside a <pre> block.
            escaped = md_source.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            return f"<pre>{escaped}</pre>"


class InspectionSummary(ElementSummary):
    """
    Summary for element inspection with tabular data.
    """

    def _format_section(self, name: str, data: Any) -> List[str]:
        """Format inspection section with element tables."""
        section_title = name.replace("_", " ").title()

        if isinstance(data, dict) and "elements" in data:
            # This is an element table section - use ### header for inspect
            elements = data["elements"]
            lines = [f"### {section_title}"]
            if elements:
                lines.extend(self._format_element_table(elements, data.get("columns", [])))
                # Add note if truncated
                if "note" in data:
                    lines.append(f"_{data['note']}_")
            else:
                lines.append("No elements found.")
        else:
            # Regular section formatting
            lines = [f"**{section_title}**: {data}"]

        return lines

    def _format_element_table(
        self, elements: List[Dict[str, Any]], columns: List[str]
    ) -> List[str]:
        """Format elements as markdown table."""
        if not elements or not columns:
            return ["No elements to display."]

        lines = [""]  # Empty line before table

        # Table header
        header_row = "| " + " | ".join(columns) + " |"
        separator = "|" + "|".join("------" for _ in columns) + "|"
        lines.extend([header_row, separator])

        # Table rows
        for element in elements:
            row_values = []
            for col in columns:
                value = element.get(col, "")
                if value is None:
                    value = ""
                elif isinstance(value, float):
                    value = str(int(round(value)))
                elif isinstance(value, str) and len(value) > 50:
                    value = value[:50] + "..."
                row_values.append(str(value))

            row = "| " + " | ".join(row_values) + " |"
            lines.append(row)

        return lines
