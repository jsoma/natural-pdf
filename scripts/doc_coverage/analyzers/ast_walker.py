"""AST-based extraction of method calls and attribute accesses from Python code."""

import ast
from dataclasses import dataclass


@dataclass
class MethodCall:
    """A method call or attribute access extracted from source code.

    Attributes:
        receiver: Variable name the method/attribute was accessed on (e.g., "page").
        method: Method or attribute name (e.g., "find" or "width").
        line: Line number in the source.
        is_call: True if this is a method call, False if attribute access.
    """

    receiver: str
    method: str
    line: int
    is_call: bool = True


class CallVisitor(ast.NodeVisitor):
    """AST visitor that extracts method calls and attribute accesses."""

    def __init__(self):
        self.calls: list[MethodCall] = []
        self._call_attributes: set[int] = set()  # Track attribute nodes that are calls

    def visit_Call(self, node: ast.Call) -> None:
        """Visit a function/method call node."""
        if isinstance(node.func, ast.Attribute):
            # Mark this attribute as being part of a call
            self._call_attributes.add(id(node.func))

            # Handle chained calls: page.find().below() -> two calls
            receiver = self._get_receiver_name(node.func.value)
            if receiver:
                self.calls.append(
                    MethodCall(
                        receiver=receiver,
                        method=node.func.attr,
                        line=node.lineno,
                        is_call=True,
                    )
                )

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Visit an attribute access node (e.g., page.width)."""
        # Skip if this attribute is part of a call (already handled)
        if id(node) in self._call_attributes:
            self.generic_visit(node)
            return

        # Only capture simple attribute accesses (not chained)
        receiver = self._get_receiver_name(node.value)
        if receiver:
            self.calls.append(
                MethodCall(
                    receiver=receiver,
                    method=node.attr,
                    line=node.lineno,
                    is_call=False,
                )
            )

        self.generic_visit(node)

    def _get_receiver_name(self, node: ast.AST) -> str | None:
        """Extract the receiver name from an AST node.

        Handles:
        - Simple names: page.find() -> "page"
        - Subscripts: pages[0].find() -> "pages[0]"
        - Chained calls: page.find().below() -> "page.find()"
        - Attributes: pdf.pages[0].find() -> "pdf.pages[0]"
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Subscript):
            # e.g., pages[0] or pdf.pages[0]
            base = self._get_receiver_name(node.value)
            if base:
                return f"{base}[]"
        elif isinstance(node, ast.Attribute):
            # e.g., pdf.pages or result.answer
            base = self._get_receiver_name(node.value)
            if base:
                return f"{base}.{node.attr}"
        elif isinstance(node, ast.Call):
            # e.g., page.find().below() - the result of find()
            if isinstance(node.func, ast.Attribute):
                return f"{node.func.attr}()"
        return None


def extract_calls(source: str) -> list[MethodCall]:
    """Extract method calls and attribute accesses from Python source code.

    Args:
        source: Python source code as a string.

    Returns:
        List of MethodCall objects. Empty list if parsing fails.
    """
    try:
        tree = ast.parse(source)
        visitor = CallVisitor()
        visitor.visit(tree)
        return visitor.calls
    except SyntaxError:
        return []
