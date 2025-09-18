"""Extended prompt-math parser with scheduling awareness.

The parser understands nested bracket expressions, binary operators, and inline
schedule annotations that attach timing metadata to tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

__all__ = ["ScheduleCall", "ASTNode", "ExtendedParser"]


@dataclass(slots=True)
class ScheduleCall:
    """Describes a parsed schedule invocation."""

    function_name: str
    args: tuple[object, ...] = ()
    kwargs: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ASTNode:
    """Simple AST representation for prompt-math expressions."""

    kind: str
    value: str | None = None
    children: List["ASTNode"] = field(default_factory=list)
    schedule: ScheduleCall | None = None

    @property
    def is_leaf(self) -> bool:
        """bool: ``True`` when the node represents a token."""

        return self.kind == "token"


class ExtendedParser:
    """Parse bracketed prompt expressions with optional scheduling."""

    _BINARY_OPERATORS = ("-", "+", "*")

    def __init__(self, expression: str) -> None:
        """Initialise the parser.

        Args:
            expression (str): Prompt-math expression such as ``"[[[a]-[b]]]"``.

        Raises:
            ValueError: If ``expression`` is empty.
        """

        if not expression:
            raise ValueError("Expression must not be empty.")
        self._expression = expression

    def parse(self) -> ASTNode:
        """Produce an abstract syntax tree for the expression.

        Returns:
            ASTNode: Root node representing the parsed expression.
        """

        inner = self._strip_outer(self._expression)
        token_part, schedule_part = self._split_top_level(inner, "@")
        if schedule_part is not None:
            token_node = self._parse_operand(token_part)
            schedule_call = self._parse_schedule(schedule_part)
            token_node.schedule = schedule_call
            return token_node
        for operator in self._BINARY_OPERATORS:
            left, right = self._split_top_level(inner, operator)
            if right is not None:
                left_node = self._parse_operand(left)
                right_node = self._parse_operand(right)
                return ASTNode(kind="op", value=operator, children=[left_node, right_node])
        return self._parse_operand(inner)

    def _parse_operand(self, text: str) -> ASTNode:
        """Parse a token or parenthesised/binary expression."""

        stripped = self._strip_outer(text)
        if self._looks_like_binary(stripped):
            return ExtendedParser(stripped).parse()
        token = self._parse_token(stripped)
        return ASTNode(kind="token", value=token)

    def _parse_token(self, text: str) -> str:
        """Extract a bare token identifier from ``text``.

        Args:
            text (str): Raw text including surrounding brackets.

        Returns:
            str: Token identifier.

        Raises:
            ValueError: If the token literal is empty.
        """

        stripped = text.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            stripped = stripped[1:-1].strip()
        if not stripped:
            raise ValueError("Empty token expression encountered.")
        return stripped

    def _parse_schedule(self, text: str) -> ScheduleCall:
        """Parse a schedule suffix such as ``"fade_in(0.2, 0.8)"``."""

        stripped = text.strip()
        if not stripped:
            raise ValueError("Schedule declaration is empty.")
        if "(" not in stripped or not stripped.endswith(")"):
            raise ValueError(f"Invalid schedule syntax: {text!r}")
        name, arg_text = stripped.split("(", 1)
        args, kwargs = self._parse_arguments(arg_text[:-1])
        return ScheduleCall(function_name=name.strip(), args=tuple(args), kwargs=kwargs)

    def _parse_arguments(self, text: str) -> tuple[list[object], dict[str, object]]:
        """Split and coerce schedule arguments into args/kwargs containers."""

        if not text.strip():
            return [], {}
        args: list[object] = []
        kwargs: dict[str, object] = {}
        for chunk in self._split_arguments(text):
            if "=" in chunk:
                key, value = chunk.split("=", 1)
                kwargs[key.strip()] = self._coerce_argument(value.strip())
            else:
                args.append(self._coerce_argument(chunk.strip()))
        return args, kwargs

    def _split_arguments(self, text: str) -> List[str]:
        """Split a function argument list while respecting nested delimiters."""

        parts: List[str] = []
        depth = 0
        current: List[str] = []
        for char in text:
            if char == "," and depth == 0:
                parts.append("".join(current).strip())
                current = []
                continue
            if char in "[(":
                depth += 1
            elif char in ")]":
                depth = max(0, depth - 1)
            current.append(char)
        if current:
            parts.append("".join(current).strip())
        return [part for part in parts if part]

    def _coerce_argument(self, value: str) -> object:
        """Coerce string arguments into native Python types."""

        if value.startswith("\"") and value.endswith("\""):
            return value[1:-1]
        if value.startswith("'") and value.endswith("'"):
            return value[1:-1]
        try:
            if "." in value or "e" in value.lower():
                return float(value)
            return int(value)
        except ValueError:
            lowered = value.lower()
            if lowered == "true":
                return True
            if lowered == "false":
                return False
            return value

    def _split_top_level(self, text: str, separator: str) -> tuple[str, str | None]:
        """Split ``text`` around the first top-level ``separator``."""

        depth = 0
        for index, char in enumerate(text):
            if char == "[":
                depth += 1
            elif char == "]":
                depth = max(0, depth - 1)
            elif char in "()":
                depth += 1 if char == "(" else -1
            elif char == separator and depth == 0:
                return text[:index], text[index + 1 :]
        return text, None

    def _strip_outer(self, text: str) -> str:
        """Remove wrapping brackets from ``text`` when they form a pair."""

        stripped = text.strip()
        while stripped.startswith("["):
            closing = self._matching_bracket_index(stripped, 0)
            if closing != len(stripped) - 1:
                break
            stripped = stripped[1:-1].strip()
        return stripped

    def _matching_bracket_index(self, text: str, start_index: int) -> int:
        """Locate the closing bracket that matches ``start_index``."""

        depth = 0
        for index, char in enumerate(text[start_index:], start=start_index):
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    return index
        return len(text) - 1

    def _looks_like_binary(self, text: str) -> bool:
        """bool: ``True`` when ``text`` contains a top-level binary operator."""

        for operator in self._BINARY_OPERATORS:
            if self._split_top_level(text, operator)[1] is not None:
                return True
        return False
