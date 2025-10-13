"""Check that docstrings follow Google style for Args/Returns sections.

Each function/method docstring with an Args or Returns section should
have properly formatted blocks. This script scans all tracked Python files
and reports any violations.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

# Flexible Google-style section patterns.
ARGS_BLOCK = re.compile(
    r"(?ms)^[ \t]{0,4}Args:\n"
    r"(?:[ \t]{4,8}[A-Za-z_]\w* \(.+?\): .+(?:\n[ \t]{8,}.+)*)+"
)
RETURNS_BLOCK = re.compile(
    r"(?ms)^[ \t]{0,4}Returns:\n" r"(?:[ \t]{4,8}.+?: .+(?:\n[ \t]{8,}.+)*)+"
)

HAS_ARGS = re.compile(r"(?m)^[ \t]{0,4}Args:\s*$")
HAS_RETURNS = re.compile(r"(?m)^[ \t]{0,4}Returns:\s*$")


def list_repo_pyfiles() -> list[Path]:
    """Return tracked Python files (respects `--all-files` in pre-commit)."""
    try:
        out = subprocess.check_output(
            ["git", "ls-files", "-z", "--", "*.py"], text=False
        )
        return [
            Path(p)
            for p in out.decode("utf-8", "replace").split("\x00")
            if p and Path(p).exists()
        ]
    except Exception:
        return [p for p in Path(".").rglob("*.py") if p.exists()]


def _is_doc_expr(node: ast.AST) -> bool:
    """True if node is an Expr containing a string literal."""
    return (
        isinstance(node, ast.Expr)
        and isinstance(getattr(node, "value", None), ast.Constant)
        and isinstance(node.value.value, str)
    )


def iter_docstrings(tree: ast.AST) -> Iterable[Tuple[str, int, str]]:
    """Yield (kind, line_no, docstring) for module, classes, and functions."""
    # Module docstring
    if isinstance(tree, ast.Module):
        if tree.body and _is_doc_expr(tree.body[0]):
            doc = ast.get_docstring(tree, clean=False)
            if doc is not None:
                yield ("module", tree.body[0].lineno, doc)

    # Classes / Functions
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body and _is_doc_expr(node.body[0]):
                doc = ast.get_docstring(node, clean=False)
                if doc is not None:
                    kind = (
                        "function"
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                        else "class"
                    )
                    yield (kind, node.body[0].lineno, doc)


def check_doc(doc: str) -> list[str]:
    """Return list of messages describing formatting problems in this docstring."""
    msgs: list[str] = []
    if HAS_ARGS.search(doc) and not ARGS_BLOCK.search(doc):
        msgs.append("Args section badly formatted")
    if HAS_RETURNS.search(doc) and not RETURNS_BLOCK.search(doc):
        msgs.append("Returns section badly formatted")
    return msgs


def check_file(path: Path) -> list[str]:
    """Check one file; return formatted error lines with file:line: message."""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        return [f"{path}:0: failed to read file ({e})"]

    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as e:
        return [f"{path}:{getattr(e, 'lineno', 0)}: failed to parse file (SyntaxError)"]

    errors: list[str] = []
    for _kind, lineno, doc in iter_docstrings(tree):
        for msg in check_doc(doc):
            errors.append(f"{path}:{lineno}: {msg}")
    return errors


def main(argv: List[str]) -> int:
    """Check all tracked Python files for docstring formatting issues.

    Args:
        argv (List[str]): Command-line arguments (unused).

    Returns:
        int: Exit code (0 if all good, 1 if issues found).
    """
    pyfiles = list_repo_pyfiles()
    all_errors: list[str] = []
    for f in pyfiles:
        all_errors.extend(check_file(f))

    if all_errors:
        print("❌ Docstring style errors detected:\n")
        for line in sorted(set(f"  - {e}" for e in all_errors)):
            print(line)
        return 1

    print("✅ All docstrings follow Google Args/Returns format.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
