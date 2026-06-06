"""Python scripts uploaded into the sandbox and executed there.

Kept as raw string constants so they ship as a single tarball-friendly blob.
"""

from __future__ import annotations


# AST-removes ``failed_ptp`` test functions/methods from test files so they
# can't crash pytest collection.  Accepts a JSON config of the form
# ``{"failed_tests": {"file.py": ["TestClass::test_m", "test_f"]}}``.
REMOVE_TESTS_SCRIPT = r'''
import ast
import json
import sys
import os

def _parse_test_id(tid):
    """Parse 'file.py::Class::method' or 'file.py::function' into (file, parts)."""
    parts = tid.split("::")
    filepath = parts[0] if parts else ""
    remainder = parts[1:] if len(parts) > 1 else []
    return filepath, remainder

def _names_to_remove(parts_list):
    """Build a set of (class_name, method_name) or (None, func_name) to remove."""
    targets = set()
    for parts in parts_list:
        if len(parts) == 1:
            # file.py::test_func  or  file.py::TestClass
            targets.add((None, parts[0]))
        elif len(parts) >= 2:
            # file.py::TestClass::test_method
            targets.add((parts[0], parts[1]))
    return targets

def remove_tests_from_file(filepath, targets):
    """Remove test functions/methods matching targets from filepath using AST."""
    if not os.path.exists(filepath):
        return 0

    with open(filepath, "r") as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 0

    lines = source.splitlines(keepends=True)
    # Collect line ranges to remove (0-indexed)
    ranges_to_remove = []

    for node in ast.iter_child_nodes(tree):
        # Top-level function: match (None, func_name)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if (None, node.name) in targets:
                start = node.lineno - 1
                end = node.end_lineno  # end_lineno is 1-based inclusive
                ranges_to_remove.append((start, end))

        # Class: check methods inside
        elif isinstance(node, ast.ClassDef):
            # Check if entire class should be removed
            if (None, node.name) in targets:
                start = node.lineno - 1
                end = node.end_lineno
                ranges_to_remove.append((start, end))
                continue

            # Check individual methods
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if (node.name, child.name) in targets:
                        start = child.lineno - 1
                        end = child.end_lineno
                        ranges_to_remove.append((start, end))

    if not ranges_to_remove:
        return 0

    # Remove lines in reverse order to preserve line numbers
    ranges_to_remove.sort(reverse=True)
    for start, end in ranges_to_remove:
        # Also remove decorators above the function/method
        while start > 0 and lines[start - 1].strip().startswith("@"):
            start -= 1
        del lines[start:end]

    with open(filepath, "w") as f:
        f.writelines(lines)

    return len(ranges_to_remove)

def main():
    with open(sys.argv[1]) as f:
        config = json.load(f)

    failed_tests = config["failed_tests"]  # {filepath: [parts_list]}
    total_removed = 0

    for filepath, parts_list in failed_tests.items():
        targets = _names_to_remove(parts_list)
        if targets:
            n = remove_tests_from_file(filepath, targets)
            total_removed += n
            if n:
                print(f"Removed {n} test(s) from {filepath}")

    print(f"Total removed: {total_removed}")

if __name__ == "__main__":
    main()
'''


# Detects and uninstalls the local package (editable or non-editable).
# Tries setup.py, pyproject.toml, setup.cfg, and falls back to directory name.
UNINSTALL_PKG_SCRIPT = r'''
import ast, configparser, os, re, subprocess, sys

def get_pkg_name():
    """Try multiple strategies to find the package name."""
    # Strategy 1: setup.py
    if os.path.exists("setup.py"):
        try:
            with open("setup.py") as f:
                tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and hasattr(node, "keywords"):
                    for kw in node.keywords:
                        if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                            return kw.value.value
        except Exception:
            pass

    # Strategy 2: pyproject.toml
    if os.path.exists("pyproject.toml"):
        try:
            with open("pyproject.toml") as f:
                m = re.search(r'^\s*name\s*=\s*["\']([^"\']+)["\']', f.read(), re.MULTILINE)
                if m:
                    return m.group(1)
        except Exception:
            pass

    # Strategy 3: setup.cfg
    if os.path.exists("setup.cfg"):
        try:
            c = configparser.ConfigParser()
            c.read("setup.cfg")
            name = c.get("metadata", "name", fallback="")
            if name:
                return name
        except Exception:
            pass

    # Strategy 4: directory name
    return os.path.basename(os.getcwd())

name = get_pkg_name()
if name:
    # Also try common normalizations (underscores, hyphens)
    variants = {name, name.replace("-", "_"), name.replace("_", "-")}
    for v in variants:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", v],
                       capture_output=True)
    print(f"Uninstalled: {name}")
else:
    print("Could not determine package name, skipping uninstall")
'''
