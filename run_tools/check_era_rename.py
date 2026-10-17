#!/usr/bin/env python3
"""Leftovers of the `period` -> `era` rename in a repository's tracked files.

Reports, with file and line:
* a Python identifier, attribute, argument or keyword that contains `period`, except cache
  validity periods and the timing periods in ALLOWED_NAMES;
* the era tuple used as a string: `{self.era}` in an f-string, `str(self.era)`,
  `"...".join(self.era)`;
* `getattr(x, "period", ...)` and `x.get("period", ...)` (the attribute or key of the era);
* `--period` and `{period}` in Python, YAML, shell and configuration files.

The tuple checks apply inside classes derived from a `...Task` or `...Workflow` class, where
`self.era` is the era parameter; elsewhere (an analysis' dataframe builder, a correction)
`self.era` is a single name. The AnaTuple column `period` and the C++ `Period` enum are
strings and C++ code, which this check does not look at. A line ending in a
`noqa: era-rename` comment is skipped (the code that refuses the old names), and so are the
files given with --exclude (glob patterns relative to the repository). Exits 1 when something
is found.
"""

import argparse
import ast
import fnmatch
import os
import re
import subprocess
import sys

ALLOWED_NAMES = {
    "validity_period",
    "validity_periods",
    "cache_validity_period",
    "local_path_cache_validity_period",
    "remote_path_cache_validity_period",
    "listing_period",
    "beat_period",
    "grace_period",
}
NOQA = "noqa: era-rename"
TEXT_SUFFIXES = (".py", ".yaml", ".yml", ".sh", ".cfg", ".ini", ".json")
TEXT_PATTERNS = {
    "--period": re.compile(r"--period\b"),
    "{period}": re.compile(r"\{period\}"),
}


def is_self_era(node):
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "era"
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )


def is_task_class(node):
    for base in node.bases:
        name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", "")
        if name.endswith(("Task", "Workflow")):
            return True
    return False


def task_class_nodes(tree):
    """The nodes inside classes that are law tasks."""
    inside = set()
    for cls in ast.walk(tree):
        if isinstance(cls, ast.ClassDef) and is_task_class(cls):
            inside.update(id(node) for node in ast.walk(cls))
    return inside


def python_findings(path, source):
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as e:
        return [(e.lineno or 0, f"cannot parse: {e.msg}")]
    found = []
    in_task = task_class_nodes(tree)

    def name(node, value):
        if "period" in value and value not in ALLOWED_NAMES and "validity" not in value:
            found.append((node.lineno, f"identifier '{value}'"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            name(node, node.id)
        elif isinstance(node, ast.Attribute):
            name(node, node.attr)
        elif isinstance(node, ast.arg):
            name(node, node.arg)
        elif isinstance(node, ast.keyword) and node.arg:
            name(node, node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name(node, node.name)
        tuple_check = id(node) in in_task
        if tuple_check and isinstance(node, ast.FormattedValue):
            if is_self_era(node.value):
                found.append((node.lineno, "the era tuple formatted as a string"))
        if isinstance(node, ast.Call):
            func = node.func
            if tuple_check and isinstance(func, ast.Name) and func.id == "str":
                if node.args and is_self_era(node.args[0]):
                    found.append((node.lineno, "str(self.era)"))
            if tuple_check and isinstance(func, ast.Attribute) and func.attr == "join":
                if node.args and is_self_era(node.args[0]):
                    found.append((node.lineno, "join(self.era)"))
            is_getattr = isinstance(func, ast.Name) and func.id == "getattr"
            is_get = isinstance(func, ast.Attribute) and func.attr == "get"
            key_index = 1 if is_getattr else 0
            if (is_getattr or is_get) and len(node.args) > key_index:
                key = node.args[key_index]
                if isinstance(key, ast.Constant) and key.value == "period":
                    found.append(
                        (
                            node.lineno,
                            f"{'getattr' if is_getattr else 'get'}(..., 'period')",
                        )
                    )
    return found


def text_findings(source):
    found = []
    for number, line in enumerate(source.splitlines(), 1):
        for label, pattern in TEXT_PATTERNS.items():
            if pattern.search(line):
                found.append((number, f"'{label}'"))
    return found


def tracked_files(repo):
    out = subprocess.run(
        ["git", "-C", repo, "ls-files", "-z"],
        check=True,
        capture_output=True,
    ).stdout.decode()
    # gitlinks (submodules) are listed but are directories: another repository's business
    return [p for p in out.split("\0") if p and os.path.isfile(os.path.join(repo, p))]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", default=".", help="repository to check")
    parser.add_argument("--exclude", action="append", default=[], help="glob to skip")
    args = parser.parse_args()
    excludes = args.exclude + [os.path.relpath(__file__, args.repo)]
    problems = []
    for path in tracked_files(args.repo):
        if any(fnmatch.fnmatch(path, pattern) for pattern in excludes):
            continue
        if not path.endswith(TEXT_SUFFIXES):
            continue
        with open(os.path.join(args.repo, path), errors="replace") as f:
            source = f.read()
        found = text_findings(source)
        if path.endswith(".py"):
            found += python_findings(path, source)
        lines = source.splitlines()
        problems += [
            f"{path}:{line}: {what}"
            for line, what in sorted(set(found))
            if not (0 < line <= len(lines) and NOQA in lines[line - 1])
        ]
    for problem in problems:
        print(problem)
    if problems:
        print(
            f"{len(problems)} leftover(s) of the period -> era rename", file=sys.stderr
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
