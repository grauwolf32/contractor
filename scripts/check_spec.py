"""Check the reference specification under docs/spec.

Rules for every Markdown document under docs/spec:

- no task references: task IDs or links into tasks/;
- requirement IDs (``**ART-R12.**`` at the start of a paragraph) are unique,
  use the prefix the document declares (``Requirement prefix: `ART```), and
  every referenced requirement ID exists;
- registry entries (``LIM-…``, ``FAIL-…``, ``LIFE-…`` in the first cell of a
  registry table row) are unique and every referenced entry exists;
- every relative link points at an existing file.

IDs and links inside inline code or fenced code are examples and are not
checked as references.

Usage: python3 scripts/check_spec.py
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "docs" / "spec"
TASK_REF = re.compile(r"\b(?:V\d+-\d{3}[A-Z]?|MVP-\d{3})\b|\btasks/[\w./-]+")
PREFIX = re.compile(r"(?m)^Requirement prefix: `([A-Z]{2,4})`")
REQUIREMENT_DEF = re.compile(r"(?m)^(?:[-*]\s+)?\*\*([A-Z]{2,4})-R(\d+)\.\*\*")
REQUIREMENT_REF = re.compile(r"\b([A-Z]{2,4}-R\d+)\b")
REGISTRY_DEF = re.compile(r"(?m)^\|\s*`?((?:LIM|FAIL|LIFE)-[A-Z0-9]+-\d+)`?\s*\|")
REGISTRY_REF = re.compile(r"\b((?:LIM|FAIL|LIFE)-[A-Z0-9]+-\d+)\b")
LINK = re.compile(r"\]\(([^)\s]+)\)")
CODE = re.compile(r"```.*?```|`[^`\n]*`", re.S)


def documents(spec: Path = SPEC) -> dict[str, str]:
    return {path.relative_to(spec).as_posix(): path.read_text(encoding="utf-8") for path in sorted(spec.rglob("*.md"))}


def check(docs: Mapping[str, str], exists: Callable[[str], bool] = lambda _: True) -> list[str]:
    errors: list[str] = []
    requirements: dict[str, str] = {}
    registry: dict[str, str] = {}
    for name, text in docs.items():
        for number, line in enumerate(text.splitlines(), 1):
            for match in TASK_REF.finditer(line):
                errors.append(f"{name}:{number}: task reference {match.group(0)!r}; specs do not cite tasks")
        declared = PREFIX.search(text)
        for match in REQUIREMENT_DEF.finditer(text):
            rid = f"{match.group(1)}-R{match.group(2)}"
            if declared and match.group(1) != declared.group(1):
                errors.append(f"{name}: requirement {rid} does not use the document prefix {declared.group(1)}")
            if rid in requirements:
                errors.append(f"{name}: requirement {rid} is also defined in {requirements[rid]}")
            requirements.setdefault(rid, name)
        for match in REGISTRY_DEF.finditer(text):
            eid = match.group(1)
            if eid in registry:
                errors.append(f"{name}: registry entry {eid} is also defined in {registry[eid]}")
            registry.setdefault(eid, name)
    for name, text in docs.items():
        prose = CODE.sub("", text)  # IDs and links inside code spans are examples, not references
        for rid in sorted(set(REQUIREMENT_REF.findall(prose)) - set(requirements)):
            errors.append(f"{name}: unknown requirement {rid}")
        for eid in sorted(set(REGISTRY_REF.findall(prose)) - set(registry)):
            errors.append(f"{name}: unknown registry entry {eid}")
        for href in sorted(set(LINK.findall(prose))):
            path = href.split("#", 1)[0]
            if not path or re.match(r"^[a-z][a-z0-9+.-]*:", path):
                continue
            target = os.path.normpath(os.path.join(os.path.dirname(name), path))
            if not exists(target):
                errors.append(f"{name}: broken link {href}")
    return errors


def main(argv: Iterable[str] = ()) -> int:
    docs = documents()
    errors = check(docs, lambda target: (SPEC / target).exists())
    for error in errors:
        print(f"docs/spec/{error}", file=sys.stderr)
    if errors:
        print(f"{len(errors)} specification problem(s)", file=sys.stderr)
        return 1
    print(f"{len(docs)} specification documents are consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
