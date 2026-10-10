"""Check the reference specification under docs/spec.

Rules for every migrated document (legacy NN-*.md files and the migration
working area are exempt until they are gone):

- no task references: task IDs or links into tasks/;
- requirement IDs (``**ART-R12.**`` at the start of a paragraph) are unique
  and every referenced requirement ID exists;
- registry entries (``LIM-…``, ``FAIL-…``, ``LIFE-…`` in the first cell of a
  registry table row) are unique and every referenced entry exists.

IDs inside inline code or fenced code are examples and are not checked as
references.

Usage: python3 scripts/check_spec.py
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "docs" / "spec"
LEGACY = re.compile(r"^(\d{2}-.*|ui-user-stories|openapi-audit-scans|core-execution-model)\.md$")
TASK_REF = re.compile(r"\b(?:V\d+-\d{3}[A-Z]?|MVP-\d{3})\b|\btasks/[\w./-]+")
REQUIREMENT_DEF = re.compile(r"(?m)^(?:[-*]\s+)?\*\*([A-Z]{2,4}-R\d+)\.\*\*")
REQUIREMENT_REF = re.compile(r"\b([A-Z]{2,4}-R\d+)\b")
REGISTRY_DEF = re.compile(r"(?m)^\|\s*`?((?:LIM|FAIL|LIFE)-[A-Z0-9]+-\d+)`?\s*\|")
REGISTRY_REF = re.compile(r"\b((?:LIM|FAIL|LIFE)-[A-Z0-9]+-\d+)\b")
CODE = re.compile(r"```.*?```|`[^`\n]*`", re.S)


def migrated_documents(spec: Path = SPEC) -> dict[str, str]:
    docs: dict[str, str] = {}
    for path in sorted(spec.rglob("*.md")):
        relative = path.relative_to(spec).as_posix()
        if relative.startswith("_migration/") or LEGACY.match(relative):
            continue
        docs[relative] = path.read_text(encoding="utf-8")
    return docs


def check(docs: Mapping[str, str]) -> list[str]:
    errors: list[str] = []
    requirements: dict[str, str] = {}
    registry: dict[str, str] = {}
    for name, text in docs.items():
        for number, line in enumerate(text.splitlines(), 1):
            for match in TASK_REF.finditer(line):
                errors.append(f"{name}:{number}: task reference {match.group(0)!r}; specs do not cite tasks")
        for match in REQUIREMENT_DEF.finditer(text):
            rid = match.group(1)
            if rid in requirements:
                errors.append(f"{name}: requirement {rid} is also defined in {requirements[rid]}")
            requirements.setdefault(rid, name)
        for match in REGISTRY_DEF.finditer(text):
            eid = match.group(1)
            if eid in registry:
                errors.append(f"{name}: registry entry {eid} is also defined in {registry[eid]}")
            registry.setdefault(eid, name)
    for name, text in docs.items():
        prose = CODE.sub("", text)  # IDs inside code spans are examples, not references
        for rid in sorted(set(REQUIREMENT_REF.findall(prose)) - set(requirements)):
            errors.append(f"{name}: unknown requirement {rid}")
        for eid in sorted(set(REGISTRY_REF.findall(prose)) - set(registry)):
            errors.append(f"{name}: unknown registry entry {eid}")
    return errors


def main(argv: Iterable[str] = ()) -> int:
    docs = migrated_documents()
    errors = check(docs)
    for error in errors:
        print(f"docs/spec/{error}", file=sys.stderr)
    if errors:
        print(f"{len(errors)} specification problem(s)", file=sys.stderr)
        return 1
    print(f"{len(docs)} migrated specification documents are consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
