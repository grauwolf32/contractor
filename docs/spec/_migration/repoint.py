"""Repoint links to legacy specification documents at their new owners.

Usage: python3 docs/spec/_migration/repoint.py BLOCKS_DIR [--write]

A link to a legacy document goes to the target that received most of that
document's words; a link with a section anchor goes to the target that
received most of that section's words. History-only sections point at
docs/decisions/spec-history.md. Prints every rewrite; --write applies them.
Also writes docs/decisions/spec-migration.md, the legacy-to-new map.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
LEGACY_NAME = r"(?:[0-9]{2}-[a-z0-9-]+|ui-user-stories|openapi-audit-scans|core-execution-model)\.md"
LINK = re.compile(r"(?P<path>(?:[\w.-]+/)*" + LEGACY_NAME + r")(?P<anchor>#[\w-]+)?")


def slug(heading: str) -> str:
    text = re.sub(r"[`*_]", "", heading.strip().lower())
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"\s", "-", text)


def destination(target: str) -> str:
    if target in ("decisions/spec-history.md", "decisions/open-questions.md") or target.startswith("proposals/"):
        return f"docs/{target}"
    if target == "drop":
        return ""
    return f"docs/spec/{target}"


def build_maps(blocks_dir: Path):
    ledger = {e["id"]: e for e in json.loads((HERE / "ledger.json").read_text())["entries"]}
    doc_words: dict[str, Counter] = defaultdict(Counter)
    section_words: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for path in blocks_dir.glob("*.json"):
        for block in json.loads(path.read_text()):
            entry = ledger[block["id"]]
            for target in entry["targets"]:
                dest = destination(target)
                if not dest:
                    continue
                share = block["words"] / len(entry["targets"])
                doc_words[block["doc"]][dest] += share
                for heading in block["path"]:
                    section_words[(block["doc"], slug(heading))][dest] += share
    doc_map = {doc: words.most_common(1)[0][0] for doc, words in doc_words.items()}
    section_map = {key: words.most_common(1)[0][0] for key, words in section_words.items()}
    return doc_map, section_map, doc_words


def legacy_file(link_path: str, source: Path) -> str | None:
    name = link_path.rsplit("/", 1)[-1]
    if "/" not in link_path:
        resolved = (source.parent / name).resolve()
    else:
        resolved = (source.parent / link_path).resolve()
    spec = (REPO / "docs" / "spec").resolve()
    if resolved.parent == spec:
        return name
    if link_path.startswith("docs/spec/") and "/" not in link_path[len("docs/spec/"):]:
        return name  # repository-relative reference in task files and configs
    return None


def main() -> int:
    blocks_dir = Path(sys.argv[1])
    write = "--write" in sys.argv[2:]
    doc_map, section_map, doc_words = build_maps(blocks_dir)
    files = subprocess.run(
        ["git", "grep", "-l", "-E", LEGACY_NAME.replace("(?:", "(")],
        cwd=REPO, capture_output=True, text=True, check=False,
    ).stdout.split()
    total = 0
    for rel in files:
        source = REPO / rel
        if source.parent == (REPO / "docs" / "spec") and LINK.fullmatch(source.name):
            continue  # legacy documents themselves are deleted
        if rel.startswith("docs/spec/_migration/"):
            continue
        text = source.read_text(encoding="utf-8")

        def replace(match: re.Match[str]) -> str:
            nonlocal total
            name = legacy_file(match.group("path"), source)
            if name is None or name not in doc_map:
                return match.group(0)
            anchor = (match.group("anchor") or "")[1:]
            dest = section_map.get((name, anchor), doc_map[name]) if anchor else doc_map[name]
            if match.group("path").startswith("docs/spec/"):
                new = dest  # keep repository-relative form
            else:
                new = os.path.relpath(REPO / dest, source.parent)
            total += 1
            print(f"{rel}: {match.group(0)} -> {new}")
            return new

        updated = LINK.sub(replace, text)
        if write and updated != text:
            source.write_text(updated, encoding="utf-8")
    print(f"{total} links {'rewritten' if write else 'to rewrite'}")

    # spec-history.md and open-questions.md quote legacy blocks verbatim; their
    # relative links were written from docs/spec/, so rebase them onto docs/decisions/.
    spec_dir, decisions = REPO / "docs" / "spec", REPO / "docs" / "decisions"
    for name in ("spec-history.md", "open-questions.md"):
        path = decisions / name
        text = path.read_text(encoding="utf-8")

        def rebase(match: re.Match[str]) -> str:
            href = match.group(1)
            if re.match(r"^[a-z]+:", href) or href.startswith(("#", "../spec/", "../proposals/")):
                return match.group(0)
            link, _, anchor = href.partition("#")
            target = Path(os.path.normpath(spec_dir / link))
            if target.parent == spec_dir and target.name in doc_map:
                dest = section_map.get((target.name, anchor), doc_map[target.name]) if anchor else doc_map[target.name]
                return f"]({os.path.relpath(REPO / dest, decisions)})"
            return f"]({os.path.relpath(target, decisions)}{'#' + anchor if anchor else ''})"

        updated = re.sub(r"\]\(([^)\s]+)\)", rebase, text)
        if write and updated != text:
            path.write_text(updated, encoding="utf-8")
        print(f"{name}: links rebased onto docs/decisions/")

    lines = [
        "# Specification migration map",
        "",
        "Where the content of each feature-ordered specification went in the",
        "2026-10 migration to the domain layout. Older task records and documents",
        "may still name the legacy files; use this map to find the current owner.",
        "",
        "| Legacy document | New documents (share of its words) |",
        "| --- | --- |",
    ]
    for doc in sorted(doc_words):
        words = doc_words[doc]
        total_words = sum(words.values()) or 1
        parts = [f"[{d.split('/', 1)[1]}](../{d.split('/', 1)[1]}) {round(100 * w / total_words)}%" for d, w in words.most_common() if w / total_words >= 0.03]
        lines.append(f"| `{doc}` | {', '.join(parts)} |")
    if write:
        (REPO / "docs" / "decisions" / "spec-migration.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
