"""Split legacy specification documents into addressable blocks.

A block is one paragraph, one list, one table, one fenced code block or one
blockquote, together with the heading path above it. Block IDs are
``<doc>#<n>`` in document order (``03#17``); ``sha`` is the SHA-1 of the
block text and lets a later migration step recognise blocks that were edited
after the ledger was built.

Usage: python3 docs/spec/_migration/split.py OUT_DIR [DOC ...]
Writes OUT_DIR/<doc>.json and prints a summary. Without DOC arguments every
legacy document in docs/spec is split.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

SPEC = Path(__file__).resolve().parent.parent
LEGACY = re.compile(r"^(\d{2})-.*\.md$")
NAMED = {
    "README.md": "readme",
    "core-execution-model.md": "core",
    "openapi-audit-scans.md": "oas",
    "ui-user-stories.md": "uis",
}
HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")


def doc_key(name: str) -> str | None:
    match = LEGACY.match(name)
    return match.group(1) if match else NAMED.get(name)


def legacy_docs() -> list[Path]:
    return sorted(p for p in SPEC.glob("*.md") if doc_key(p.name))


def kind_of(lines: list[str]) -> str:
    first = lines[0]
    if first.lstrip().startswith("```") or first.lstrip().startswith("~~~"):
        return "code"
    if first.lstrip().startswith("|"):
        return "table"
    if first.lstrip().startswith(">"):
        return "quote"
    if LIST_ITEM.match(first):
        return "list"
    return "para"


def split(path: Path) -> list[dict]:
    key = doc_key(path.name)
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks: list[dict] = []
    headings: list[tuple[int, str]] = []
    current: list[str] = []
    start = 0
    in_fence = False
    fence = ""

    def flush(end: int) -> None:
        nonlocal current
        while current and not current[-1].strip():
            current.pop()
        if current:
            text = "\n".join(current)
            blocks.append(
                {
                    "id": f"{key}#{len(blocks) + 1}",
                    "doc": path.name,
                    "lines": [start + 1, end],
                    "path": [h for _, h in headings],
                    "kind": kind_of(current),
                    "words": len(text.split()),
                    "sha": hashlib.sha1(text.encode("utf-8")).hexdigest()[:12],
                    "text": text,
                }
            )
        current = []

    for number, line in enumerate(lines):
        stripped = line.strip()
        if in_fence:
            current.append(line)
            if stripped.startswith(fence):
                in_fence = False
                flush(number + 1)
            continue
        if stripped.startswith("```") or stripped.startswith("~~~"):
            flush(number)
            in_fence = True
            fence = stripped[:3]
            start = number
            current = [line]
            continue
        heading = HEADING.match(line)
        if heading:
            flush(number)
            level = len(heading.group(1))
            headings = [h for h in headings if h[0] < level] + [(level, heading.group(2).strip())]
            continue
        if not stripped:
            # A blank line ends a paragraph, but a list or table continues
            # while the next non-blank line still belongs to it.
            if current and kind_of(current) in ("list", "table"):
                nxt = next((l for l in lines[number + 1 :] if l.strip()), "")
                if kind_of(current) == "list" and (LIST_ITEM.match(nxt) or nxt.startswith("  ")):
                    current.append(line)
                    continue
            flush(number)
            continue
        if not current:
            start = number
        elif kind_of(current) == "table" and not stripped.startswith("|"):
            flush(number)
            start = number
        current.append(line)
    flush(len(lines))
    return blocks


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    names = sys.argv[2:]
    docs = [SPEC / n for n in names] if names else legacy_docs()
    total_blocks = total_words = 0
    for path in docs:
        blocks = split(path)
        (out / f"{doc_key(path.name)}.json").write_text(
            json.dumps(blocks, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        words = sum(b["words"] for b in blocks)
        total_blocks += len(blocks)
        total_words += words
        print(f"{path.name:52} {len(blocks):5} blocks {words:7} words")
    print(f"{'total':52} {total_blocks:5} blocks {total_words:7} words")
    return 0


if __name__ == "__main__":
    sys.exit(main())
