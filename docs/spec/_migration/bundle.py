"""Build one source bundle per target document from the blocks and the ledger.

Usage: python3 docs/spec/_migration/bundle.py BLOCKS_DIR OUT_DIR
Reads BLOCKS_DIR/<doc>.json (from split.py) and docs/spec/_migration/ledger.json,
writes OUT_DIR/<target with / as __>.md and OUT_DIR/index.json.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTLINE = re.compile(r"^\| `([^`]+)` \| (\w+|—) \| (.+) \|$")


def outline() -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for line in (HERE / "README.md").read_text(encoding="utf-8").splitlines():
        match = OUTLINE.match(line)
        if match:
            rows[match.group(1)] = {"prefix": match.group(2), "scope": match.group(3)}
    return rows


def main() -> int:
    blocks_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    ledger = {e["id"]: e for e in json.loads((HERE / "ledger.json").read_text(encoding="utf-8"))["entries"]}
    blocks = {}
    order = []
    for path in sorted(blocks_dir.glob("*.json")):
        for block in json.loads(path.read_text(encoding="utf-8")):
            blocks[block["id"]] = block
            order.append(block["id"])
    by_target: dict[str, list[str]] = defaultdict(list)
    for bid in order:
        for target in ledger[bid]["targets"]:
            by_target[target].append(bid)
    rows = outline()
    index = {}
    for target, ids in sorted(by_target.items()):
        meta = rows.get(target, {"prefix": "—", "scope": ""})
        words = sum(blocks[b]["words"] for b in ids)
        lines = [
            f"# Source bundle for `{target}`",
            "",
            f"Requirement prefix: {meta['prefix']}. Scope: {meta['scope']}",
            f"{len(ids)} blocks, {words} words.",
            "",
        ]
        for bid in ids:
            block, entry = blocks[bid], ledger[bid]
            others = [t for t in entry["targets"] if t != target]
            info = [f"kind {entry['kind']}"]
            if entry.get("tags"):
                info.append("tags " + ", ".join(entry["tags"]))
            if others:
                primary = entry["targets"][0]
                if primary == target:
                    info.append("shared with " + ", ".join(others) + "; PRIMARY OWNER is this document: state the shared rule here")
                else:
                    info.append(
                        f"shared with {', '.join(others)}; primary owner is {primary}: link to it and restate only"
                        " a part that belongs to this document alone"
                    )
            if entry.get("note"):
                info.append("note: " + entry["note"])
            lines += [
                f"## {bid} · {block['doc']} › {' › '.join(block['path'])}",
                f"<!-- {'; '.join(info)} -->",
                "",
                block["text"],
                "",
            ]
        name = target.replace("/", "__")
        (out / name).write_text("\n".join(lines), encoding="utf-8")
        index[target] = {"file": str(out / name), "blocks": ids, "words": words, **meta}
    (out / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    for target, item in sorted(index.items(), key=lambda kv: -kv[1]["words"]):
        print(f"{target:52} {len(item['blocks']):5} blocks {item['words']:7} words")
    return 0


if __name__ == "__main__":
    sys.exit(main())
