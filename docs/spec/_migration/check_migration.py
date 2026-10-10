"""Check that the migrated documents preserve every legacy block.

Usage: python3 docs/spec/_migration/check_migration.py BLOCKS_DIR RESULTS_JSON [TARGET ...]

- Provenance coverage: every block of a target appears in a <!-- from: --> comment
  of that target, or the drafter listed it in unused_blocks with a reason that
  names another of the block's targets which does cover it.
- Token preservation: inline code spans, numbers with units and snake_case or
  dotted identifiers of each block appear somewhere in the union of the
  block's target documents and the registry rows built from them.

Without TARGET arguments every target with a written document is checked.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
FROM = re.compile(r"<!--\s*from:\s*([^>]*?)\s*-->")
BLOCK_ID = re.compile(r"[a-z0-9]+#\d+")
CODE = re.compile(r"`([^`\n]{2,80})`")
NUMBER_UNIT = re.compile(
    r"\b(\d[\d,.]*\s?(?:KiB|MiB|GiB|TiB|KB|MB|GB|bytes?|ms|s|seconds?|minutes?|hours?|days?|%|tokens?|characters?|chars?|items?|entries|rows?|files?|calls?))\b"
)
IDENT = re.compile(r"\b([a-z][a-z0-9]+(?:_[a-z0-9]+)+|[a-z][a-z0-9-]+@\d+)\b")
SKIP_KINDS = {"history", "navigation", "open-question"}


def out_path(target: str) -> Path:
    if target.startswith(("proposals/", "decisions/")):
        return REPO / "docs" / target
    return REPO / "docs" / "spec" / target


def norm(text: str) -> str:
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text.replace("\\_", "_"))
    return re.sub(r"\s+", " ", text).lower()


def tokens(text: str) -> set[str]:
    found = {m.strip() for m in CODE.findall(text)}
    found |= {m.replace(",", "") for m in NUMBER_UNIT.findall(text)}
    found |= set(IDENT.findall(text))
    return {t for t in found if len(t) > 2}


def main() -> int:
    blocks_dir, results_path = Path(sys.argv[1]), Path(sys.argv[2])
    only = set(sys.argv[3:])
    ledger = {e["id"]: e for e in json.loads((HERE / "ledger.json").read_text())["entries"]}
    blocks = {b["id"]: b for p in blocks_dir.glob("*.json") for b in json.loads(p.read_text())}
    results = json.loads(results_path.read_text())
    drafts = {r["target"]: r["draft"] for r in results if r.get("draft")}
    texts: dict[str, str] = {}
    covered: dict[str, set[str]] = defaultdict(set)
    for target in drafts:
        path = out_path(target)
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            texts[target] = text
            for group in FROM.findall(text):
                covered[target] |= set(BLOCK_ID.findall(group))
    registry_text = {t: json.dumps({k: d.get(k, []) for k in ("limits", "errors", "failures", "lifecycle", "glossary")}) for t, d in drafts.items()}
    unused = {t: {u["id"]: u["reason"] for u in d.get("unused_blocks", [])} for t, d in drafts.items()}
    checked = [t for t in texts if not only or t in only]
    problems = 0
    for target in sorted(checked):
        ids = [bid for bid, e in ledger.items() if target in e["targets"] and e["kind"] not in SKIP_KINDS]
        missing_cov = []
        missing_tok: dict[str, list[str]] = {}
        for bid in ids:
            entry = ledger[bid]
            others = [t for t in entry["targets"] if t != target]
            if bid not in covered[target]:
                reason = unused[target].get(bid, "")
                elsewhere = any(bid in covered.get(o, set()) for o in others)
                if not (reason and (elsewhere or any(o in reason for o in others) or "history" in reason)):
                    missing_cov.append(bid)
            union = norm(" ".join(texts.get(t, "") + registry_text.get(t, "") for t in entry["targets"]))
            lost = sorted(tok for tok in tokens(blocks[bid]["text"]) if norm(tok) not in union)
            if lost:
                missing_tok[bid] = lost
        if missing_cov or missing_tok:
            problems += 1
        print(f"== {target}: {len(ids)} blocks, {len(missing_cov)} uncovered, {len(missing_tok)} with missing tokens")
        for bid in missing_cov:
            print(f"   uncovered {bid}")
        for bid, lost in missing_tok.items():
            print(f"   tokens {bid}: {', '.join(lost[:12])}{' …' if len(lost) > 12 else ''}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
