"""Assemble platform registries, the spec history and open questions.

Usage: python3 docs/spec/_migration/assemble.py BLOCKS_DIR RESULTS_JSON

- Registry rows returned by the drafters become the tables of
  docs/spec/platform/{limits,errors,failure-model,lifecycle}.md, inserted at the
  "<!-- registry-table -->" marker.
- Blocks of kind history go verbatim to docs/decisions/spec-history.md and
  open questions plus reported conflicts to docs/decisions/open-questions.md,
  grouped by legacy document.
- Glossary entries are written to RESULTS_JSON's directory as glossary.json for
  the glossary step.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
SPEC = REPO / "docs" / "spec"
MARKER = "<!-- registry-table -->"
END = "<!-- /registry-table -->"


def cell(value: object) -> str:
    return " ".join(str(value or "").split()).replace("|", "\\|") or "—"


def owner(target: str, requirement: str) -> str:
    link = f"../{target}" if not target.startswith("platform/") else target.split("/", 1)[1]
    label = requirement if re.fullmatch(r"[A-Z]{2,4}-R\d+", requirement or "") else target.split("/")[-1].removesuffix(".md")
    return f"[{label}]({link})"


def write_registry(name: str, title: str, intro: str, header: list[str], rows: list[list[str]]) -> None:
    path = SPEC / "platform" / name
    table = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    table += ["| " + " | ".join(row) + " |" for row in rows]
    block = MARKER + "\n\n" + "\n".join(table) + "\n\n" + END
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        if MARKER in text and END in text:
            text = text[: text.index(MARKER)] + block + text[text.index(END) + len(END) :]
        elif MARKER in text:
            text = text.replace(MARKER, block)
        else:
            text = text.rstrip() + "\n\n## Registry\n\n" + block + "\n"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = f"# {title}\n\n[Specification index](../README.md)\n\n{intro}\n\n## Registry\n\n{block}"
    path.write_text(text, encoding="utf-8")


def main() -> int:
    blocks_dir, results_path = Path(sys.argv[1]), Path(sys.argv[2])
    ledger = json.loads((HERE / "ledger.json").read_text())["entries"]
    blocks = {b["id"]: b for p in blocks_dir.glob("*.json") for b in json.loads(p.read_text())}
    results = [r for r in json.loads(results_path.read_text()) if r.get("draft")]

    limits, failures, lifecycle = [], [], []
    errors: dict[str, list[str]] = {}
    glossary, conflicts = [], []
    for result in sorted(results, key=lambda r: r["target"]):
        target, draft = result["target"], result["draft"]
        for e in draft.get("limits", []):
            limits.append([f"`{e['id']}`", cell(e["value"]), cell(e["applies_to"]), cell(e["enforced_by"]), owner(target, e["requirement"])])
        for e in draft.get("failures", []):
            failures.append([f"`{e['id']}`", cell(e["operation"]), cell(e["outcomes"]), owner(target, e["requirement"])])
        for e in draft.get("lifecycle", []):
            lifecycle.append([f"`{e['id']}`", cell(e["entity"]), cell(e["deleted_by"]), cell(e["blocked_by"]), cell(e["cascades"]), cell(e["retained"]), owner(target, e["requirement"])])
        for e in draft.get("errors", []):
            code = e["code"].strip("` ")
            row = errors.get(code)
            if row is None:
                errors[code] = [f"`{code}`", cell(e["http_status"]), cell(e["retryable"]), cell(e["meaning"]), owner(target, e["requirement"])]
            else:
                row[4] += ", " + owner(target, e["requirement"])
        glossary += [{**g, "owner": target} for g in draft.get("glossary", [])]
        conflicts += [{**c, "target": target} for c in draft.get("conflicts", []) if not c["description"].startswith("verifier issue rejected")]

    key = lambda row: [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", row[0])]
    write_registry("limits.md", "Limits", "Every numeric bound and cardinality target, defined once.", ["ID", "Value", "Applies to", "Enforced by", "Owner"], sorted(limits, key=key))
    write_registry("errors.md", "Error and stop codes", "Every error and stop code, defined once.", ["Code", "HTTP status", "Retryable", "Meaning", "Owner"], sorted(errors.values(), key=key))
    write_registry("failure-model.md", "Failure model", "How each external call and mutation behaves on every outcome.", ["ID", "Operation", "Outcomes", "Owner"], sorted(failures, key=key))
    write_registry("lifecycle.md", "Entity lifecycle", "Who deletes each entity, what blocks it, what cascades and what is retained.", ["ID", "Entity", "Deleted by", "Blocked by", "Cascades", "Retained", "Owner"], sorted(lifecycle, key=key))

    by_doc: dict[str, list[dict]] = defaultdict(list)
    questions: dict[str, list[dict]] = defaultdict(list)
    for entry in ledger:
        block = blocks[entry["id"]]
        if entry["kind"] == "history":
            by_doc[block["doc"]].append(block)
        elif entry["kind"] == "open-question":
            questions[block["doc"]].append(block)

    def section(doc: str, items: list[dict]) -> list[str]:
        out = [f"## From `{doc}`", ""]
        last = None
        for b in items:
            path = " › ".join(b["path"])
            if path != last:
                out += [f"### {path or doc}", ""]
                last = path
            out += [b["text"], ""]
        return out

    history = [
        "# Specification history",
        "",
        "Delivery plans, increments, acceptance narratives and implementation status",
        "that the feature-ordered specifications carried until the 2026-10 migration",
        "to the domain layout. Kept verbatim for reference; current status lives in",
        "the task files and the backlog, current behaviour in [the specification](../spec/README.md).",
        "",
    ]
    for doc in sorted(by_doc):
        history += section(doc, by_doc[doc])
    (REPO / "docs" / "decisions" / "spec-history.md").write_text("\n".join(history).rstrip() + "\n", encoding="utf-8")

    open_q = [
        "# Open questions",
        "",
        "Unresolved design questions and contradictions between the legacy",
        "specifications found during the 2026-10 migration. Resolve each one in the",
        "owning specification and delete it here.",
        "",
    ]
    for doc in sorted(questions):
        open_q += section(doc, questions[doc])
    if conflicts:
        open_q += ["## Contradictions found during the migration", ""]
        for c in sorted(conflicts, key=lambda c: c["target"]):
            ids = ", ".join(c["blocks"]) or "—"
            link = f"../{c['target']}" if c["target"].startswith("proposals/") else f"../spec/{c['target']}"
            open_q += [f"- **[{c['target']}]({link})** ({ids}): {' '.join(c['description'].split())}"]
    (REPO / "docs" / "decisions" / "open-questions.md").write_text("\n".join(open_q).rstrip() + "\n", encoding="utf-8")

    (results_path.parent / "glossary.json").write_text(json.dumps(glossary, indent=1), encoding="utf-8")
    print(f"limits {len(limits)}, errors {len(errors)}, failures {len(failures)}, lifecycle {len(lifecycle)}, "
          f"glossary {len(glossary)}, conflicts {len(conflicts)}, history docs {len(by_doc)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
