#!/usr/bin/env python3
"""Prepare native Eval authoring files from frozen inputs; never dispatch a model."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUBRIC = (
    "Review each free-form observation against its attached original memory. "
    "Measure unsupported assertions, duplicate/spam observations and useful-fact recall. "
    "Preserve uncertainty and the scope of negative probes; source attachment is not verification. "
    "Report model calls, observed tokens, latency and known cost separately from semantic quality."
)


def load_corpus():
    lock = json.loads((ROOT / "corpus-v4.lock.json").read_text())
    for name, digest in lock["files"].items():
        if "sha256:" + hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Frozen corpus drift: " + name)
    return json.loads((ROOT / "corpus.json").read_text())


def dataset(project, revisions, case_ids):
    corpus = load_corpus()
    selected = [case for case in corpus["cases"] if case["id"] in case_ids]
    if len(selected) != len(case_ids) or set(revisions) != set(case_ids):
        raise ValueError(
            "Supply one exact uploaded revision for each selected corpus case"
        )
    cases = []
    for case in selected:
        data = (ROOT / case["snapshot"]).read_bytes()
        if not isinstance(revisions[case["id"]], str) or not revisions[case["id"]]:
            raise ValueError("Uploaded revisions must be nonempty strings")
        cases.append(
            {
                "id": case["id"],
                "task": {
                    "kind": "memory-extraction",
                    "objective": "Select useful reported observations from exact memories",
                    "parameters": {},
                },
                "inputs": {
                    "memories": {
                        "scope": "project",
                        "scopeId": project,
                        "namespace": "fact-eval",
                        "name": case["id"],
                        "revision": revisions[case["id"]],
                        "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
                        "mediaType": "application/json",
                        "sizeBytes": len(data),
                    }
                },
                "requires": [],
                "outputs": {},
            }
        )
    return {
        "datasetId": "project-facts-v1",
        "name": "Reported memory observations v1",
        "cases": cases,
        "privateChecks": [
            {
                "id": "memory-quality",
                "revision": "v1",
                "rubric": RUBRIC,
                "expected": {
                    "cases": json.dumps(selected, ensure_ascii=False, sort_keys=True)
                },
            }
        ],
    }


def experiment(revision, case_ids, repetitions):
    load_corpus()
    result = json.loads((ROOT / "experiment-template-v4.json").read_text())
    result["draft"]["dataset"]["revision"] = revision
    result["draft"]["caseIds"] = case_ids
    result["draft"]["repetitions"] = repetitions
    result["draft"]["budgets"]["maxMembers"] = len(case_ids) * repetitions * 2
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["dataset", "experiment", "verify"])
    parser.add_argument("--project")
    parser.add_argument("--revisions", type=Path)
    parser.add_argument("--dataset-revision")
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    corpus = load_corpus()
    ids = args.cases or [case["id"] for case in corpus["cases"]]
    if len(ids) != len(set(ids)) or not set(ids) <= {
        case["id"] for case in corpus["cases"]
    }:
        parser.error("Select distinct known corpus cases")
    if not 1 <= args.repetitions <= 100:
        parser.error("repetitions must be from 1 to 100")
    if args.kind == "verify":
        print(f"Frozen extraction corpus verified: {len(corpus['cases'])} cases")
        return
    if args.output is None:
        parser.error(
            "--output is required; authoring files may contain private ground truth"
        )
    if args.kind == "dataset":
        if not args.project or args.revisions is None:
            parser.error("dataset requires --project and --revisions")
        result = dataset(args.project, json.loads(args.revisions.read_text()), ids)
    else:
        if not args.dataset_revision:
            parser.error("experiment requires --dataset-revision")
        result = experiment(args.dataset_revision, ids, args.repetitions)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
