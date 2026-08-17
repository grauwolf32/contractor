"""Small dependency-free CLI for the research registry."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from research.models import Hypothesis
from research.registry import DEFAULT_ROOT, RegistryError, load_registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="contractor-research")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="research registry root")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="validate all records and cross-references")

    list_parser = subparsers.add_parser("list", help="list hypotheses")
    list_parser.add_argument("--status")
    list_parser.add_argument("--direction")

    show_parser = subparsers.add_parser("show", help="show one hypothesis and linked records")
    show_parser.add_argument("hypothesis_id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        registry = load_registry(args.root)
    except RegistryError as exc:
        print(f"invalid registry: {exc}")
        return 2

    if args.command == "validate":
        print(
            f"valid: {len(registry.hypotheses)} hypotheses, "
            f"{len(registry.experiments)} experiments, {len(registry.decisions)} decisions"
        )
        return 0

    if args.command == "list":
        hypotheses: Iterable[Hypothesis] = registry.hypotheses.values()
        if args.status:
            hypotheses = (item for item in hypotheses if item.status == args.status)
        if args.direction:
            hypotheses = (item for item in hypotheses if item.direction == args.direction.upper())
        for item in hypotheses:
            print(f"{item.id:6} {item.status:12} {item.title}")
        return 0

    hypothesis = registry.hypotheses.get(args.hypothesis_id.upper())
    if hypothesis is None:
        print(f"unknown hypothesis: {args.hypothesis_id}")
        return 1
    experiments, decisions = registry.records_for(hypothesis.id)
    payload = hypothesis.model_dump(mode="json")
    payload["experiments"] = [item.model_dump(mode="json") for item in experiments]
    payload["decisions"] = [item.model_dump(mode="json") for item in decisions]
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
