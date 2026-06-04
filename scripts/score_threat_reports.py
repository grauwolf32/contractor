#!/usr/bin/env python3
"""Offline re-score of persisted threat_analysis reports for an A/B compare.

Usage: score_threat_reports.py <label>:<case_dir> <fixture_slug> [...]
  <case_dir> is a dir containing
  .../vulnerability-reports/threat-model/versions/<n>/<report-file>
Re-scores with the CURRENT scorer so arms run at different times are comparable.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import yaml

from tests.eval.scorers import score_threat_analysis

ROOT = "/home/ruslan/src/contractor"


def latest_reports(case_dir: str) -> list[dict]:
    base = None
    for d in glob.glob(f"{case_dir}/**/vulnerability-reports/threat-model/versions",
                       recursive=True):
        base = d
    if not base:
        return []
    versions = sorted(int(v) for v in os.listdir(base) if v.isdigit())
    if not versions:
        return []
    files = [p for p in glob.glob(f"{base}/{versions[-1]}/*") if not p.endswith(".json")]
    if not files:
        return []
    with open(files[0]) as fh:
        raw = yaml.safe_load(fh) or {}
    return [{"name": n, **r} for n, r in raw.items() if isinstance(r, dict)]


def expected_vulns(slug: str) -> list[dict]:
    p = f"{ROOT}/tests/eval/fixtures/{slug}/vulnerabilities.expected.json"
    return json.load(open(p)) if os.path.isfile(p) else []


def main() -> int:
    case = {"min_threats": 3, "min_stride_categories": 3,
            "min_shape_recall": 0.6, "min_endpoint_recall": 0.2}
    for arg in sys.argv[1:]:
        label, _, rest = arg.partition(":")
        case_dir, _, slug = rest.partition("|")
        slug = slug or label
        reports = latest_reports(case_dir if os.path.isabs(case_dir) else f"{ROOT}/{case_dir}")
        if not reports:
            print(f"[{label}] no reports found under {case_dir}")
            continue
        res = score_threat_analysis(reports, case, expected_vulns(slug))
        ep = res.meta.get("endpoint_score")
        print(f"\n[{label}]  reports={res.meta['report_count']}  "
              f"stride={res.meta['stride_letters']}  "
              f"shape={res.meta['shape_recall']}  "
              f"endpoint_recall={round(ep.recall, 3) if ep else 'n/a'}  "
              f"passed={res.passed}")
        for c in res.checks:
            print(f"    {'OK ' if c.passed else 'XX '}{c.name}: {c.details}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
