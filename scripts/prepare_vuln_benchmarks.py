#!/usr/bin/env python3
"""Prepare vulnerability benchmark fixtures for the eval harness.

Reads ground-truth data from RealVuln and eyeballvul repos (both expected
under tests/playground/) and generates fixture directories under
tests/eval/fixtures/ that the eval harness can consume.

For RealVuln, the script also clones the vulnerable repos at their pinned
commits into tests/playground/realvuln-repos/.

For eyeballvul, the script downloads the data into the eyeballvul cache,
selects a curated subset of Python repos, clones them, and generates
fixtures.

Usage:
    python scripts/prepare_vuln_benchmarks.py realvuln            # RealVuln only
    python scripts/prepare_vuln_benchmarks.py realvuln --repos realvuln-vampi realvuln-dsvw
    python scripts/prepare_vuln_benchmarks.py eyeballvul          # eyeballvul only
    python scripts/prepare_vuln_benchmarks.py all                 # both
    python scripts/prepare_vuln_benchmarks.py --list-realvuln     # list available repos
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAYGROUND_ROOT = REPO_ROOT / "tests" / "playground"
FIXTURES_ROOT = REPO_ROOT / "tests" / "eval" / "fixtures"
REALVULN_ROOT = PLAYGROUND_ROOT / "Real-Vuln-Benchmark"
REALVULN_GT_DIR = REALVULN_ROOT / "ground-truth"
REALVULN_REPOS_DIR = PLAYGROUND_ROOT / "realvuln-repos"
EYEBALLVUL_ROOT = PLAYGROUND_ROOT / "eyeballvul"
EYEBALLVUL_REPOS_DIR = PLAYGROUND_ROOT / "eyeballvul-repos"

DEFAULT_REALVULN_SLUGS = [
    "realvuln-vampi",
    "realvuln-dsvw",
    "realvuln-vfapi",
    "realvuln-vulnerable-flask-app",
]

CWE_FAMILIES = {
    "injection": ["CWE-78", "CWE-79", "CWE-89", "CWE-90", "CWE-91", "CWE-94",
                   "CWE-95", "CWE-96", "CWE-564", "CWE-943"],
    "auth": ["CWE-284", "CWE-285", "CWE-306", "CWE-307", "CWE-639", "CWE-749",
             "CWE-798", "CWE-862", "CWE-863"],
    "data_exposure": ["CWE-200", "CWE-203", "CWE-204", "CWE-209", "CWE-213",
                       "CWE-256", "CWE-257", "CWE-312", "CWE-916"],
    "crypto": ["CWE-321", "CWE-327", "CWE-328", "CWE-347"],
    "config": ["CWE-215", "CWE-489", "CWE-532"],
}


# ---------------------------------------------------------------------------
# RealVuln
# ---------------------------------------------------------------------------

def load_realvuln_gt(slug: str) -> dict[str, Any]:
    gt_path = REALVULN_GT_DIR / slug / "ground-truth.json"
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")
    with gt_path.open() as f:
        return json.load(f)


def clone_realvuln_repo(slug: str, url: str, sha: str) -> Path:
    repo_path = REALVULN_REPOS_DIR / slug
    if repo_path.is_dir():
        print(f"  [{slug}] already cloned")
        return repo_path

    REALVULN_REPOS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  [{slug}] cloning {url} ...", end=" ", flush=True)

    result = subprocess.run(
        ["git", "clone", "--depth=1", url, str(repo_path)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        print(f"FAILED: {result.stderr.strip()[:200]}")
        raise RuntimeError(f"Clone failed for {slug}")

    if sha:
        subprocess.run(
            ["git", "-C", str(repo_path), "fetch", "--depth=1", "origin", sha],
            capture_output=True, text=True, timeout=60,
        )
        subprocess.run(
            ["git", "-C", str(repo_path), "checkout", sha],
            capture_output=True, text=True, timeout=30,
        )
        print(f"OK (pinned to {sha[:8]})")
    else:
        print("OK (HEAD)")
    return repo_path


def realvuln_gt_to_vuln_cases(gt: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert RealVuln ground-truth findings to vuln-cases.json format."""
    cases: list[dict[str, Any]] = []
    for finding in gt["findings"]:
        loc = finding.get("location", {})
        case: dict[str, Any] = {
            "id": finding["id"],
            "is_vulnerable": finding["is_vulnerable"],
            "vulnerability_class": finding.get("vulnerability_class", "unknown"),
            "primary_cwe": finding["primary_cwe"],
            "acceptable_cwes": finding["acceptable_cwes"],
            "file": finding["file"],
            "start_line": loc.get("start_line"),
            "end_line": loc.get("end_line"),
            "function": loc.get("function"),
            "severity": finding.get("severity", "medium"),
            "description": finding.get("evidence", {}).get("description", ""),
        }
        if finding.get("acceptable_locations"):
            case["acceptable_locations"] = finding["acceptable_locations"]
        cases.append(case)
    return cases


def generate_realvuln_fixture(slug: str) -> None:
    gt = load_realvuln_gt(slug)
    repo_url = gt["repo_url"]
    sha = gt.get("commit_sha", "")

    clone_realvuln_repo(slug, repo_url, sha)

    fixture_dir = FIXTURES_ROOT / slug
    fixture_dir.mkdir(parents=True, exist_ok=True)

    source_root = f"tests/playground/realvuln-repos/{slug}"

    meta = {
        "slug": slug,
        "language": gt.get("language", "python"),
        "framework": gt.get("framework") or "unknown",
        "source_root": source_root,
        "benchmark": "realvuln",
        "repo_url": repo_url,
        "commit_sha": sha,
        "description": (
            f"RealVuln benchmark: {gt.get('repo_id', slug)} — "
            f"{gt.get('language', 'python')}/{gt.get('framework', 'unknown')} app "
            f"with {sum(1 for f in gt['findings'] if f['is_vulnerable'])} known vulnerabilities "
            f"and {sum(1 for f in gt['findings'] if not f['is_vulnerable'])} FP traps."
        ),
    }
    with (fixture_dir / "meta.yaml").open("w") as f:
        yaml.safe_dump(meta, f, sort_keys=False, allow_unicode=True)

    vuln_cases = realvuln_gt_to_vuln_cases(gt)
    with (fixture_dir / "vuln-cases.json").open("w") as f:
        json.dump(vuln_cases, f, indent=2)
        f.write("\n")

    print(f"  [{slug}] fixture written to {fixture_dir}")


# ---------------------------------------------------------------------------
# eyeballvul
# ---------------------------------------------------------------------------

EYEBALLVUL_CURATED: list[dict[str, Any]] = [
    # Small Python repos with known CVEs, good for eval
    {
        "project": "https://github.com/pallets/flask",
        "max_vulns": 5,
        "language_filter": "Python",
    },
    {
        "project": "https://github.com/tornadoweb/tornado",
        "max_vulns": 5,
        "language_filter": "Python",
    },
    {
        "project": "https://github.com/django/django",
        "max_vulns": 5,
        "language_filter": "Python",
    },
]


def generate_eyeballvul_fixtures() -> None:
    """Generate eyeballvul fixtures.

    Requires the eyeballvul Python package and its data to be downloaded.
    Run: pip install -e tests/playground/eyeballvul && python -c 'import eyeballvul; eyeballvul.download_data()'
    """
    try:
        sys.path.insert(0, str(EYEBALLVUL_ROOT))
        from eyeballvul import get_revisions, get_vulns  # type: ignore[import]
    except ImportError:
        print(
            "eyeballvul not importable. Install it first:\n"
            "  pip install -e tests/playground/eyeballvul\n"
            "  python -c 'import eyeballvul; eyeballvul.download_data()'"
        )
        return

    for spec in EYEBALLVUL_CURATED:
        project_url = spec["project"]
        project_name = project_url.rstrip("/").split("/")[-1]
        slug = f"eyeballvul-{project_name}"

        print(f"\n  [{slug}] querying eyeballvul DB for {project_url}")
        vulns = get_vulns(project=project_url, before="2025-01-01")
        if not vulns:
            print(f"  [{slug}] no vulns found, skipping")
            continue

        revisions = get_revisions(project=project_url, before="2025-01-01")
        if not revisions:
            print(f"  [{slug}] no revisions found, skipping")
            continue

        rev = revisions[0]
        commit = rev.commit

        max_vulns = spec.get("max_vulns", 5)
        vulns = vulns[:max_vulns]

        repo_dir = EYEBALLVUL_REPOS_DIR / slug
        if not repo_dir.is_dir():
            EYEBALLVUL_REPOS_DIR.mkdir(parents=True, exist_ok=True)
            print(f"  [{slug}] cloning {project_url} at {commit[:8]}...", flush=True)
            subprocess.run(
                ["git", "clone", "--depth=1", project_url, str(repo_dir)],
                capture_output=True, text=True, timeout=120,
            )
            subprocess.run(
                ["git", "-C", str(repo_dir), "fetch", "--depth=1", "origin", commit],
                capture_output=True, text=True, timeout=60,
            )
            subprocess.run(
                ["git", "-C", str(repo_dir), "checkout", commit],
                capture_output=True, text=True, timeout=30,
            )

        fixture_dir = FIXTURES_ROOT / slug
        fixture_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "slug": slug,
            "language": "python",
            "framework": "unknown",
            "source_root": f"tests/playground/eyeballvul-repos/{slug}",
            "benchmark": "eyeballvul",
            "repo_url": project_url,
            "commit_sha": commit,
            "description": (
                f"eyeballvul benchmark: {project_name} at {commit[:8]} — "
                f"{len(vulns)} known CVEs."
            ),
        }
        with (fixture_dir / "meta.yaml").open("w") as f:
            yaml.safe_dump(meta, f, sort_keys=False, allow_unicode=True)

        vuln_cases: list[dict[str, Any]] = []
        for vuln in vulns:
            case: dict[str, Any] = {
                "id": vuln.id,
                "is_vulnerable": True,
                "primary_cwe": vuln.cwes[0] if vuln.cwes else "CWE-unknown",
                "acceptable_cwes": vuln.cwes or [],
                "description": vuln.details,
                "summary": vuln.summary or "",
                "severity": (
                    vuln.severity[0].get("score", "medium")
                    if vuln.severity
                    else "medium"
                ),
            }
            vuln_cases.append(case)

        with (fixture_dir / "vuln-cases.json").open("w") as f:
            json.dump(vuln_cases, f, indent=2)
            f.write("\n")

        print(f"  [{slug}] fixture written ({len(vuln_cases)} CVEs)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def list_realvuln_repos() -> None:
    if not REALVULN_GT_DIR.is_dir():
        print("RealVuln ground-truth directory not found. Clone it first.")
        return
    for d in sorted(REALVULN_GT_DIR.iterdir()):
        gt_file = d / "ground-truth.json"
        if not d.is_dir() or not gt_file.exists():
            continue
        gt = json.load(gt_file.open())
        vulns = sum(1 for f in gt["findings"] if f["is_vulnerable"])
        fps = sum(1 for f in gt["findings"] if not f["is_vulnerable"])
        loc = gt.get("loc", "?")
        fw = gt.get("framework") or "none"
        print(f"  {d.name:50s}  {vulns:3d} vulns  {fps:2d} FP traps  {fw:10s}  LOC={loc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare vuln benchmark fixtures")
    parser.add_argument(
        "benchmark",
        nargs="?",
        choices=["realvuln", "eyeballvul", "all"],
        help="Which benchmark to prepare",
    )
    parser.add_argument("--repos", nargs="+", help="Specific RealVuln slugs")
    parser.add_argument("--list-realvuln", action="store_true",
                        help="List available RealVuln repos")
    args = parser.parse_args()

    if args.list_realvuln:
        list_realvuln_repos()
        return 0

    if not args.benchmark:
        parser.print_help()
        return 1

    if args.benchmark in ("realvuln", "all"):
        slugs = args.repos or DEFAULT_REALVULN_SLUGS
        print(f"\n=== RealVuln: preparing {len(slugs)} fixtures ===\n")
        for slug in slugs:
            try:
                generate_realvuln_fixture(slug)
            except Exception as e:
                print(f"  [{slug}] ERROR: {e}")

    if args.benchmark in ("eyeballvul", "all"):
        print("\n=== eyeballvul: preparing fixtures ===\n")
        generate_eyeballvul_fixtures()

    return 0


if __name__ == "__main__":
    sys.exit(main())
