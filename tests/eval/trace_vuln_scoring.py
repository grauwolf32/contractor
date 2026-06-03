"""Score reported vulnerabilities from a ``trace_annotation`` task run.

The trace_annotation task surfaces findings two ways:

1. ``report_vulnerability`` tool calls → ``vulnerability-reports`` artifacts
   (machine-readable, but the worker uses the tool sparingly).
2. Structured ``Shape A/B/C`` blocks in each path's ``trace_annotation/result``
   text (the "Security Summary" section) — where most findings actually land.

This module unions both, normalizes each finding to a **general AppSec family**
(not a benchmark-specific sink — see ``_FAMILY_KEYWORDS``), attributes it to the
operation path it was found under, and matches the result against a fixture's
``vulnerabilities.expected.json`` (keyed by ``vulnerability`` class + ``path``)
to produce detection precision / recall / F1.

The family taxonomy is deliberately coarse and standard so the scorer measures
"did it find a vuln of this class on this endpoint", not "did it phrase it the
benchmark's way". Keep it that way (no per-fixture special cases).
"""

from __future__ import annotations

import glob
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# ── Canonical vuln families ────────────────────────────────────────────────
# Ground-truth ``vulnerability`` class → canonical family.
_CLASS_FAMILY: dict[str, str] = {
    "sqli": "sqli",
    "ssrf": "ssrf",
    "path-traversal": "path-traversal",
    "idor": "idor",
    "broken-access-control": "broken-access-control",
    "csrf": "csrf",
    "credential-exposure": "sensitive-data",
    "pii-exposure": "sensitive-data",
    "pan-exposure": "sensitive-data",
    "pan-storage": "sensitive-data",
    "signature-bypass": "auth-crypto",
    "totp-replay": "auth-crypto",
    "credential-stuffing": "rate-limit-abuse",
    "otp-brute-force": "rate-limit-abuse",
    "totp-brute-force": "rate-limit-abuse",
    "sms-bomb": "rate-limit-abuse",
    "business-logic": "business-logic",
    "race-condition": "business-logic",
    "ssti": "ssti",
    "rce": "rce",
    "xss": "xss",
    "open-redirect": "open-redirect",
    "deserialization": "deserialization",
    "mass-assignment": "mass-assignment",
}

# Reported free-text → canonical family. Each family lists regex fragments
# matched (case-insensitive) against the finding text + control_missing tag.
# Order matters only for first-match wins within a single finding block.
_FAMILY_KEYWORDS: list[tuple[str, list[str]]] = [
    ("sqli", [r"\bsql[\s-]?injection\b", r"\bsqli\b"]),
    ("ssti", [r"\bssti\b", r"server[\s-]?side template", r"template injection"]),
    ("rce", [r"\brce\b", r"remote code execution", r"command injection",
             r"\bshell\b.*inject", r"os command"]),
    ("ssrf", [r"\bssrf\b", r"server[\s-]?side request forgery"]),
    ("path-traversal", [r"path[\s-]?traversal", r"directory traversal",
                        r"arbitrary file (read|write|delete)",
                        r"\bpath_confinement\b", r"\blfi\b"]),
    ("open-redirect", [r"open[\s-]?redirect"]),
    ("xss", [r"\bxss\b", r"cross[\s-]?site scripting"]),
    ("csrf", [r"\bcsrf\b", r"cross[\s-]?site request forgery"]),
    ("idor", [r"\bidor\b", r"\bbola\b", r"object[\s-]?level auth",
              r"\bownership_check\b", r"ownership check", r"insecure direct object"]),
    ("broken-access-control", [r"broken access", r"missing authoriz",
                               r"\bauthz\b", r"function[\s-]?level auth",
                               r"\bbfla\b", r"privilege escalation",
                               r"access control"]),
    ("auth-crypto", [r"constant[\s-]?time", r"signature (bypass|never|not)",
                     r"\breplay\b", r"weak crypto", r"timing attack",
                     r"\bnone\b algorithm", r"forg(e|ed|ery)"]),
    ("rate-limit-abuse", [r"rate[\s-]?limit", r"brute[\s-]?force",
                          r"credential stuffing", r"\bstuffing\b",
                          r"enumeration"]),
    ("deserialization", [r"deserializ", r"insecure deserial", r"pickle",
                         r"yaml\.unsafe"]),
    ("mass-assignment", [r"mass[\s-]?assignment"]),
    ("sensitive-data", [r"hardcoded\b.{0,20}(secret|key|credential|password|token)",
                        r"plaintext (password|credential)", r"cleartext",
                        r"sensitive data exposure", r"\bpii\b",
                        r"\bpan\b", r"card number", r"\bsecret_storage\b",
                        r"\boutput_filter\b", r"information disclosure",
                        r"data exposure", r"over[\s-]?expos", r"unmasked",
                        r"secret.*logged", r"credential.*expos"]),
    ("business-logic", [r"business[\s-]?logic", r"race condition", r"\btoctou\b",
                        r"idempoten", r"atomicit", r"double[\s-]?spend",
                        r"over[\s-]?sell", r"over[\s-]?redeem", r"state[\s-]?machine",
                        r"check[\s-]?then[\s-]?act", r"workflow bypass",
                        r"negative (amount|balance|transfer|quantity|value)"]),
]

_TARGET_RE = re.compile(r"target=(GET|POST|PUT|DELETE|PATCH)_(/[A-Za-z0-9/_{}.\-]*)",
                        re.IGNORECASE)
_METHODPATH_RE = re.compile(
    r"\b(GET|POST|PUT|DELETE|PATCH)\s+(/[A-Za-z0-9/_{}.\-]*)", re.IGNORECASE)
_CWE_RE = re.compile(r"CWE-\d+")


def _norm_path(path: str) -> str:
    """Canonical path: lowercase, ``{param}`` collapsed, no trailing slash."""
    p = re.sub(r"\{[^}]*\}", "{}", path.strip().lower())
    return p.rstrip("/") or "/"


def family_of_class(vuln_class: str) -> str:
    return _CLASS_FAMILY.get((vuln_class or "").strip().lower(), "other")


def family_of_text(text: str) -> str | None:
    """First family whose keywords appear in *text*, else ``None``."""
    low = text.lower()
    for family, pats in _FAMILY_KEYWORDS:
        for pat in pats:
            if re.search(pat, low):
                return family
    return None


@dataclass(frozen=True)
class ReportedVuln:
    family: str
    path: str          # normalized; "" when no path could be attributed
    title: str = ""
    source: str = "result"   # "result" | "report_tool"

    def key(self) -> tuple[str, str]:
        return (self.family, self.path)


def _split_finding_blocks(result_text: str) -> list[str]:
    """Slice a result's Security Summary into per-finding text blocks.

    Findings appear under headers like ``### Finding N:`` or as ``- shape:`` /
    ``**shape**:`` lists; we split on those delimiters and keep each chunk.
    Falls back to the whole "Security Summary" section if no delimiter is found.
    """
    sec = result_text
    m = re.search(r"##\s*Security Summary", result_text, re.IGNORECASE)
    if m:
        sec = result_text[m.start():]
    parts = re.split(r"(?im)^\s*(?:#{2,4}\s*Finding|[-*]\s*\*{0,2}shape\*{0,2}\s*:)",
                     sec)
    blocks = [p for p in parts[1:] if p.strip()]
    return blocks or ([sec] if sec.strip() else [])


def extract_from_result(result_text: str, *, default_path: str | None = None) -> list[ReportedVuln]:
    """Reported vulns from one ``trace_annotation/result`` text.

    Each result is scoped to a single operation; its path is taken from the
    ``@trace target=METHOD_/path`` markers (or *default_path*). A finding's
    own ``METHOD /path`` mention, when present, overrides that.
    """
    op_path = default_path or ""
    tm = _TARGET_RE.search(result_text)
    if tm:
        op_path = tm.group(2)
    op_path = _norm_path(op_path) if op_path else ""

    out: list[ReportedVuln] = []
    seen: set[tuple[str, str]] = set()
    for block in _split_finding_blocks(result_text):
        fam = family_of_text(block)
        if fam is None:
            continue
        bm = _METHODPATH_RE.search(block)
        path = _norm_path(bm.group(2)) if bm else op_path
        title = block.strip().splitlines()[0].strip(":-* ")[:120]
        k = (fam, path)
        if k in seen:
            continue
        seen.add(k)
        out.append(ReportedVuln(family=fam, path=path, title=title, source="result"))
    return out


def _latest_versions(glob_pat: str) -> list[str]:
    files = glob.glob(glob_pat, recursive=True)
    return sorted(files, key=lambda p: int(p.split("/versions/")[1].split("/")[0]))


def extract_from_run_dir(artifacts_dir: Path) -> list[ReportedVuln]:
    """Union of reported vulns across all result versions + report-tool artifacts."""
    artifacts_dir = Path(artifacts_dir)
    by_key: dict[tuple[str, str], ReportedVuln] = {}

    # Each result version ≈ one operation's Security Summary.
    for f in _latest_versions(
        str(artifacts_dir / "**" / "trace_annotation" / "result" / "versions" / "*" / "result")
    ):
        for rv in extract_from_result(Path(f).read_text(encoding="utf-8", errors="ignore")):
            by_key.setdefault(rv.key(), rv)

    # report_vulnerability artifacts (CWE in details → family via text).
    for f in _latest_versions(
        str(artifacts_dir / "**" / "vulnerability-reports" / "**" / "versions" / "*" / "*")
    ):
        if f.endswith("metadata.json"):
            continue
        try:
            reports = yaml.safe_load(Path(f).read_text(encoding="utf-8", errors="ignore")) or {}
        except yaml.YAMLError:
            continue
        if not isinstance(reports, dict):
            continue
        for _name, r in reports.items():
            if not isinstance(r, dict):
                continue
            text = f"{r.get('title', '')} {r.get('details', '')}"
            fam = family_of_text(text)
            if fam is None:
                continue
            rv = ReportedVuln(family=fam, path="", title=str(r.get("title", ""))[:120],
                              source="report_tool")
            by_key.setdefault(rv.key(), rv)
    return list(by_key.values())


def load_expected(path: Path) -> list[dict]:
    """Normalized expected vulns: ``[{family, path, vuln_class, severity}, ...]``."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    out = []
    for c in raw:
        out.append({
            "family": family_of_class(c.get("vulnerability", "")),
            "path": _norm_path(c.get("path", "")),
            "vuln_class": c.get("vulnerability", ""),
            "severity": c.get("severity", ""),
            "method": (c.get("method") or "").upper(),
        })
    return out


@dataclass
class VulnScore:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    matched: list[dict] = field(default_factory=list)
    missed: list[dict] = field(default_factory=list)
    extra: list[dict] = field(default_factory=list)
    per_family: dict[str, dict] = field(default_factory=dict)

    def explain(self) -> str:
        return (f"vulns: TP={self.tp} FP={self.fp} FN={self.fn}  "
                f"P={self.precision:.3f} R={self.recall:.3f} F1={self.f1:.3f}")


def score_vulns(reported: list[ReportedVuln], expected: list[dict]) -> VulnScore:
    """Greedy one-to-one match: same family, path-exact preferred over path-agnostic.

    A reported finding with no attributed path can satisfy a same-family expected
    item (lenient on recall); path-exact matches are taken first so a reported
    finding is not wasted on the wrong endpoint.
    """
    s = VulnScore()
    remaining = list(reported)

    def _take(family: str, path: str, *, exact: bool):
        for i, rv in enumerate(remaining):
            if rv.family != family:
                continue
            if exact and rv.path and rv.path != path:
                continue
            if exact and not rv.path:
                continue           # path-agnostic handled in the lenient pass
            if not exact and rv.path and rv.path != path:
                continue
            return remaining.pop(i)
        return None

    # Pass 1: exact-path matches. Pass 2: path-agnostic / no-path reported.
    for exact in (True, False):
        for exp in expected:
            if exp.get("_done"):
                continue
            rv = _take(exp["family"], exp["path"], exact=exact)
            if rv is not None:
                exp["_done"] = True
                s.matched.append({"family": exp["family"], "path": exp["path"],
                                  "vuln_class": exp["vuln_class"],
                                  "reported_title": rv.title})

    for exp in expected:
        fam = exp["family"]
        pf = s.per_family.setdefault(fam, {"tp": 0, "fn": 0, "fp": 0})
        if exp.get("_done"):
            pf["tp"] += 1
        else:
            pf["fn"] += 1
            s.missed.append({"family": fam, "path": exp["path"],
                             "vuln_class": exp["vuln_class"], "severity": exp["severity"]})
    for rv in remaining:
        s.per_family.setdefault(rv.family, {"tp": 0, "fn": 0, "fp": 0})["fp"] += 1
        s.extra.append({"family": rv.family, "path": rv.path, "title": rv.title})

    s.tp = len(s.matched)
    s.fn = len(s.missed)
    s.fp = len(s.extra)
    s.precision = round(s.tp / (s.tp + s.fp), 3) if (s.tp + s.fp) else 0.0
    s.recall = round(s.tp / (s.tp + s.fn), 3) if (s.tp + s.fn) else 0.0
    s.f1 = round(2 * s.precision * s.recall / (s.precision + s.recall), 3) \
        if (s.precision + s.recall) else 0.0
    # strip the scratch flag
    for exp in expected:
        exp.pop("_done", None)
    return s
