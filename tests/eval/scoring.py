from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class Score:
    precision: float
    recall: float
    f1: float
    matched: frozenset
    missing: frozenset
    extra: frozenset

    def passes(self, *, min_precision: float, min_recall: float) -> bool:
        return self.precision >= min_precision and self.recall >= min_recall

    def explain(self, label: str) -> str:
        lines = [
            f"{label}: precision={self.precision:.2f} recall={self.recall:.2f} f1={self.f1:.2f}",
        ]
        if self.missing:
            lines.append(f"  missing ({len(self.missing)}): {sorted(self.missing)}")
        if self.extra:
            lines.append(f"  extra   ({len(self.extra)}): {sorted(self.extra)}")
        return "\n".join(lines)


def _score_sets(actual: Iterable, expected: Iterable) -> Score:
    a = frozenset(actual)
    e = frozenset(expected)
    matched = a & e
    missing = e - a
    extra = a - e
    precision = len(matched) / len(a) if a else 1.0 if not e else 0.0
    recall = len(matched) / len(e) if e else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return Score(
        precision=precision,
        recall=recall,
        f1=f1,
        matched=matched,
        missing=missing,
        extra=extra,
    )


def _norm_path(path: str) -> str:
    return "/" + path.strip().strip("/").lower()


def _norm_method(method: str) -> str:
    return method.strip().lower()


def oas_endpoint_set(schema: dict[str, Any]) -> set[tuple[str, str]]:
    """Extract (method, path) pairs from an OpenAPI schema."""
    paths = (schema or {}).get("paths") or {}
    methods = {"get", "post", "put", "patch", "delete", "head", "options", "trace"}
    out: set[tuple[str, str]] = set()
    for path, item in paths.items():
        if not isinstance(item, dict):
            continue
        for method in item.keys():
            m = _norm_method(method)
            if m in methods:
                out.add((m, _norm_path(path)))
    return out


def score_endpoints(
    actual_schema: dict[str, Any], expected_schema: dict[str, Any]
) -> Score:
    return _score_sets(
        oas_endpoint_set(actual_schema), oas_endpoint_set(expected_schema)
    )


def oas_component_set(
    schema: dict[str, Any], kind: str = "schemas"
) -> set[str]:
    components = (schema or {}).get("components") or {}
    bucket = components.get(kind) or {}
    return set(bucket.keys())


def score_components(
    actual_schema: dict[str, Any],
    expected_schema: dict[str, Any],
    kind: str = "schemas",
) -> Score:
    return _score_sets(
        oas_component_set(actual_schema, kind),
        oas_component_set(expected_schema, kind),
    )


def vulnerability_key(v: dict[str, Any]) -> tuple[str, str, str]:
    """Normalize a vulnerability record to a comparable key."""
    return (
        _norm_method(str(v.get("method", ""))),
        _norm_path(str(v.get("path", ""))),
        str(v.get("tag", "")).strip().lower(),
    )


def score_vulnerabilities(
    actual: list[dict[str, Any]], expected: list[dict[str, Any]]
) -> Score:
    return _score_sets(
        {vulnerability_key(v) for v in actual},
        {vulnerability_key(v) for v in expected},
    )


def has_phrase(text: str, phrase: str, case_sensitive: bool = False) -> bool:
    if case_sensitive:
        return phrase in text
    return phrase.lower() in text.lower()


def score_phrases(
    text: str, expected_phrases: list[str], case_sensitive: bool = False
) -> Score:
    found = {p for p in expected_phrases if has_phrase(text, p, case_sensitive)}
    return _score_sets(found, set(expected_phrases))


_MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)


def markdown_headings(text: str) -> set[str]:
    """Return the lowercased text of every ATX-style markdown heading."""
    return {m.group(1).strip().lower() for m in _MARKDOWN_HEADING_RE.finditer(text)}


def score_markdown_sections(text: str, expected_sections: list[str]) -> Score:
    """Score how many expected headings the document contains.

    Matching is case-insensitive substring — an expected ``"Configuration"``
    matches any heading containing the word. Use this for structural
    checks on Markdown task outputs (e.g. ``project_information``'s
    nine numbered categories).
    """
    headings = markdown_headings(text)
    expected_norm = [s.strip().lower() for s in expected_sections]
    found = {
        exp for exp in expected_norm if any(exp in h for h in headings)
    }
    return _score_sets(found, set(expected_norm))


# ---------------------------------------------------------------------------
# Vulnerability detection scoring
# ---------------------------------------------------------------------------

LINE_TOLERANCE = 10


def _normalise_vuln_path(path: str) -> str:
    """Normalise a file path for vuln matching (strip leading ./ and /)."""
    path = path.replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path.lstrip("/")


@dataclass(frozen=True)
class VulnMatchResult:
    classification: str  # "TP" | "FP" | "FN" | "TN"
    ground_truth_id: Optional[str]
    finding_file: Optional[str] = None
    finding_cwe: Optional[str] = None


@dataclass(frozen=True)
class VulnScore:
    tp: int
    fp: int
    fn: int
    tn: int
    matches: list[VulnMatchResult]

    @property
    def precision(self) -> float:
        total = self.tp + self.fp
        return self.tp / total if total else (1.0 if self.fn == 0 else 0.0)

    @property
    def recall(self) -> float:
        total = self.tp + self.fn
        return self.tp / total if total else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def f2(self) -> float:
        p, r = self.precision, self.recall
        return 5 * p * r / (4 * p + r) if (4 * p + r) > 0 else 0.0

    def passes(self, *, min_precision: float = 0.0, min_recall: float = 0.0) -> bool:
        return self.precision >= min_precision and self.recall >= min_recall

    def explain(self) -> str:
        lines = [
            f"vuln_score: TP={self.tp} FP={self.fp} FN={self.fn} TN={self.tn}",
            f"  precision={self.precision:.2f} recall={self.recall:.2f} "
            f"f1={self.f1:.2f} f2={self.f2:.2f}",
        ]
        for m in self.matches:
            if m.classification == "FN":
                lines.append(f"  FN: gt={m.ground_truth_id}")
            elif m.classification == "FP":
                lines.append(
                    f"  FP: file={m.finding_file} cwe={m.finding_cwe}"
                )
            elif m.classification == "TP":
                lines.append(
                    f"  TP: gt={m.ground_truth_id} file={m.finding_file}"
                )
        return "\n".join(lines)


@dataclass(frozen=True)
class AgentFinding:
    """A vulnerability finding reported by the agent."""
    file: str
    cwe: Optional[str] = None
    line: Optional[int] = None
    title: Optional[str] = None
    severity: Optional[str] = None


def _line_in_range(
    finding_line: Optional[int],
    gt_start: Optional[int],
    gt_end: Optional[int] = None,
    tolerance: int = LINE_TOLERANCE,
) -> bool:
    if finding_line is None or gt_start is None:
        return True
    low = gt_start - tolerance
    high = (gt_end if gt_end is not None else gt_start) + tolerance
    return low <= finding_line <= high


def _finding_matches_gt(finding: AgentFinding, gt: dict[str, Any]) -> bool:
    """Check if an agent finding matches a ground-truth entry."""
    norm_finding = _normalise_vuln_path(finding.file)
    norm_gt = _normalise_vuln_path(gt["file"])

    if norm_finding != norm_gt:
        return False

    if finding.cwe and finding.cwe not in gt.get("acceptable_cwes", [gt["primary_cwe"]]):
        return False

    if not _line_in_range(finding.line, gt.get("start_line"), gt.get("end_line")):
        # Check acceptable_locations
        for alt_loc in gt.get("acceptable_locations", []):
            alt_file = _normalise_vuln_path(alt_loc.get("file", ""))
            if alt_file == norm_finding and _line_in_range(
                finding.line, alt_loc.get("start_line"), alt_loc.get("end_line")
            ):
                return True
        return False

    return True


def score_vuln_findings(
    findings: list[AgentFinding],
    ground_truth: list[dict[str, Any]],
) -> VulnScore:
    """Score agent vulnerability findings against ground truth.

    Matching uses file path + CWE (if reported) + line proximity (±10).
    Ground-truth entries with ``is_vulnerable: false`` are FP traps.
    """
    results: list[VulnMatchResult] = []
    matched_gt_ids: set[str] = set()

    for finding in findings:
        candidates: list[dict[str, Any]] = []
        for gt in ground_truth:
            if gt["id"] in matched_gt_ids:
                continue
            if _finding_matches_gt(finding, gt):
                candidates.append(gt)

        if candidates:
            candidates.sort(key=lambda g: (not g.get("is_vulnerable", True),))
            best = candidates[0]
            classification = "TP" if best.get("is_vulnerable", True) else "FP"
            results.append(VulnMatchResult(
                classification=classification,
                ground_truth_id=best["id"],
                finding_file=finding.file,
                finding_cwe=finding.cwe,
            ))
            matched_gt_ids.add(best["id"])
        else:
            results.append(VulnMatchResult(
                classification="FP",
                ground_truth_id=None,
                finding_file=finding.file,
                finding_cwe=finding.cwe,
            ))

    for gt in ground_truth:
        if gt["id"] not in matched_gt_ids:
            classification = "FN" if gt.get("is_vulnerable", True) else "TN"
            results.append(VulnMatchResult(
                classification=classification,
                ground_truth_id=gt["id"],
            ))

    tp = sum(1 for r in results if r.classification == "TP")
    fp = sum(1 for r in results if r.classification == "FP")
    fn = sum(1 for r in results if r.classification == "FN")
    tn = sum(1 for r in results if r.classification == "TN")

    return VulnScore(tp=tp, fp=fp, fn=fn, tn=tn, matches=results)
