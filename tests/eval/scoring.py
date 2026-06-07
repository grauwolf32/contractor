from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Composite eval result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalCheck:
    """A single named check within an eval result."""

    name: str
    passed: bool
    details: str


@dataclass
class EvalResult:
    """Composite eval result — one or more named checks.

    Tests assert ``result.passed``; scripts can also inspect individual
    checks and the ``meta`` dict for underlying Score objects / run info.
    """

    checks: list[EvalCheck]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def explain(self) -> str:
        lines: list[str] = []
        for c in self.checks:
            status = "PASS" if c.passed else "FAIL"
            lines.append(f"[{status}] {c.name}")
            for detail_line in c.details.splitlines():
                lines.append(f"  {detail_line}")
        for key, value in self.meta.items():
            lines.append(f"{key}={value}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Primitive score
# ---------------------------------------------------------------------------


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
        for method in item:
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
    ground_truth_id: str | None
    finding_file: str | None = None
    finding_cwe: str | None = None


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
    cwe: str | None = None
    line: int | None = None
    title: str | None = None
    severity: str | None = None


def _line_in_range(
    finding_line: int | None,
    gt_start: int | None,
    gt_end: int | None = None,
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


def partition_findings_by_read(
    findings: list[AgentFinding],
    read_paths: Iterable[str],
) -> tuple[list[AgentFinding], list[AgentFinding]]:
    """Split findings into (grounded, ungrounded) by emitted-vs-read cross-check.

    A finding is *grounded* when the file it points at (``finding.file``) was
    actually opened/read by the worker — i.e. it appears in ``read_paths``.
    A finding whose file was NEVER read is *ungrounded*: a likely hallucination
    (e.g. a CRUD endpoint or file absent from the source). This is a purely
    deterministic, side-effect-free filter — it never inspects content.

    Path comparison uses :func:`_normalise_vuln_path` on both sides (strip
    leading ``./`` and ``/``, normalise slashes) so the finding's ``place`` and
    the worker's read paths match regardless of leading-slash conventions.

    Findings whose ``file`` is empty or whose location is URL-shaped (contains
    ``://``) are passed through as **grounded** — only file-type places are
    checkable against the read set (URL-type places come from live HTTP probing,
    not source reads, so this filter has nothing to say about them).

    Edge case — empty ``read_paths``: every file-type finding is ungrounded.
    This is intentional and faithful: if the read set is genuinely empty there
    is no evidence the worker read anything, so no file finding can be grounded.
    Callers that cannot reliably derive a read set should keep the gate OFF
    rather than pass an empty set and silently drop every finding.
    """
    read_norm = {_normalise_vuln_path(p) for p in read_paths if p}

    grounded: list[AgentFinding] = []
    ungrounded: list[AgentFinding] = []
    for finding in findings:
        place = finding.file or ""
        # URL-shaped or empty places are not file-checkable → pass through.
        if not place or "://" in place:
            grounded.append(finding)
            continue
        if _normalise_vuln_path(place) in read_norm:
            grounded.append(finding)
        else:
            ungrounded.append(finding)
    return grounded, ungrounded


_SEVERITY_RANK = {
    "critical": 5,
    "high": 4,
    "medium": 3,
    "moderate": 3,
    "low": 2,
    "info": 1,
    "informational": 1,
}

# Stop-words stripped from titles before token comparison so that boilerplate
# phrasing ("possible SQL injection vulnerability in handler") doesn't dominate
# the Jaccard score over the substantive tokens.
_TITLE_STOPWORDS = frozenset({
    "a", "an", "the", "in", "on", "of", "to", "for", "and", "or", "at",
    "is", "are", "via", "with", "by", "from",
    "vulnerability", "vuln", "issue", "finding", "possible", "potential",
})

_TITLE_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Default Jaccard threshold for collapsing near-identical titles within a
# (file, cwe) group. >= this fraction of shared tokens => same underlying issue.
TITLE_JACCARD_THRESHOLD = 0.6


def _severity_score(severity: str | None) -> int:
    """Map a severity label to an orderable rank (higher = more severe)."""
    if not severity:
        return 0
    return _SEVERITY_RANK.get(severity.strip().lower(), 0)


def _norm_cwe(cwe: str | None) -> str:
    """Normalise a CWE id for grouping (case-folded, whitespace-stripped)."""
    return (cwe or "").strip().upper()


def _title_tokens(title: str | None) -> frozenset[str]:
    """Tokenise a title into a normalised stop-word-filtered token set."""
    if not title:
        return frozenset()
    toks = _TITLE_TOKEN_RE.findall(title.lower())
    return frozenset(t for t in toks if t not in _TITLE_STOPWORDS)


def _titles_near_identical(
    a: str | None, b: str | None, threshold: float = TITLE_JACCARD_THRESHOLD
) -> bool:
    """Whether two titles are near-identical by normalised-token Jaccard.

    Two empty/blank titles are treated as identical (both describe the same
    file+cwe with no distinguishing text). A blank vs non-blank title is NOT
    near-identical — the non-blank one carries distinguishing information.
    """
    ta, tb = _title_tokens(a), _title_tokens(b)
    if not ta and not tb:
        return True
    if not ta or not tb:
        return False
    inter = len(ta & tb)
    union = len(ta | tb)
    return union > 0 and (inter / union) >= threshold


def _more_severe(candidate: AgentFinding, current: AgentFinding) -> bool:
    """Whether ``candidate`` should replace ``current`` as a group's representative.

    Ranks by severity, then by whether a line number / cwe is present (more
    specific findings win), so the kept representative is the most informative.
    """
    cand_key = (
        _severity_score(candidate.severity),
        candidate.line is not None,
        bool(candidate.cwe),
    )
    cur_key = (
        _severity_score(current.severity),
        current.line is not None,
        bool(current.cwe),
    )
    return cand_key > cur_key


def dedupe_findings(
    findings: list[AgentFinding],
    *,
    threshold: float = TITLE_JACCARD_THRESHOLD,
) -> list[AgentFinding]:
    """Collapse near-duplicate findings, keeping the strongest representative.

    Pure, deterministic, side-effect-free. Duplicates are findings that point
    at the *same underlying issue* reported more than once. The merge is
    deliberately conservative — it only ever collapses findings that share **all**
    of:

    * the same normalised file path (:func:`_normalise_vuln_path`), and
    * the same primary CWE (case-folded; empty CWE only merges with empty CWE), and
    * near-identical titles (normalised-token Jaccard ``>= threshold``).

    Within such a cluster the single kept representative is the most severe /
    most specific finding (see :func:`_more_severe`): higher severity wins,
    ties broken toward a present line number then a present CWE.

    Anything that differs in file, CWE, or has a clearly different title is a
    *distinct* finding and is KEPT. Findings are never merged across files.
    Findings are processed most-severe-first, so the kept representative is the
    most severe member and the result is deterministic and idempotent
    (independent of input order).
    """
    # Bucket by (normalised file, normalised cwe) first — these never merge
    # across each other. Within a bucket, greedily cluster by title similarity.
    clusters: list[tuple[tuple[str, str], list[AgentFinding]]] = []
    index_by_bucket: dict[tuple[str, str], list[int]] = {}

    # Process most-severe-first (stable) so each cluster's anchor IS the kept
    # representative — makes dedup order-independent and idempotent.
    ordered = sorted(
        findings,
        key=lambda f: (_severity_score(f.severity), f.line is not None, bool(f.cwe)),
        reverse=True,
    )
    for finding in ordered:
        bucket = (_normalise_vuln_path(finding.file), _norm_cwe(finding.cwe))
        placed = False
        for cluster_idx in index_by_bucket.get(bucket, []):
            rep = clusters[cluster_idx][1][0]
            if _titles_near_identical(rep.title, finding.title, threshold):
                clusters[cluster_idx][1].append(finding)
                placed = True
                break
        if not placed:
            clusters.append((bucket, [finding]))
            index_by_bucket.setdefault(bucket, []).append(len(clusters) - 1)

    out: list[AgentFinding] = []
    for _bucket, members in clusters:
        rep = members[0]
        for other in members[1:]:
            if _more_severe(other, rep):
                rep = other
        out.append(rep)
    return out


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
