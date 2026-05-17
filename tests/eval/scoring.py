from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable


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
