"""Unit tests for ``dedupe_findings`` — the QW7/K deterministic dedup/merge
pass that collapses near-duplicate vuln findings before scoring.

The function is pure and deterministic; these tests pin its contract:
  * same file + same CWE + near-identical title -> merged to 1 (keep stronger)
  * same file, different CWE                     -> both kept
  * different files                              -> both kept (never merge across files)
  * empty input / idempotency                    -> stable, no surprises
  * clearly different titles in same file+cwe    -> kept distinct
"""

from __future__ import annotations

from tests.eval.scoring import AgentFinding, dedupe_findings


def _f(
    file: str,
    cwe: str | None = "CWE-89",
    title: str | None = "t",
    severity: str | None = "low",
    line: int | None = None,
) -> AgentFinding:
    return AgentFinding(file=file, cwe=cwe, title=title, severity=severity, line=line)


def test_same_file_cwe_near_identical_title_merged_keeps_higher_severity():
    # Two reports of the same issue; titles differ only by boilerplate phrasing.
    low = _f(
        "app/views.py",
        cwe="CWE-89",
        title="SQL injection in user login query",
        severity="low",
    )
    high = _f(
        "app/views.py",
        cwe="CWE-89",
        title="Possible SQL injection vulnerability in user login query",
        severity="critical",
    )
    out = dedupe_findings([low, high])
    assert len(out) == 1
    assert out[0] is high  # the more-severe representative is kept
    assert out[0].severity == "critical"


def test_same_file_different_cwe_both_kept():
    a = _f("app/views.py", cwe="CWE-89", title="SQL injection in login")
    b = _f("app/views.py", cwe="CWE-79", title="SQL injection in login")
    out = dedupe_findings([a, b])
    assert len(out) == 2
    assert set(out) == {a, b}


def test_different_files_both_kept():
    # Identical CWE + title, but different files => distinct issues, never merged.
    a = _f("app/views.py", cwe="CWE-89", title="SQL injection in query")
    b = _f("app/models.py", cwe="CWE-89", title="SQL injection in query")
    out = dedupe_findings([a, b])
    assert len(out) == 2
    assert set(out) == {a, b}


def test_same_file_cwe_clearly_different_titles_kept():
    a = _f("app/views.py", cwe="CWE-89", title="SQL injection in login query")
    b = _f("app/views.py", cwe="CWE-89", title="Hardcoded admin password constant")
    out = dedupe_findings([a, b])
    assert len(out) == 2
    assert set(out) == {a, b}


def test_empty_input_returns_empty():
    assert dedupe_findings([]) == []


def test_idempotent():
    findings = [
        _f("app/views.py", cwe="CWE-89", title="SQL injection in login", severity="high"),
        _f("app/views.py", cwe="CWE-89", title="SQL injection at login", severity="low"),
        _f("app/models.py", cwe="CWE-79", title="Reflected XSS in name field"),
    ]
    once = dedupe_findings(findings)
    twice = dedupe_findings(once)
    assert once == twice
    assert len(once) == 2  # the two app/views.py CWE-89 reports collapse to one


def test_path_normalisation_merges_across_slash_conventions():
    a = _f("/app/views.py", title="SQL injection in login")
    b = _f("./app/views.py", title="SQL injection in login")
    out = dedupe_findings([a, b])
    assert len(out) == 1


def test_empty_titles_merge_within_same_file_cwe():
    a = _f("app/views.py", cwe="CWE-89", title=None, severity="low")
    b = _f("app/views.py", cwe="CWE-89", title="", severity="high")
    out = dedupe_findings([a, b])
    assert len(out) == 1
    assert out[0].severity == "high"
