"""Unit tests for ``partition_findings_by_read`` — the QW1/AC2 emitted-vs-read
cross-check that drops vuln findings whose file was never read by the worker.

The function is pure and deterministic; these tests pin its contract:
  * file in read set        -> grounded
  * file NOT in read set    -> ungrounded (likely hallucination)
  * URL-type / empty place  -> grounded (passthrough; not file-checkable)
  * empty read set          -> every file finding ungrounded (documented edge)
  * path normalisation      -> leading ``/`` / ``./`` differences don't matter
"""

from __future__ import annotations

from tests.eval.scoring import AgentFinding, partition_findings_by_read


def _finding(file: str) -> AgentFinding:
    return AgentFinding(file=file, cwe="CWE-89", line=10, title="t", severity="high")


def test_file_in_read_set_is_grounded():
    findings = [_finding("app/views.py")]
    grounded, ungrounded = partition_findings_by_read(findings, {"app/views.py"})
    assert grounded == findings
    assert ungrounded == []


def test_file_not_in_read_set_is_ungrounded():
    findings = [_finding("app/ghost_crud.py")]
    grounded, ungrounded = partition_findings_by_read(findings, {"app/views.py"})
    assert grounded == []
    assert ungrounded == findings


def test_url_type_place_passes_through_as_grounded():
    # URL-shaped places aren't file-checkable; pass through regardless of read set.
    findings = [AgentFinding(file="https://host/api/users", cwe=None)]
    grounded, ungrounded = partition_findings_by_read(findings, {"app/views.py"})
    assert grounded == findings
    assert ungrounded == []


def test_empty_place_passes_through_as_grounded():
    findings = [AgentFinding(file="", cwe=None)]
    grounded, ungrounded = partition_findings_by_read(findings, {"app/views.py"})
    assert grounded == findings
    assert ungrounded == []


def test_empty_read_set_marks_all_file_findings_ungrounded():
    # Documented edge: no evidence of any read => no file finding can be grounded.
    findings = [_finding("app/views.py"), _finding("app/models.py")]
    grounded, ungrounded = partition_findings_by_read(findings, set())
    assert grounded == []
    assert ungrounded == findings


def test_path_normalisation_matches_across_slash_conventions():
    # Finding place has a leading slash; read path is relative with ./ prefix.
    findings = [_finding("/app/views.py")]
    grounded, ungrounded = partition_findings_by_read(findings, {"./app/views.py"})
    assert grounded == findings
    assert ungrounded == []


def test_mixed_batch_partitions_correctly():
    read = _finding("app/read.py")
    unread = _finding("app/hallucinated.py")
    url = AgentFinding(file="http://host/api", cwe=None)
    findings = [read, unread, url]
    grounded, ungrounded = partition_findings_by_read(findings, {"app/read.py"})
    assert grounded == [read, url]
    assert ungrounded == [unread]


def test_empty_read_set_still_passes_through_url_findings():
    url = AgentFinding(file="https://host/api", cwe=None)
    grounded, ungrounded = partition_findings_by_read([url], set())
    assert grounded == [url]
    assert ungrounded == []
