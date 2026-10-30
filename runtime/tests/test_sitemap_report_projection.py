"""Strict normalized-report adapters retain source facts without executing tools."""

import copy
import json

import pytest
from fakes.sitemap_reports import report_bytes

from contractor_runtime.toolsets.common.report_observations import project_report

AT = "2026-10-29T12:00:00Z"


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


def project(kind, value):
    return project_report(encoded(value), kind + "-report@1", AT)


@pytest.mark.parametrize("kind", ["ffuf", "katana"])
def test_shared_report_projects_actual_responses_with_publication_timestamp(kind):
    result = project_report(report_bytes(kind), kind + "-report@1", AT)
    assert result == {
        "kind": kind,
        "observations": [
            {
                "url": "https://example.test/users?expand=profile",
                "method": "GET",
                "outcome": "response",
                "status": 200,
                "at": AT,
            }
        ],
    }


@pytest.mark.parametrize(
    "kind,path,value",
    [
        ("ffuf", ("extra",), True),
        ("ffuf", ("schemaVersion",), True),
        ("ffuf", ("schemaVersion",), 2),
        ("ffuf", ("tool",), "scan_katana"),
        ("ffuf", ("inputDigest",), "sha256:wrong"),
        ("ffuf", ("observation", "results"), None),
        ("ffuf", ("observation", "results", 0, "headers"), {}),
        ("ffuf", ("observation", "results", 0, "status"), True),
        ("ffuf", ("observation", "results", 0, "status"), 0),
        ("ffuf", ("observation", "results", 0, "length"), -1),
        ("ffuf", ("observation", "results", 0, "position"), 2),
        ("ffuf", ("observation", "results", 0, "url"), "file:///tmp/report"),
        ("ffuf", ("observation", "results", 0, "url"), "https://alice:pass@example.test/"),
        ("ffuf", ("observation", "results", 0, "url"), "https://example.test/#fragment"),
        ("ffuf", ("observation", "results", 0, "contentType"), "x" * (128 << 10)),
        ("ffuf", ("observation", "payloadsAttempted"), None),
        ("ffuf", ("observation", "stdout"), "raw diagnostics"),
        ("ffuf", ("observation", "invalidResultLines"), 2**53),
        ("katana", ("observation", "results", 0, "url"), "https://foreign.test/"),
        ("katana", ("observation", "results", 0, "method"), "POST"),
        ("katana", ("observation", "targetsDigest"), "sha256:" + "b" * 64),
        ("katana", ("observation", "coverage", "limitations"), ["unknown"]),
        ("katana", ("observation", "coverage", "responses"), 2),
        ("katana", ("observation", "targetsArtifact", "revision"), "another"),
        ("katana", ("observation", "limits", "maxTargets"), 101),
        ("katana", ("observation", "discoveryComplete"), True),
    ],
    ids=lambda value: str(value)[:60],
)
def test_shared_malformed_report_is_rejected_before_batch_publication(kind, path, value):
    envelope = json.loads(report_bytes(kind))
    target = envelope
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        project(kind, envelope)


def test_shared_report_rejects_missing_nullable_members_and_non_strict_json():
    envelope = json.loads(report_bytes("ffuf"))
    del envelope["observation"]["errorCode"]
    with pytest.raises(ValueError):
        project("ffuf", envelope)
    source = report_bytes("ffuf")
    for raw in (
        source.replace(b'"schemaVersion": 1', b'"schemaVersion": 1, "schemaVersion": 1'),
        source.replace(b'"schemaVersion": 1', b'"SchemaVersion": 1'),
        source + b"{}",
        b'"' + b"a" * (512 << 10) + b'"',
        source.replace(b'"durationMs": 0', b'"durationMs": NaN'),
        source.replace(b'"contentType": "application/json"', b'"contentType": "\\ud800"'),
    ):
        with pytest.raises(ValueError):
            project_report(raw, "ffuf-report@1", AT)


def test_shared_reports_preserve_partial_results_and_aggregate_truthful_source_gaps():
    ffuf = json.loads(report_bytes("ffuf"))
    ffuf["observation"].update(
        status="failed",
        errorCode="scan_request_failed",
        scanComplete=False,
        wordlistEntries=10,
        payloadsAttempted=7,
        requestErrors=2,
        resultsTruncated=True,
        invalidResultLines=3,
    )
    result = project("ffuf", ffuf)
    assert len(result["observations"]) == 1
    assert {gap["reason"]: gap["count"] for gap in result["gaps"]} == {
        "incomplete": 1,
        "projection": 3,
        "report-truncated": 1,
        "request-errors": 2,
    }
    katana = json.loads(report_bytes("katana"))
    report = katana["observation"]
    report.update(
        status="failed",
        errorCode="scan_request_failed",
        requestErrors=2,
        invalidResultLines=3,
        resultsTruncated=True,
    )
    report["coverage"].update(
        responses=5,
        uniqueTargets=5,
        omittedTargets=4,
        requestErrors=2,
        invalidResultLines=3,
        depthLimited=1,
        unsupportedMethods=1,
        observations=12,
    )
    report["coverage"]["limitations"] += [
        "depth_limit",
        "request_errors",
        "unsupported_methods",
        "invalid_output",
        "target_export_limit",
    ]
    result = project("katana", katana)
    assert len(result["observations"]) == 1
    assert {gap["reason"]: gap["count"] for gap in result["gaps"]} == {
        "incomplete": 1,
        "projection": 5,
        "members": 4,
        "report-truncated": 1,
        "request-errors": 2,
    }


def test_shared_report_accepts_one_hundred_results_and_empty_report_without_inventing_response():
    envelope = json.loads(report_bytes("ffuf"))
    report = envelope["observation"]
    row = report["results"][0]
    report.update(wordlistEntries=100, payloadsAttempted=100)
    report["results"] = [
        {**copy.deepcopy(row), "position": index + 1, "url": f"https://example.test/{index}"}
        for index in range(100)
    ]
    assert len(project("ffuf", envelope)["observations"]) == 100
    report["results"].append(copy.deepcopy(row))
    with pytest.raises(ValueError):
        project("ffuf", envelope)
    report["results"] = []
    assert project("ffuf", envelope) == {"kind": "ffuf", "observations": []}
