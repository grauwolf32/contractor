"""Problem eligibility, actual injectable parameters and independent surface facts."""

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.toolsets.scan.findings import (
    CANDIDATE_MEDIA_TYPE,
    CANDIDATE_SCHEMA,
    nuclei_candidates,
    safe_subject,
    sqlmap_candidates,
    sqlmap_points,
)
from contractor_runtime.toolsets.scan.nuclei_package import finding_manifest_input
from contractor_runtime.toolsets.scan.tools import NucleiTool


def match(severity="medium", *, rule="git-config", url="https://target.test/.git/config"):
    return {
        "template-id": rule,
        "info": {"name": "Exposed configuration", "severity": severity},
        "matched-at": url,
        "request": "GET /.git/config HTTP/1.1\r\nHost: target.test\r\n\r\n",
        "response": "HTTP/1.1 200 OK\r\n\r\n",
    }


def observation(*rows, **changes):
    return {
        "status": "completed",
        "results": list(rows),
        "actualPackageDigest": "sha256:" + "a" * 64,
        **changes,
    }


@pytest.mark.parametrize("severity", ["low", "medium", "high", "critical"])
def test_nuclei_problem_severity_publishes_candidate_and_preserves_actual_surface(severity):
    result = nuclei_candidates(observation(match(severity)))
    assert len(result["candidates"]) == len(result["evidence"]["records"]) == 1
    assert result["candidates"][0]["severity"] == severity
    assert result["surface"] == [
        {"url": "https://target.test/.git/config", "method": "GET", "status": 200, "note": ""}
    ]
    assert result["complete"] is True


@pytest.mark.parametrize(
    "severity,reason",
    [
        ("info", "informational"),
        ("unknown", "severity_unavailable"),
        (None, "severity_unavailable"),
    ],
)
def test_nuclei_nonproblem_severity_keeps_technology_without_finding(severity, reason):
    row = match(severity, rule="nginx-detect", url="https://target.test/")
    row["info"]["metadata"] = {"product": "nginx"}
    row["extracted-results"] = ["1.24.0", "secret-token=must-not-enter-sitemap"]
    result = nuclei_candidates(observation(row))
    assert result["candidates"] == [] and result["skipped"] == {reason: 1}
    assert result["surface"][0]["note"] == "Technology: nginx; reported versions: 1.24.0"
    assert "secret-token" not in json.dumps(result)


def test_partial_scan_preserves_complete_positive_and_duplicate_row_identity():
    row = match()
    result = nuclei_candidates(
        observation(row, copy.deepcopy(row), status="failed", errorCode="scan_timeout")
    )
    assert result["complete"] is False and len(result["candidates"]) == 1
    assert result["skipped"] == {"duplicate_row": 1}
    assert result["candidates"][0]["qualifiedKey"] is not None


def test_nuclei_missing_http_facts_never_manufactures_method_or_status():
    row = match()
    del row["request"], row["response"]
    result = nuclei_candidates(observation(row))
    assert result["surface"][0]["method"] == "" and result["surface"][0]["status"] is None
    assert result["candidates"][0]["qualifiedKey"] is None


def test_endpoint_keys_exclude_credentials_and_query_values():
    assert (
        safe_subject("https://target.test/api?id=secret&token=secret")
        == "https://target.test/api?id=&token="
    )
    with pytest.raises(ValueError):
        safe_subject("https://user:secret@target.test/")
    result = nuclei_candidates(observation(match(url="https://target.test/api?token=secret")))
    assert "secret" not in json.dumps(result)


def test_sqlmap_technique_without_actual_parameter_cannot_fan_out():
    points = sqlmap_points(b"Type: boolean-based blind\nPayload: secret\n")
    assert points == []
    result = sqlmap_candidates(
        {"status": "completed", "injectionTechniques": ["boolean-based blind"]},
        url="https://target.test/?a=1&b=2",
        method="GET",
        points=points,
    )
    assert result["candidates"] == []
    assert result["skipped"] == {"injectable_parameter_unavailable": 1}


def test_sqlmap_actual_parameter_location_and_endpoint_are_distinct_keys():
    output = (
        b"Parameter: id (GET)\n Type: boolean-based blind\n Payload: secret\n"
        b"Parameter: other (POST)\n Type: time-based blind\n"
    )
    points = sqlmap_points(output)
    assert points == [
        {"name": "id", "place": "GET", "techniques": ["boolean-based blind"]},
        {"name": "other", "place": "POST", "techniques": ["time-based blind"]},
    ]
    first = sqlmap_candidates(
        {"status": "failed"}, url="https://target.test/orders?id=1", method="POST", points=points
    )
    other = sqlmap_candidates(
        {"status": "completed"}, url="https://target.test/users?id=1", method="POST", points=points
    )
    assert first["complete"] is False and len(first["candidates"]) == 2
    keys = [c["key"] for c in first["candidates"] + other["candidates"]]
    assert len(set(keys)) == 4 and "secret" not in json.dumps(first)


def test_candidate_cardinality_is_bounded_before_publication():
    result = nuclei_candidates(observation(*(match(rule=f"rule-{i}") for i in range(100))))
    assert len(result["candidates"]) == 100
    with pytest.raises(ValueError):
        nuclei_candidates(observation(*(match(rule=f"rule-{i}") for i in range(101))))


def test_explicit_output_binding_materializes_candidates_without_implicit_publication():
    async def scenario():
        writes = []

        class Client:
            async def read_artifact(self, ref, **kwargs):
                return SimpleNamespace(
                    data=json.dumps(
                        {
                            "schemaVersion": 2,
                            "selectedTemplateCount": 1,
                            "source": {"kind": "installed"},
                            "packageDigest": "sha256:" + "a" * 64,
                            "engineVersion": "v3.11.1",
                        }
                    ).encode(),
                    media_type="application/vnd.contractor.nuclei-manifest+json",
                )

            async def write_artifact(self, ref, **kwargs):
                writes.append((ref, kwargs))
                return SimpleNamespace(
                    artifact=ArtifactRef(
                        namespace=ref.namespace, name=ref.name, revision="exact-" + ref.name
                    )
                )

        tool = NucleiTool(SimpleNamespace(artifact_client=Client()), None)
        value = {
            "findingCandidates": nuclei_candidates(observation(match())),
            "templateManifest": {
                "namespace": "scanner",
                "name": "manifest",
                "revision": "manifest-r1",
            },
        }
        assert await tool.publish_outputs(value, {}) == value and not writes
        bindings = {
            "candidates": ArtifactRef(namespace="scanner", name="candidates"),
            "evidence": ArtifactRef(namespace="scanner", name="evidence"),
        }
        published = await tool.publish_outputs(value, bindings)
        assert len(writes) == 2 and set(published["artifacts"]) == set(bindings)
        batch = json.loads(writes[1][1]["data"])
        assert (
            batch["schema"] == CANDIDATE_SCHEMA
            and writes[1][1]["media_type"] == CANDIDATE_MEDIA_TYPE
        )
        assert batch["evidence"]["ref"]["revision"] == "exact-evidence"

    asyncio.run(scenario())


def test_nuclei_exact_template_method_supports_omitted_raw_messages():
    row = match("info", rule="nginx-detect")
    row["info"]["metadata"] = {"product": "nginx"}
    del row["request"], row["response"]
    result = nuclei_candidates(observation(row), methods={"nginx-detect": "HEAD"})
    assert result["surface"][0]["method"] == "HEAD"
    assert result["surface"][0]["status"] is None


def test_nuclei_manifest_selection_deduplicates_rules_and_rejects_unselected_rule():
    manifest = {
        "schemaVersion": 2,
        "packageDigest": "sha256:" + "a" * 64,
        "engineVersion": "v3.11.1",
        "selectedTemplateCount": 2,
        "source": {
            "kind": "rulepack",
            "artifact": {"namespace": "inputs", "name": "pack", "revision": "exact"},
            "contentDigest": "sha256:" + "b" * 64,
        },
        "selectedTemplates": [
            {"id": "git-config", "path": "git.yaml", "digest": "sha256:" + "c" * 64, "size": 123},
            {"id": "other", "path": "other.yaml", "digest": "sha256:" + "d" * 64, "size": 456},
        ],
    }
    batch = nuclei_candidates(observation(match(), match(url="https://target.test/other")))
    assert len(batch["candidates"]) == 2
    rules = {candidate["rule"] for candidate in batch["candidates"]}
    selected = finding_manifest_input(
        json.dumps(manifest).encode(),
        {"namespace": "scanner", "name": "manifest", "revision": "exact"},
        rules,
    )
    assert selected["templates"] == manifest["selectedTemplates"][:1]
    with pytest.raises(ValueError, match="absent"):
        finding_manifest_input(json.dumps(manifest).encode(), {}, {"not-selected"})


def test_empty_nuclei_candidates_do_not_read_or_retain_manifest():
    async def scenario():
        writes = []

        class Client:
            async def read_artifact(self, *args, **kwargs):
                pytest.fail("empty batch must not read a manifest")

            async def write_artifact(self, ref, **kwargs):
                writes.append(kwargs["data"])
                return SimpleNamespace(artifact=ref.model_copy(update={"revision": "exact"}))

        tool = NucleiTool(SimpleNamespace(artifact_client=Client()), None)
        await tool.publish_outputs(
            {"findingCandidates": nuclei_candidates(observation())},
            {
                name: ArtifactRef(namespace="scanner", name=name)
                for name in ("candidates", "evidence")
            },
        )
        assert json.loads(writes[0]) == {
            "schema": "contractor.finding.scanner-evidence.v1",
            "records": [],
        }

    asyncio.run(scenario())


def test_nuclei_report_projection_bounds_bytes_and_records_lost_candidates():
    rows = []
    for i in range(100):
        row = match(rule=f"rule-{i}")
        row["info"]["description"] = "x" * 8192
        rows.append(row)
    result = nuclei_candidates(observation(*rows))
    assert len(json.dumps(result, separators=(",", ":")).encode()) <= 256 << 10
    assert result["complete"] is False
    assert len(result["candidates"]) + result["skipped"]["candidate_bytes"] == 100


def test_shared_output_publication_failure_keeps_original_scanner_report(tmp_path):
    from test_tool_worker import ArtifactStore, FixtureTool, request, tool_spec, worker

    async def scenario():
        store, tool, spec = ArtifactStore(), FixtureTool(), tool_spec()
        tool.result = {"status": "completed", "results": [{"matched": True}]}

        async def fail_outputs(_observation, _outputs, _arguments):
            raise RuntimeError("selected output unavailable")

        tool.publish_outputs = fail_outputs
        runtime = await worker(tmp_path, store, tool, spec=spec)
        result = await runtime.invoke(request(spec))
        assert result.failure.code == "tool_report_failed"
        assert json.loads(store.values[(spec.namespace, "report")][0])["observation"] == tool.result
        assert await runtime.invoke(request(spec)) == result and len(tool.calls) == 1
        recreated = await worker(tmp_path, store, tool, spec=spec)
        assert (await recreated.invoke(request(spec))).failure.code == "tool_report_failed"
        assert len(tool.calls) == 1
        await recreated.finalize(spec.lease_expires_at)
        await runtime.finalize(spec.lease_expires_at)

    asyncio.run(scenario())


def test_sqlmap_json_parameter_uses_actual_body_location_without_tested_parameter_fanout():
    points = sqlmap_points(
        b"Parameter: JSON username ((custom) POST)\n Type: boolean-based blind\n"
    )
    assert points == [{"name": "username", "place": "JSON", "techniques": ["boolean-based blind"]}]
    result = sqlmap_candidates(
        {"status": "completed"}, url="https://target.test/users", method="POST", points=points
    )
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["qualifiedKey"][-2:] == ["JSON", "username"]
