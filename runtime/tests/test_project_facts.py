from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import get_args

import jcs
import pytest
from fakes.worker_state import WorkerState
from google.adk.tools import FunctionTool
from test_finding_facades import context
from test_security_findings_toolset import FakeFindingClient, _settings, _workspace

from contractor_runtime.artifacts import ArtifactClient
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.toolsets.common.finding_locations import FindingSeverity
from contractor_runtime.toolsets.security_findings.facades import (
    GeneralFindingsToolsetFactory,
)
from contractor_runtime.toolsets.security_findings.submission import (
    MAX_FACT_AUTHORED_BYTES,
    MAX_FACT_EVIDENCE,
    build_submission,
    validate_informational_payload,
)


def test_record_fact_is_separately_selected_and_retains_optional_association():
    async def scenario():
        client = FakeFindingClient()
        tools = await GeneralFindingsToolsetFactory(
            lambda _allocation, _settings: ArtifactClient("allocation-test", client)
        ).create_selected(
            selected=["record_fact"],
            allocation_id="allocation-test",
            run_id="run-test",
            namespace="worker",
            runtime_settings=_settings(),
            workspace=_workspace(),
            resources=WorkerState().resources,
        )
        assert set(tools) == {"record_fact"}
        tool = tools["record_fact"]
        declaration = FunctionTool(tool)._get_declaration()
        assert declaration.name == "record_fact"
        evidence = [{"namespace": "notes", "name": "selected", "revision": "r1"}]
        for extra in ({}, {"item_key": "item-00"}, {"item_key": "item-01", "incidental": False}):
            result = await tool(
                title="nginx",
                description="Reported by the selected source, unverified.",
                evidence_refs=evidence,
                tool_context=context(),
                **extra,
            )
            assert result["finding_id"] == "finding-1"
            request = client.requests[-1]
            assert request["proposal"]["severity_suggestion"] == "informational"
            assert request.get("incidental", False) == extra.get("incidental", True)
            assert request.get("itemKey") == extra.get("item_key")
            assert "publishProblem" not in request
        with pytest.raises(ToolInputError, match="one to four"):
            await tool(title="nginx", description="Unsupported", tool_context=context())
        assert len(client.requests) == 3

    asyncio.run(scenario())


def test_shared_fact_payload_and_severity_contract():
    fixture = json.loads(
        (Path(__file__).parents[2] / "api/testdata/finding-fact-cases.json").read_text()
    )
    assert fixture["maxEvidence"] == MAX_FACT_EVIDENCE
    assert fixture["maxAuthoredBytes"] == MAX_FACT_AUTHORED_BYTES
    assert fixture["severities"] == list(get_args(FindingSeverity))
    for case in fixture["payloads"]:
        refs = [
            {"namespace": "proof", "name": f"proof-{i}", "revision": "r1"}
            for i in range(case["sources"])
        ]
        request = build_submission(
            invocation_id="invocation",
            client_key="fact",
            title="Candidate",
            description="x" * case["textBytes"],
            evidence_refs=refs,
            standard_refs=None,
            severity_suggestion="informational",
        )
        if case.get("authoredBytes"):
            proposal = request["proposal"]
            proposal["description"] = "x" * (
                case["authoredBytes"] - len(jcs.canonicalize(proposal)) + case["textBytes"]
            )
        if case["valid"]:
            validate_informational_payload(request["proposal"], case["sources"])
        else:
            with pytest.raises(ToolInputError):
                validate_informational_payload(request["proposal"], case["sources"])


@pytest.mark.parametrize(
    "namespace,name",
    [("worker", "memory.private"), ("worker", "http.body.private"), ("skills", "trace")],
)
def test_fact_sources_follow_common_model_visibility(namespace, name):
    with pytest.raises(ToolInputError, match="reserved"):
        build_submission(
            invocation_id="invocation",
            client_key="fact",
            title="Observed",
            description="Reported",
            evidence_refs=[{"namespace": namespace, "name": name, "revision": "r1"}],
            standard_refs=None,
        )
