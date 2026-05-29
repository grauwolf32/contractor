"""Unit tests for evidence_request_ids capture on verified findings.

The exploit agent passes the request_tag values of its proof probes; they
must round-trip through the verification artifact so the pipeline can collect
the raw HTTP chain afterwards.
"""
from __future__ import annotations

import pytest
from google.genai import types

from contractor.tools.vuln import (VerifiedFinding, VerifiedFindingFormat,
                                    VerifiedFindingsTools, verification_tools)


class InMemoryArtifactCtx:
    """Minimal ToolContext stand-in backed by an in-memory artifact store."""

    def __init__(self) -> None:
        self.store: dict[str, types.Part] = {}

    async def save_artifact(self, *, filename: str, artifact: types.Part) -> None:
        self.store[filename] = artifact

    async def load_artifact(self, *, filename: str) -> types.Part | None:
        return self.store.get(filename)


def _tool(tools, name):
    return next(t for t in tools if t.__name__ == name)


@pytest.mark.asyncio
async def test_submit_verdict_persists_request_ids():
    ctx = InMemoryArtifactCtx()
    tools = verification_tools(name="ns", fmt=VerifiedFindingFormat("yaml"))
    submit = _tool(tools, "submit_verdict")

    await submit(
        name="finding-1",
        verdict="exploitable",
        summary="reflected",
        entry_point="/api/x",
        evidence="see chain",
        tool_context=ctx,
        request_ids=["rABC-h000001", "rABC-c000002"],
    )

    store = VerifiedFindingsTools(name="ns")
    await store.load(ctx)
    finding = store.findings["finding-1"]
    assert finding.evidence_request_ids == ["rABC-h000001", "rABC-c000002"]


@pytest.mark.asyncio
async def test_report_verification_defaults_to_empty_ids():
    ctx = InMemoryArtifactCtx()
    tools = verification_tools(name="ns")
    report = _tool(tools, "report_verification")

    await report(
        name="finding-2",
        source_namespace="ns",
        verdict="inconclusive",
        summary="unclear",
        attacker_control_at_sink="none",
        sink_reached=False,
        entry_point="/api/y",
        data_flow=[],
        impact="",
        notes="tried stuff",
        tool_context=ctx,
    )

    store = VerifiedFindingsTools(name="ns")
    await store.load(ctx)
    assert store.findings["finding-2"].evidence_request_ids == []


def test_markdown_renders_request_ids():
    finding = VerifiedFinding(
        name="f",
        source_namespace="ns",
        verdict="exploitable",
        summary="s",
        attacker_control_at_sink="full",
        sink_reached=True,
        entry_point="/x",
        evidence_request_ids=["rABC-h000001"],
    )
    md = VerifiedFindingFormat("markdown").format_finding(finding)
    assert "rABC-h000001" in md
