from __future__ import annotations

import json
from pathlib import Path

from google.adk.tools import FunctionTool
from google.genai import types

from contractor_runtime.llm.openai import _tools
from contractor_runtime.toolsets.security_findings.facades import (
    CodeFindingTool,
    GeneralFindingTool,
    HTTPFindingTool,
)
from contractor_runtime.toolsets.security_findings.reader import ListFindingsTool


def provider_contracts() -> dict:
    tools = {
        "general": GeneralFindingTool(None),
        "code": CodeFindingTool(None),
        "http": HTTPFindingTool(None),
        "reader": ListFindingsTool((), "sha256:" + "a" * 64, None, ()),
    }
    return {
        name: _tools([types.Tool(function_declarations=[FunctionTool(tool)._get_declaration()])])[
            0
        ]["function"]
        for name, tool in tools.items()
    }


def test_complete_finding_worker_contract_is_unchanged() -> None:
    expected = json.loads(
        (Path(__file__).parent / "testdata/finding-tool-contracts.json").read_text()
    )
    assert provider_contracts() == expected


def test_shared_finding_fingerprint_canonical_vectors() -> None:
    from contractor_runtime.digests import jcs_digest

    cases = json.loads(
        (Path(__file__).parents[2] / "api/testdata/finding-fingerprint-cases.json").read_text()
    )
    for case in cases["vectors"]:
        assert jcs_digest(case["tuple"]) == case["digest"], case["name"]
