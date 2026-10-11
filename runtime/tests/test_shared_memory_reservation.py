"""The reserved-binding rule the Runtime shares with the Go Server.

internal/artifactpolicy and internal/contracts read the same api/testdata
fixture; make test-wire-cross-language runs both sides.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from contractor_runtime.contracts import WorkerCompletion
from contractor_runtime.contracts.artifacts import (
    MEMORY_ARTIFACT_PREFIX,
    PURPOSE_RESERVED_NAMESPACES,
    is_reserved_memory_binding,
    is_reserved_result_binding,
)
from contractor_runtime.toolsets.memory.tools import MemoryToolsetFactory

FIXTURES = Path(__file__).resolve().parents[2] / "api" / "testdata" / "v1alpha1"


def _cases() -> dict[str, Any]:
    cases = json.loads((FIXTURES / "memory-reservation-cases.json").read_text(encoding="utf-8"))
    if set(cases) != {"description", "memoryPrefix", "purposeReservedNamespaces", "bindings"}:
        raise ValueError("memory-reservation-cases.json has an unexpected shape")
    for binding in cases["bindings"]:
        if set(binding) != {"namespace", "name", "memoryReserved", "resultReserved"}:
            raise ValueError(f"binding {binding} needs a namespace, a name and both verdicts")
    return cases


CASES = _cases()


def test_shared_memory_reservation_cases() -> None:
    assert CASES["memoryPrefix"] == MEMORY_ARTIFACT_PREFIX
    assert CASES["purposeReservedNamespaces"] == sorted(PURPOSE_RESERVED_NAMESPACES)
    assert CASES["bindings"]
    for binding in CASES["bindings"]:
        namespace, name = binding["namespace"], binding["name"]
        assert is_reserved_memory_binding(namespace, name) is binding["memoryReserved"], binding
        assert is_reserved_result_binding(namespace, name) is binding["resultReserved"], binding


@pytest.mark.parametrize(
    "binding", CASES["bindings"], ids=lambda value: f"{value['namespace']}/{value['name']}"
)
def test_shared_worker_result_refuses_reserved_bindings(binding: dict[str, Any]) -> None:
    # The Go twin is TestSharedWorkerResultReservedBindings.
    completion = json.loads(
        (FIXTURES / "valid" / "worker-completion-success.json").read_text(encoding="utf-8")
    )
    completion["result"]["artifacts"] = {
        "report": {"namespace": binding["namespace"], "name": binding["name"], "revision": "r1"}
    }
    if binding["resultReserved"]:
        with pytest.raises(ValidationError):
            WorkerCompletion.model_validate_json(json.dumps(completion))
    else:
        WorkerCompletion.model_validate_json(json.dumps(completion))


@pytest.mark.parametrize("namespace", CASES["purposeReservedNamespaces"])
def test_shared_memory_tools_refuse_purpose_reserved_namespaces(namespace: str) -> None:
    # The Server never binds a Stage Agent, and so MemoryTools, to these
    # Namespaces (artifactpolicy.IsPurposeReservedNamespace).
    async def scenario() -> None:
        factory = MemoryToolsetFactory(lambda *_args: None)
        with pytest.raises(ValueError, match="non-purpose Agent Namespace"):
            await factory.create_selected(
                selected=["list_memories"],
                allocation_id="allocation",
                run_id="run",
                namespace=namespace,
                runtime_settings=None,
                workspace=None,
                state=None,
            )

    asyncio.run(scenario())
