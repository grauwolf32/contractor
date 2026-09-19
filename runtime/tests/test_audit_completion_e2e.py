"""Real Runner -> mTLS Artifact API probe, launched by the Go importer gate.

The Go parent owns disposable PostgreSQL, trusted Run creation, certificates,
fault injection and importer assertions. Only the model is scripted here.
"""

import asyncio
import hashlib
import io
import json
import os
import zipfile
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fakes.model import scripted_model, text_result, tool_call
from test_adk_runtime import build_context, stage_request
from test_audit_completion_continuation import CountedState

from contractor_runtime.artifacts import ArtifactClient, MTLSArtifactTransport
from contractor_runtime.contracts import WorkerCompletionContract
from contractor_runtime.mtls import runtime_agent_client_context
from contractor_runtime.toolsets.audit_results.collector import AuditCollectionError
from contractor_runtime.toolsets.audit_results.v2 import AuditResultsToolsetFactory
from contractor_runtime.worker.runtime import AdkWorkerRuntime


def test_runtime_zip_for_go_importer(tmp_path):
    specification = os.environ.get("CONTRACTOR_AUDIT_COMPLETION_BRIDGE")
    if not specification:
        pytest.skip("launched by make test-audit-completion-e2e with a real Go Artifact API")
    spec = json.loads(Path(specification).read_text())

    async def scenario():
        contract = WorkerCompletionContract.model_validate(spec["contract"])
        mode = spec["mode"]
        transport = MTLSArtifactTransport(
            spec["apiURL"],
            runtime_agent_client_context(
                ca_file=spec["ca"],
                certificate_file=spec["certificate"],
                private_key_file=spec["privateKey"],
            ),
            timeout_seconds=3,
            runtime_instance_id="runtime-1",
        )
        client = ArtifactClient(spec["allocationID"], transport)
        state = CountedState()
        context = replace(
            build_context(tmp_path, state, {}),
            allocation_id=spec["allocationID"],
            run_id=spec["runID"],
            namespace=contract.result_artifact.namespace,
        )
        write_started = asyncio.Event()
        original_write = client.write_artifact

        async def write_with_fault(*args, **kwargs):
            if mode == "cancel-before-write":
                write_started.set()
                await asyncio.Event().wait()
            result = await original_write(*args, **kwargs)
            if mode in {"crash", "proposal-crash"}:
                # Abrupt process loss: no Runtime completion or cleanup can run.
                os._exit(73)
            if mode == "cancel-after-write":
                write_started.set()
                await asyncio.Event().wait()
            return result

        client.write_artifact = write_with_fault
        selected = await AuditResultsToolsetFactory(lambda *_: client).create_selected(
            selected=["read_audit_task", "submit_check_result"],
            allocation_id=spec["allocationID"],
            namespace=context.namespace,
            runtime_settings=context.runtime_settings,
            state=state,
            completion_contract=contract,
        )

        def submit(key, *, invalid=False):
            return tool_call(
                "submit_check_result",
                {
                    "item_key": key,
                    "assessment": "satisfied",
                    "summary": "Different result." if mode == "changed" else "Verified control.",
                    "completed": ["source-trace"],
                    "gaps": [],
                    "evidence": []
                    if invalid
                    else [{"kind": "source-trace", "summary": "Guard at app.py:12."}],
                    "proposal_keys": ["candidate"] if mode.startswith("proposal") else [],
                },
                call_id=key + ("-invalid" if invalid else "-valid"),
            )

        first, second = spec["itemKeys"]
        responses = [submit(second), text_result("Partial work."), submit(first), text_result("")]
        if mode == "correction":
            responses.insert(0, submit(second, invalid=True))
        elif mode in {"missing", "partial", "invalid"}:
            responses = [] if mode == "missing" else [submit(second, invalid=mode == "invalid")]
            responses += [text_result("Everything is complete.")] * 3
        model = scripted_model(responses)
        context = replace(
            context,
            tools=selected,
            completion_contract=contract,
            model_policy=context.model_policy.model_copy(
                update={
                    "max_model_calls": len(responses),
                    "max_tool_calls": 3 if mode == "correction" else 2,
                    "max_total_tokens": 10 * len(responses),
                }
            ),
        )
        runtime = AdkWorkerRuntime(context, model)
        await runtime.start()
        request = stage_request().model_copy(
            update={
                "artifacts": {
                    "task": contract.task,
                    "execution_manifest": contract.execution_manifest,
                },
                "result_artifacts": {"result": contract.result_artifact},
            }
        )
        if mode.startswith("cancel-"):
            running = asyncio.create_task(runtime.invoke(request))
            await asyncio.wait_for(write_started.wait(), 15)
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
            report = {"outcome": "cancelled"}
        else:
            completion = await runtime.invoke(request)
            if spec.get("expectConflict"):
                assert completion.result is None, completion
                assert completion.failure.code == "audit_result_publication_conflict", completion
                report = {"outcome": "conflict"}
            elif mode in {"missing", "partial", "invalid"}:
                assert completion.result is None, completion
                assert completion.failure.code == "audit_result_incomplete", completion
                report = {"outcome": "incomplete"}
            else:
                assert completion.failure is None, completion
                assert not completion.result.summarized
                exact = completion.result.artifacts["result"]
                assert exact.namespace == contract.result_artifact.namespace
                assert exact.name == contract.result_artifact.name and exact.revision
                stored = await client.read_artifact(exact)
                with zipfile.ZipFile(io.BytesIO(stored.data)) as archive:
                    results = json.loads(archive.read("check-results.json"))["results"]
                assert [value["item_key"] for value in results] == spec["itemKeys"]
                assert len(model.requests) == len(responses)
                assert "Runtime completion reminder" in model.requests[-1]["contentText"]
                report = {
                    "outcome": "succeeded",
                    "artifact": exact.model_dump(by_alias=True),
                    "digest": "sha256:" + hashlib.sha256(stored.data).hexdigest(),
                }
        assert runtime._result_finalizer is None
        assert all(not value["hasResponseSchema"] for value in model.requests)
        assert len(model.requests) <= len(responses)
        assert state.begins == state.completions == 1
        with pytest.raises(AuditCollectionError):
            runtime._completion.current()
        await runtime.abort(datetime.now(UTC) + timedelta(seconds=2))
        report["modelCalls"] = len(model.requests)
        Path(spec["report"]).write_text(json.dumps(report))

    asyncio.run(scenario())
