"""Production ADK Runner and completion dispatch with offline model/artifact IO."""

import asyncio
import io
import json
import zipfile
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from fakes.model import scripted_model, text_result, tool_call
from test_adk_runtime import build_context, stage_request
from test_audit_completion_continuation import CountedState
from test_audit_result_collector import arguments
from test_audit_result_publication import assigned

from contractor_runtime.artifacts import ArtifactAPIError, ArtifactValue
from contractor_runtime.contracts import ArtifactRef, ArtifactWriteResult, WorkerCompletionContract
from contractor_runtime.toolsets.audit_results.collector import AuditCollectionError
from contractor_runtime.toolsets.audit_results.v2 import AuditResultsToolsetFactory
from contractor_runtime.worker.runtime import AdkWorkerRuntime


class AuditClient:
    allocation_id = "allocation-1"

    def __init__(self, inputs):
        self.inputs = inputs
        self.reads = []
        self.writes = []
        self.on_write = None
        self.write_started = asyncio.Event()
        self.block_write = False
        self.write_error = None

    async def read_artifact(self, ref, *, max_bytes):
        self.reads.append(ref.model_copy(deep=True))
        assert ref.revision == "pinned-r1"
        data, media = (
            (self.inputs.task_package, "application/zip")
            if ref.name == "task"
            else (self.inputs.execution_manifest, "application/json")
        )
        assert len(data) <= max_bytes
        return ArtifactValue(
            ref.model_copy(deep=True), media, data, datetime.now(UTC), datetime.now(UTC)
        )

    async def write_artifact(self, target, *, data, media_type, expected_revision):
        assert expected_revision is None
        assert target == ArtifactRef(namespace="builder", name="report")
        self.writes.append(data)
        self.write_started.set()
        if self.on_write:
            await self.on_write()
        if self.block_write:
            await asyncio.Event().wait()
        if self.write_error:
            raise self.write_error
        return ArtifactWriteResult(
            apiVersion="contractor/v1alpha1",
            artifact=target.model_copy(update={"revision": "published-r1"}),
            mediaType=media_type,
            size=len(data),
        )


def contract():
    return WorkerCompletionContract(
        kind="audit-check-results@1",
        task=ArtifactRef(namespace="inputs", name="task", revision="pinned-r1"),
        executionManifest=ArtifactRef(
            namespace="inputs", name="execution_manifest", revision="pinned-r1"
        ),
        resultArtifact=ArtifactRef(namespace="builder", name="report"),
    )


async def runtime_for(
    tmp_path,
    responses,
    *,
    batch=False,
    model_calls=8,
    tool_calls=8,
    total_tokens=1000,
    block_call=None,
    extra_tools=None,
):
    expected, inputs = assigned(batch=batch)
    client = AuditClient(inputs)
    state = CountedState()
    context = build_context(tmp_path, state, {})
    tools = await AuditResultsToolsetFactory(lambda *_: client).create_selected(
        selected=["read_audit_task", "submit_check_result"],
        allocation_id="allocation-1",
        namespace="builder",
        runtime_settings=context.runtime_settings,
        state=state,
        completion_contract=contract(),
    )
    context = replace(
        context,
        tools={**tools, **(extra_tools or {})},
        completion_contract=contract(),
        model_policy=context.model_policy.model_copy(
            update={
                "max_model_calls": model_calls,
                "max_tool_calls": tool_calls,
                "max_total_tokens": total_tokens,
            }
        ),
    )
    model = scripted_model(responses, block_call_number=block_call)
    runtime = AdkWorkerRuntime(context, model)
    await runtime.start()
    return runtime, model, client, state, expected


def submit(item, call="submit-1"):
    return tool_call(
        "submit_check_result", {"item_key": item.item_key, **arguments(item)}, call_id=call
    )


def test_real_runner_continues_once_with_one_owner_and_publishes_on_last_allowed_call(tmp_path):
    async def scenario():
        expected, _ = assigned(batch=True)
        first, second = [entry.value for entry in expected.items]
        runtime, model, client, state, _ = await runtime_for(
            tmp_path,
            [
                submit(second),
                text_result("Partial work."),
                submit(first, "submit-2"),
                text_result(""),
            ],
            batch=True,
            model_calls=4,
            tool_calls=2,
            total_tokens=40,
        )

        async def before_write():
            snapshot = await state.snapshot()
            assert snapshot["currentInvocation"] is not None
            assert state.begins == 1 and state.completions == 0

        client.on_write = before_write
        completion = await runtime.invoke(stage_request())
        assert completion.failure is None, completion
        assert completion.result.summarized is False
        assert completion.result.artifacts == {
            "report": ArtifactRef(namespace="builder", name="report", revision="published-r1")
        }
        assert runtime._result_finalizer is None
        assert len(model.requests) == 4 and len(client.writes) == 1
        assert len(client.reads) == 2
        assert "Runtime completion reminder" in model.requests[-1]["contentText"]
        assert all(not request["hasResponseSchema"] for request in model.requests)
        assert state.begins == state.completions == 1
        snapshot = await state.snapshot()
        metrics = snapshot["lastCompletedInvocation"]["metrics"]
        assert metrics["modelCalls"] == 4 and metrics["toolCalls"] == 2
        assert runtime._plugin._next_tool_ordinal == 3
        assert completion.result.observations.tools["submit_check_result"].calls == 2
        with zipfile.ZipFile(io.BytesIO(client.writes[0])) as archive:
            results = json.loads(archive.read("check-results.json"))["results"]
        assert [result["item_key"] for result in results] == list(expected.owner.item_keys)
        with pytest.raises(AuditCollectionError):
            runtime._completion.current()
        await runtime.abort(datetime.now(UTC) + timedelta(seconds=2))

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["missing", "partial", "invalid"])
def test_missing_partial_invalid_results_cannot_be_bypassed_by_prose(tmp_path, mode):
    async def scenario():
        expected, _ = assigned(batch=True)
        responses = []
        if mode != "missing":
            value = expected.items[0].value
            responses.append(submit(replace(value, evidence=()) if mode == "invalid" else value))
        responses += [text_result("Everything is complete and published.")] * 3
        runtime, model, client, state, _ = await runtime_for(tmp_path, responses, batch=True)
        result = await runtime.invoke(stage_request())
        assert result.result is None and result.failure.code == "audit_result_incomplete"
        assert result.failure.retryable
        assert not client.writes and len(model.requests) == len(responses)
        assert state.begins == state.completions == 1
        assert (
            sum("Runtime completion reminder" in req["contentText"] for req in model.requests) == 2
        )
        await runtime.abort(datetime.now(UTC) + timedelta(seconds=2))

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["model", "tool", "tokens", "main-error"])
def test_hard_failures_win_even_after_full_collection(tmp_path, mode):
    async def scenario():
        expected, _ = assigned()
        item = expected.items[0].value
        responses = [submit(item)]
        if mode == "tool":
            responses += [submit(item, "forbidden"), text_result("Done")]
        elif mode != "main-error":
            responses += [text_result("Done")]
        runtime, model, client, _, _ = await runtime_for(
            tmp_path,
            responses,
            model_calls=1 if mode == "model" else 8,
            tool_calls=1,
            total_tokens=19 if mode == "tokens" else 1000,
        )
        result = await runtime.invoke(stage_request())
        assert result.result is None and not client.writes
        assert result.failure.code == (
            "worker_execution_failed" if mode == "main-error" else "worker_budget_exhausted"
        )
        assert len(model.requests) <= (3 if mode == "tool" else 2)
        await runtime.abort(datetime.now(UTC) + timedelta(seconds=2))

    asyncio.run(scenario())


def test_missing_collection_cannot_exceed_model_budget_for_reminder(tmp_path):
    async def scenario():
        runtime, model, client, _, _ = await runtime_for(
            tmp_path, [text_result("Done")], model_calls=1
        )
        result = await runtime.invoke(stage_request())
        assert result.failure.code == "worker_budget_exhausted"
        assert len(model.requests) == 1 and not client.writes
        await runtime.abort(datetime.now(UTC) + timedelta(seconds=2))

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["continuation", "publication"])
def test_cancellation_discards_collection_and_finishes_state_once(tmp_path, phase):
    async def scenario():
        expected, _ = assigned(batch=phase == "continuation")
        responses = [submit(expected.items[0].value), text_result("Done"), text_result("Done")]
        runtime, model, client, state, _ = await runtime_for(
            tmp_path,
            responses,
            batch=phase == "continuation",
            block_call=3 if phase == "continuation" else None,
        )
        client.block_write = phase == "publication"
        running = asyncio.create_task(runtime.invoke(stage_request()))
        await asyncio.wait_for(
            model.blocked.wait() if phase == "continuation" else client.write_started.wait(), 2
        )
        collector = runtime._completion.current()
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert state.begins == state.completions == 1
        assert (await state.snapshot())["lastCompletedInvocation"]["phase"] == "cancelled"
        with pytest.raises(AuditCollectionError):
            await collector.snapshot()
        assert len(client.writes) == int(phase == "publication")
        await runtime.abort(datetime.now(UTC) + timedelta(seconds=2))

    asyncio.run(scenario())


def test_shared_session_reuse_never_inherits_accepted_results_or_tool_authority(tmp_path):
    async def scenario():
        expected, _ = assigned()
        runtime, model, client, state, _ = await runtime_for(
            tmp_path,
            [
                submit(expected.items[0].value),
                text_result("Done"),
                text_result("Already done"),
                text_result("Already done"),
                text_result("Already done"),
            ],
        )
        first = await runtime.invoke(stage_request())
        second = await runtime.invoke(stage_request())
        assert first.result is not None and second.failure.code == "audit_result_incomplete"
        assert first.invocation_id != second.invocation_id
        assert len(client.writes) == 1 and len(client.reads) == 2
        assert state.begins == state.completions == 2
        stale = await runtime._context.tools["submit_check_result"](
            SimpleNamespace(invocation_id=first.invocation_id),
            **arguments(expected.items[0].value),
        )
        assert stale["status"] == "error"
        assert len(model.requests) == 5
        await runtime.abort(datetime.now(UTC) + timedelta(seconds=2))

    asyncio.run(scenario())


def test_publication_failure_is_not_misclassified_as_model_failure(tmp_path):
    async def scenario():
        expected, _ = assigned()
        runtime, _, client, state, _ = await runtime_for(
            tmp_path, [submit(expected.items[0].value), text_result("Done")]
        )
        client.write_error = ArtifactAPIError(403, "allocation_fenced", False)
        result = await runtime.invoke(stage_request())
        assert result.failure.code == "audit_result_publication_failed"
        assert not result.failure.retryable and result.result is None
        assert not state.metrics.counters.get("llm_errors", 0)
        await runtime.abort(datetime.now(UTC) + timedelta(seconds=2))

    asyncio.run(scenario())


def test_exact_preparation_rejects_alias_receipt_and_toolset_without_contract(tmp_path):
    async def scenario():
        _, inputs = assigned()
        client = AuditClient(inputs)
        state = CountedState()
        context = build_context(tmp_path, state, {})
        factory = AuditResultsToolsetFactory(lambda *_: client)
        kwargs = dict(
            selected=["read_audit_task", "submit_check_result"],
            allocation_id="allocation-1",
            namespace="builder",
            runtime_settings=context.runtime_settings,
            state=state,
        )
        with pytest.raises(ValueError):
            await factory.create_selected(**kwargs)
        assert not client.reads
        original = client.read_artifact

        async def wrong_receipt(ref, **kwargs):
            value = await original(ref, **kwargs)
            return replace(value, artifact=ref.model_copy(update={"revision": "new-alias-r2"}))

        client.read_artifact = wrong_receipt
        with pytest.raises(ValueError):
            await factory.create_selected(**kwargs, completion_contract=contract())

    asyncio.run(scenario())


def test_artifact_observations_survive_a_reminder_and_join_verified_publication(tmp_path):
    async def scenario():
        exact = ArtifactRef(namespace="builder", name="trace", revision="trace-r1")

        async def observe_trace() -> dict:
            """Observe one exact fixture artifact."""
            observe_trace.known_exact_refs = (exact,)
            observe_trace.artifact_observation_cursor = 1
            return {"artifact": exact.model_dump(by_alias=True)}

        observe_trace.known_exact_refs = ()
        observe_trace.artifact_observation_cursor = 0
        observe_trace.observed_exact_refs_since = lambda cursor: (exact,) if cursor == 0 else ()
        expected, _ = assigned()
        runtime, _, client, _, _ = await runtime_for(
            tmp_path,
            [
                tool_call("observe_trace", {}, call_id="trace"),
                text_result("Investigated"),
                submit(expected.items[0].value),
                text_result("Done"),
            ],
            extra_tools={"observe_trace": observe_trace},
            model_calls=4,
            tool_calls=2,
        )
        request = stage_request()
        request.result_artifacts["trace"] = ArtifactRef(namespace="builder", name="trace")
        result = await runtime.invoke(request)
        assert result.failure is None
        assert result.result.artifacts["trace"] == exact
        assert result.result.artifacts["report"].revision == "published-r1"
        assert len(client.writes) == 1
        await runtime.abort(datetime.now(UTC) + timedelta(seconds=2))

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["model-event", "sandbox", "timeout"])
def test_fatal_model_event_sandbox_or_deadline_prevents_audit_success(tmp_path, failure):
    async def scenario():
        from contractor_runtime.sandbox.contracts import SandboxErrorCode

        expected, _ = assigned()
        final = text_result("Done")
        if failure == "model-event":
            final.error_code = "MODEL_ERROR"
            final.error_message = "private provider message"
        runtime, _, client, state, _ = await runtime_for(
            tmp_path,
            [submit(expected.items[0].value), final],
            block_call=2 if failure == "timeout" else None,
        )
        if failure == "timeout":
            runtime._context.runtime_settings.request_timeout_seconds = 1
        if failure == "sandbox":

            async def fail_sandbox():
                state.execution.fail(SandboxErrorCode.UNAVAILABLE)

            client.on_write = fail_sandbox
        result = await runtime.invoke(stage_request())
        assert result.result is None
        assert (
            result.failure.code
            == {
                "model-event": "worker_execution_failed",
                "sandbox": "sandbox_unavailable",
                "timeout": "worker_timeout",
            }[failure]
        )
        assert state.begins == state.completions == 1
        assert not client.writes or failure == "sandbox"
        with pytest.raises(AuditCollectionError):
            runtime._completion.current()
        await runtime.abort(datetime.now(UTC) + timedelta(seconds=2))

    asyncio.run(scenario())


def test_allocation_preparation_and_registration_require_positive_completion_support(tmp_path):
    async def scenario():
        from fakes.spec import allocation_spec
        from test_capabilities import make_settings

        from contractor_runtime.allocation import AllocationError, AllocationService
        from contractor_runtime.capabilities import CapabilitySnapshot, discover_capabilities
        from contractor_runtime.contracts import ToolsetRef, ToolsetSelection
        from contractor_runtime.digests import _agent_template_digest
        from contractor_runtime.factories import built_in_factories
        from contractor_runtime.state import RuntimeState

        expected, inputs = assigned()
        client = AuditClient(inputs)
        model = scripted_model([submit(expected.items[0].value), text_result("Done")])
        factories = built_in_factories(
            tmp_path, artifact_client_factory=lambda *_: client, model_factory=lambda _: model
        )
        # Restrict probes to this test's actual dependencies.
        factories = replace(
            factories, toolsets={"audit-results@2": factories.toolsets["audit-results@2"]}
        )
        capabilities = await discover_capabilities(factories)
        assert capabilities.completion_contracts == ("audit-check-results@1",)
        state = RuntimeState(capabilities=capabilities)
        registration = await state.registration(make_settings(tmp_path))
        assert registration.capabilities.completion_contracts == ["audit-check-results@1"]
        await state.mark_registered()
        service = AllocationService(
            state, factories, capabilities, a2a_base_url="https://runtime.example"
        )
        spec = allocation_spec()
        spec.agent_template.toolsets = [
            ToolsetSelection(
                ref=ToolsetRef(toolsetId="audit-results", version="2"),
                tools=["read_audit_task", "submit_check_result"],
            )
        ]
        spec.agent_template.ref.digest = _agent_template_digest(spec.agent_template)
        with pytest.raises(AllocationError, match="trusted"):
            await service.prepare(spec)
        assert not client.reads
        spec.completion_contract = contract()
        response = await service.prepare(spec)
        assert response.worker_handle.allocation_id == spec.allocation_id
        assert (await service.prepare(spec)) == response
        worker = service._context.worker
        assert worker._context.completion_contract == contract()
        assert (await worker.invoke(stage_request())).result is not None
        await worker.abort(datetime.now(UTC) + timedelta(seconds=2))
        disabled = CapabilitySnapshot.create(
            runtimes=["adk@1"],
            sandbox_profiles=["local-workdir@1"],
            toolsets={"audit-results@2": ["read_audit_task", "submit_check_result"]},
        )
        unsupported = AllocationService(
            RuntimeState(capabilities=disabled),
            factories,
            disabled,
            a2a_base_url="https://runtime.example",
        )
        with pytest.raises(AllocationError):
            unsupported._validate_spec(spec)
        missing_transport = built_in_factories(tmp_path)
        assert not await missing_transport.toolsets["audit-results@2"].probe()

    asyncio.run(scenario())


def test_cancellation_during_normal_discard_keeps_ownership_until_cleanup_settles(tmp_path):
    async def scenario():
        expected, _ = assigned()
        runtime, _, client, state, _ = await runtime_for(
            tmp_path,
            [submit(expected.items[0].value), text_result("Done")],
        )
        started, release = asyncio.Event(), asyncio.Event()

        async def delay_discard():
            collector = runtime._completion.current()
            original = collector.discard

            async def discard():
                started.set()
                await release.wait()
                await original()

            collector.discard = discard

        client.on_write = delay_discard
        invocation = asyncio.create_task(runtime.invoke(stage_request()))
        await asyncio.wait_for(started.wait(), 2)
        invocation.cancel()
        await asyncio.sleep(0)
        assert not invocation.done() and runtime._invoke_lock.locked()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await invocation
        assert state.begins == state.completions == 1
        assert (await state.snapshot())["lastCompletedInvocation"]["phase"] == "cancelled"
        await runtime.abort(datetime.now(UTC) + timedelta(seconds=2))

    asyncio.run(scenario())
