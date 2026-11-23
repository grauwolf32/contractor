"""Real public ADK events and dispatch prove interrupted history remains truthful."""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fakes.model import ScriptedLlm, text_result, tool_call
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.events import Event
from google.adk.plugins import BasePlugin
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import Field

from contractor_runtime.contracts.base import json_byte_size
from contractor_runtime.llm.openai import _completion_request
from contractor_runtime.worker import history as history_module
from contractor_runtime.worker.history import ToolHistory, ToolHistoryError

APP = "history_proof"
USER = "user"
AGENT = "contractor_worker"
LOGICAL = "logical-worker"


def source(*ids: str, sdk_id: str = LOGICAL) -> Event:
    return Event(
        invocation_id=sdk_id,
        author=AGENT,
        content=types.Content(
            role="model",
            parts=[
                types.Part(function_call=types.FunctionCall(id=call_id, name="probe", args={}))
                for call_id in ids
            ],
        ),
    )


def context(session, call_id: str, sdk_id: str = LOGICAL):
    return SimpleNamespace(
        invocation_id=sdk_id,
        function_call_id=call_id,
        session=session,
        branch=None,
        isolation_scope=None,
    )


async def prepared(*ids: str, service=None, **bounds):
    service = service if service is not None else InMemorySessionService()
    session = await service.create_session(app_name=APP, user_id=USER)
    history = ToolHistory(APP, USER, **bounds)
    history.begin_turn(LOGICAL, LOGICAL, session.id)
    event = source(*ids)
    history.observe_event(event)
    await service.append_event(session, event)
    history.persisted(event)
    return history, service, session, event


async def stored_responses(service, session):
    actual = await service.get_session(app_name=APP, user_id=USER, session_id=session.id)
    return [response for event in actual.events for response in event.get_function_responses()]


def assert_code(error: pytest.ExceptionInfo[ToolHistoryError], code: str):
    assert error.value.code == code
    assert str(error.value) == f"Worker tool history failed ({code})"
    assert error.value.__cause__ is None


def test_seal_preserves_completed_result_and_distinguishes_unadmitted_cancelled_and_unknown():
    async def scenario():
        history, service, session, _ = await prepared(
            "completed", "unadmitted", "cancelled", "unknown"
        )
        result = {"ok": True, "evidence": "точный результат", "nested": [1, {"x": True}]}
        for call_id in ("completed", "cancelled", "unknown"):
            history.admitted(context(session, call_id))
        await history.completed(context(session, "completed"), result)
        result["nested"].append("later mutation")
        await history.failed(context(session, "cancelled"), None, cancelled=True)
        await history.failed(context(session, "unknown"), None)
        await history.seal(service, session)
        responses = {
            response.id: response.response for response in await stored_responses(service, session)
        }
        assert responses == {
            "completed": {"ok": True, "evidence": "точный результат", "nested": [1, {"x": True}]},
            "unadmitted": {"ok": False, "error": {"code": "tool_call_not_executed"}},
            "cancelled": {"ok": False, "error": {"code": "tool_call_cancelled"}},
            "unknown": {"ok": False, "error": {"code": "tool_call_outcome_unknown"}},
        }
        assert not history.needs_seal
        await history.seal(service, session)
        assert len(await stored_responses(service, session)) == 4
        history.begin_turn(LOGICAL, LOGICAL + ":turn:1", session.id)

    asyncio.run(scenario())


def test_native_response_acknowledgement_releases_snapshot_and_allows_reused_call_id():
    async def scenario():
        history, service, session, _ = await prepared(
            "same-id", max_calls=2, max_retained_bytes=160
        )
        for ordinal in range(2):
            if ordinal:
                event = source("same-id")
                history.observe_event(event)
                await service.append_event(session, event)
                history.persisted(event)
            history.admitted(context(session, "same-id"))
            await history.completed(context(session, "same-id"), {"value": "x" * 50})
            event = Event(
                invocation_id=LOGICAL,
                author=AGENT,
                content=types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                id="same-id", name="probe", response={"value": "x" * 50}
                            )
                        )
                    ],
                ),
            )
            history.observe_event(event)
            assert history.needs_seal  # observing is not an append acknowledgement
            await service.append_event(session, event)
            history.persisted(event)
            assert not history.needs_seal
        assert len(await stored_responses(service, session)) == 2

    asyncio.run(scenario())


def test_actual_stored_response_wins_when_its_append_acknowledgement_was_not_delivered():
    async def scenario():
        history, service, session, _ = await prepared("call")
        history.admitted(context(session, "call"))
        await history.completed(context(session, "call"), {"callback": "known result"})
        event = Event(
            invocation_id=LOGICAL,
            author=AGENT,
            content=types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            id="call", name="probe", response={"actual": "final native result"}
                        )
                    )
                ],
            ),
        )
        history.observe_event(event)
        await service.append_event(session, event)
        await history.seal(service, session)
        responses = await stored_responses(service, session)
        assert len(responses) == 1
        assert responses[0].response == {"actual": "final native result"}
        assert not history.needs_seal

    asyncio.run(scenario())


def test_seal_preserves_native_scope_and_wraps_non_mapping_results():
    async def scenario():
        service = InMemorySessionService()
        session = await service.create_session(app_name=APP, user_id=USER)
        history = ToolHistory(APP, USER)
        history.begin_turn(LOGICAL, LOGICAL, session.id)
        event = source("array")
        event.branch = "worker.branch"
        event.isolation_scope = "scope"
        event.node_info.path = "worker/1"
        history.observe_event(event)
        await service.append_event(session, event)
        history.persisted(event)
        tool_context = context(session, "array")
        tool_context.branch = event.branch
        tool_context.isolation_scope = event.isolation_scope
        history.admitted(tool_context)
        await history.completed(tool_context, ["точно", {"x": 2}])
        await history.seal(service, session)
        actual = await service.get_session(app_name=APP, user_id=USER, session_id=session.id)
        sealed = actual.events[-1]
        assert (sealed.branch, sealed.isolation_scope, sealed.node_info.path) == (
            event.branch,
            event.isolation_scope,
            event.node_info.path,
        )
        assert sealed.get_function_responses()[0].response == {"result": ["точно", {"x": 2}]}

    asyncio.run(scenario())


def test_partial_calls_do_not_create_a_stored_history_obligation():
    async def scenario():
        service = InMemorySessionService()
        session = await service.create_session(app_name=APP, user_id=USER)
        history = ToolHistory(APP, USER)
        history.begin_turn(LOGICAL, LOGICAL, session.id)
        event = source("fragment")
        event.partial = True
        history.observe_event(event)
        history.persisted(event)
        await history.seal(service, session)
        assert not history.needs_seal
        assert not await stored_responses(service, session)

    asyncio.run(scenario())


def test_acknowledged_source_missing_from_bounded_suffix_fences_instead_of_claiming_nonexecution():
    async def scenario():
        history, service, session, _ = await prepared("call")
        for _ in range(6):
            await service.append_event(
                session,
                Event(
                    invocation_id=LOGICAL,
                    author=AGENT,
                    content=types.Content(role="model", parts=[types.Part(text="later event")]),
                ),
            )
        with pytest.raises(ToolHistoryError) as error:
            await history.seal(service, session)
        assert_code(error, "tool_history_unavailable")
        assert history.needs_seal
        assert not await stored_responses(service, session)

    asyncio.run(scenario())


@pytest.mark.parametrize("committed", [False, True])
def test_unacknowledged_selecting_event_is_reconciled_without_inventing_execution(committed):
    async def scenario():
        service = InMemorySessionService()
        session = await service.create_session(app_name=APP, user_id=USER)
        history = ToolHistory(APP, USER)
        history.begin_turn(LOGICAL, LOGICAL, session.id)
        event = source("unacknowledged")
        history.observe_event(event)
        if committed:
            await service.append_event(session, event)
        await history.seal(service, session)
        responses = await stored_responses(service, session)
        assert len(responses) == int(committed)
        if responses:
            assert responses[0].response["error"]["code"] == "tool_call_not_executed"
        assert not history.needs_seal

    asyncio.run(scenario())


class FailingService(InMemorySessionService):
    def __init__(self):
        super().__init__()
        self.failure: str | None = None
        self.appended_ids = []
        self.read_limits = []
        self.committed = asyncio.Event()

    async def get_session(self, **kwargs):
        config = kwargs.get("config")
        if config is not None:
            self.read_limits.append(config.num_recent_events)
        return await super().get_session(**kwargs)

    async def append_event(self, session, event):
        if event.id.startswith("worker-history-"):
            self.appended_ids.append(event.id)
            failure, self.failure = self.failure, None
            if failure == "before":
                raise RuntimeError("private-transport-value")
            result = await super().append_event(session, event)
            if failure == "after":
                raise RuntimeError("private-transport-value")
            if failure == "cancel-after":
                self.committed.set()
                await asyncio.Event().wait()
            return result
        return await super().append_event(session, event)


def test_lost_append_ack_is_reconciled_once_with_a_bounded_public_read():
    async def scenario():
        service = FailingService()
        history, _, session, _ = await prepared("call", service=service)
        service.failure = "after"
        await history.seal(service, session)
        assert not history.needs_seal
        assert len(await stored_responses(service, session)) == 1
        assert len(service.appended_ids) == 1
        assert service.read_limits == [5, 5]

    asyncio.run(scenario())


def test_failed_append_latches_failure_and_forbids_blind_retry():
    async def scenario():
        service = FailingService()
        history, _, session, _ = await prepared("call", service=service)
        service.failure = "before"
        with pytest.raises(ToolHistoryError) as error:
            await history.seal(service, session)
        assert_code(error, "tool_history_unavailable")
        assert "private-transport-value" not in repr(error.value)
        assert history.needs_seal
        with pytest.raises(ToolHistoryError) as error:
            history.check()
        assert_code(error, "tool_history_unavailable")
        with pytest.raises(ToolHistoryError):
            await history.seal(service, session)
        assert len(service.appended_ids) == 1
        assert not await stored_responses(service, session)
        history.clear()  # Only after the owner records failure and fences the session.

    asyncio.run(scenario())


def test_cancel_after_seal_commit_preserves_cancellation_and_reconciles_without_duplicate():
    async def scenario():
        service = FailingService()
        history, _, session, _ = await prepared("call", service=service)
        service.failure = "cancel-after"
        task = asyncio.create_task(history.seal(service, session))
        await service.committed.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert history.needs_seal
        await history.seal(service, session)
        history.begin_turn(LOGICAL, LOGICAL + ":turn:1", session.id)
        assert len(service.appended_ids) == 1
        assert len(await stored_responses(service, session)) == 1

    asyncio.run(scenario())


def test_completed_snapshot_joins_serialization_and_retains_known_result_on_cancel(monkeypatch):
    original = history_module._encode_bounded
    entered, release = threading.Event(), threading.Event()

    def blocked_encode(value, maximum):
        entered.set()
        release.wait()
        return original(value, maximum)

    monkeypatch.setattr(history_module, "_encode_bounded", blocked_encode)

    async def scenario():
        history, service, session, _ = await prepared("completed")
        tool_context = context(session, "completed")
        history.admitted(tool_context)
        task = asyncio.create_task(history.completed(tool_context, {"exact": "already completed"}))
        await asyncio.to_thread(entered.wait)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await history.failed(tool_context, None, cancelled=True)
        await history.seal(service, session)
        assert (await stored_responses(service, session))[0].response == {
            "exact": "already completed"
        }

    asyncio.run(scenario())


@pytest.mark.parametrize("invalid_waiter_result", [False, True])
def test_returned_result_joins_snapshot_lock_wait_even_under_repeated_cancel(
    monkeypatch, invalid_waiter_result
):
    original = history_module._encode_bounded
    entered, release = threading.Event(), threading.Event()

    def blocked_first_encode(value, maximum):
        if value == {"exact": "first"}:
            entered.set()
            release.wait()
        return original(value, maximum)

    monkeypatch.setattr(history_module, "_encode_bounded", blocked_first_encode)

    async def scenario():
        history, service, session, _ = await prepared("first", "returned-waiter")
        waiting = asyncio.Event()

        class ObservedLock(asyncio.Lock):
            async def acquire(self):
                if self.locked():
                    waiting.set()
                return await super().acquire()

        history._snapshot_lock = ObservedLock()
        for call_id in ("first", "returned-waiter"):
            history.admitted(context(session, call_id))
        first = asyncio.create_task(
            history.completed(context(session, "first"), {"exact": "first"})
        )
        await asyncio.to_thread(entered.wait)
        result = (
            {"unsupported": object()} if invalid_waiter_result else {"exact": "already returned"}
        )
        waiter = asyncio.create_task(history.completed(context(session, "returned-waiter"), result))
        await waiting.wait()
        try:
            waiter.cancel()
            await asyncio.sleep(0)
            waiter.cancel()
            await asyncio.sleep(0)
            assert not waiter.done()
        finally:
            release.set()
            await first
            with pytest.raises(asyncio.CancelledError):
                await waiter
        assert not history._snapshot_lock.locked()
        if invalid_waiter_result:
            # Snapshot failure still fences the owner, while the caller retains
            # its original cancellation rather than an encoding exception.
            with pytest.raises(ToolHistoryError) as error:
                history.check()
            assert_code(error, "invalid_tool_history")
            with pytest.raises(ToolHistoryError) as error:
                await history.seal(service, session)
            assert_code(error, "invalid_tool_history")
        else:
            await history.seal(service, session)
            assert {
                response.id: response.response
                for response in await stored_responses(service, session)
            } == {
                "first": {"exact": "first"},
                "returned-waiter": {"exact": "already returned"},
            }

    asyncio.run(scenario())


@pytest.mark.parametrize("bad", [float("nan"), object(), {"bad": {1: "not-string-key"}}])
def test_non_json_results_fail_content_free_and_leave_history_fenced(bad):
    async def scenario():
        history, _, session, _ = await prepared("call")
        with pytest.raises(ToolHistoryError) as error:
            await history.completed(context(session, "call"), bad)
        assert_code(error, "invalid_tool_history")
        assert history.needs_seal

    asyncio.run(scenario())


def test_multibyte_and_escaped_result_overflow_latches_and_keeps_known_siblings():
    async def scenario():
        history, service, session, _ = await prepared("small", "large", max_retained_bytes=256)
        await history.completed(context(session, "small"), {"safe": "first"})
        with pytest.raises(ToolHistoryError) as error:
            await history.completed(context(session, "large"), {"payload": "\0я" * 200})
        assert_code(error, "tool_history_limit_exceeded")
        assert history.needs_seal
        # The SDK may catch the callback exception. The failure must still stop
        # publication, next model admission and a successful candidate.
        with pytest.raises(ToolHistoryError) as error:
            history.check()
        assert_code(error, "tool_history_limit_exceeded")
        with pytest.raises(ToolHistoryError):
            await history.failed(context(session, "large"), {"error": "safe SDK response"})
        with pytest.raises(ToolHistoryError):
            await history.seal(service, session)
        assert json.loads(history._calls["small"].result) == {"safe": "first"}
        assert not await stored_responses(service, session)

    asyncio.run(scenario())


def test_shared_history_preflight_uses_the_private_wire_json_byte_measure():
    fixture = (
        Path(__file__).resolve().parents[2] / "api/testdata/v1alpha1/json-byte-size-cases.json"
    )
    for case in json.loads(fixture.read_text())["cases"]:
        encoded = history_module._encode_bounded(case["value"], case["bytes"])
        assert len(encoded) == json_byte_size(case["value"]) == case["bytes"], case["name"]
        assert json.loads(encoded) == case["value"]
        with pytest.raises(ToolHistoryError) as error:
            history_module._encode_bounded(case["value"], case["bytes"] - 1)
        assert_code(error, "tool_history_limit_exceeded")


def test_observed_call_cap_applies_across_turns_but_new_logical_invocation_resets_it():
    async def scenario():
        history, service, session, _ = await prepared("one", max_calls=1)
        await history.seal(service, session)
        history.begin_turn(LOGICAL, LOGICAL + ":turn:1", session.id)
        with pytest.raises(ToolHistoryError) as error:
            history.observe_event(source("two", sdk_id=LOGICAL + ":turn:1"))
        assert_code(error, "tool_history_limit_exceeded")
        with pytest.raises(ToolHistoryError):
            history.begin_turn("next-logical", "next-logical", session.id)
        history.clear()  # The owner has fenced and is retiring the failed session.
        history.begin_turn("next-logical", "next-logical", session.id)
        history.observe_event(source("two", sdk_id="next-logical"))

    asyncio.run(scenario())


@pytest.mark.parametrize("bad", ["turn", "session", "branch", "isolation"])
def test_scope_rejection_latches_and_prevents_active_history_mutation(bad):
    async def scenario():
        history, service, session, _ = await prepared("call")
        bad_context = context(session, "call")
        if bad == "turn":
            bad_context.invocation_id = "foreign-turn"
        elif bad == "session":
            bad_context.session = session.model_copy(update={"id": "foreign-session"})
        elif bad == "branch":
            bad_context.branch = "foreign"
        else:
            bad_context.isolation_scope = "foreign"
        with pytest.raises(ToolHistoryError) as error:
            history.admitted(bad_context)
        assert_code(error, "invalid_tool_history")
        with pytest.raises(ToolHistoryError):
            await history.completed(context(session, "call"), {"would": "change history"})
        with pytest.raises(ToolHistoryError):
            await history.seal(service, session)
        assert not await stored_responses(service, session)
        history.clear()
        assert not history.needs_seal

    asyncio.run(scenario())


def test_stale_sdk_event_cannot_reenter_a_settled_new_turn():
    async def scenario():
        history, service, session, event = await prepared("call")
        await history.seal(service, session)
        history.begin_turn(LOGICAL, LOGICAL + ":turn:1", session.id)
        with pytest.raises(ToolHistoryError) as error:
            history.persisted(event)
        assert_code(error, "invalid_tool_history")
        with pytest.raises(ToolHistoryError):
            history.check()

    asyncio.run(scenario())


class CapturingLlm(ScriptedLlm):
    payloads: list[dict[str, Any]] = Field(default_factory=list, exclude=True)

    async def generate_content_async(self, llm_request, stream=False):
        self.payloads.append(_completion_request(self.model, llm_request))
        async for response in super().generate_content_async(llm_request, stream=stream):
            yield response


class HistoryPlugin(BasePlugin):
    def __init__(self, history, *, rejected):
        super().__init__(name="history_proof")
        self.history = history
        self.rejected = rejected
        self.admissions = 0
        self.order = []

    async def before_run_callback(self, *, invocation_context):
        self.history.begin_turn(
            LOGICAL, invocation_context.invocation_id, invocation_context.session.id
        )

    async def on_event_callback(self, *, invocation_context, event):
        self.history.observe_event(event)
        if event.get_function_calls():
            self.order.append("select")

    async def before_tool_callback(self, *, tool, tool_args, tool_context):
        self.order.append("admit:" + tool_args["tag"])
        if self.rejected and self.admissions:
            raise RuntimeError("proof budget rejected")
        self.history.admitted(tool_context)
        self.admissions += 1

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result):
        await self.history.completed(tool_context, result)

    async def on_tool_error_callback(self, *, tool, tool_args, tool_context, error):
        await self.history.failed(
            tool_context, None, cancelled=isinstance(error, asyncio.CancelledError)
        )
        if not isinstance(error, asyncio.CancelledError):
            raise error


def test_real_runner_cannot_continue_after_a_caught_snapshot_failure():
    class CatchingPlugin(HistoryPlugin):
        caught = False

        async def after_tool_callback(self, **kwargs):
            try:
                await super().after_tool_callback(**kwargs)
            except ToolHistoryError:
                # Simulate a framework error handler returning the tool result.
                # Later persistence and model admission must still be fenced.
                self.caught = True

    async def scenario():
        effects = []

        async def probe(tag: str) -> dict:
            """Produce a real result larger than the controlled pending-history cap."""
            effects.append(tag)
            return {"payload": "x" * 1000}

        model = CapturingLlm(
            model="scripted",
            responses=[
                tool_call("probe", {"tag": "once"}, call_id="call"),
                text_result("false completion"),
            ],
            auto_result_finalizer=False,
        )
        service = InMemorySessionService()
        session = await service.create_session(app_name=APP, user_id=USER)
        history = ToolHistory(APP, USER, max_retained_bytes=512)
        plugin = CatchingPlugin(history, rejected=False)
        runner = Runner(
            app=App(
                name=APP,
                root_agent=LlmAgent(name=AGENT, model=model, tools=[probe]),
                plugins=[plugin],
            ),
            session_service=service,
        )
        try:
            with pytest.raises(RuntimeError, match="tool_history_limit_exceeded"):
                async with contextlib.aclosing(
                    runner.run_async(
                        user_id=USER,
                        session_id=session.id,
                        invocation_id=LOGICAL,
                        new_message=types.Content(role="user", parts=[types.Part(text="task")]),
                    )
                ) as events:
                    async for event in events:
                        history.persisted(event)
            assert plugin.caught
            assert effects == ["once"]
            assert len(model.payloads) == 1
            with pytest.raises(ToolHistoryError) as error:
                history.check()
            assert_code(error, "tool_history_limit_exceeded")
        finally:
            await runner.close()
            history.clear()

    asyncio.run(scenario())


@pytest.mark.parametrize("cancelled", [False, True])
def test_real_runner_next_request_keeps_completed_sibling_and_truthful_interrupted_call(cancelled):
    async def scenario():
        entered = asyncio.Event()
        effects = []

        async def probe(tag: str) -> dict:
            """Return one exact proof result, or wait for controlled cancellation."""
            effects.append(tag)
            if tag == "waiting":
                entered.set()
                await asyncio.Event().wait()
            return {"ok": True, "exact": tag, "ordinal": len(effects)}

        response = tool_call("probe", {"tag": "completed"}, call_id="completed-id")
        response.content.parts.extend(
            tool_call(
                "probe", {"tag": "waiting" if cancelled else "unadmitted"}, call_id="interrupted-id"
            ).content.parts
        )
        model = CapturingLlm(
            model="scripted",
            responses=[response, text_result("continued")],
            auto_result_finalizer=False,
        )
        service = InMemorySessionService()
        session = await service.create_session(app_name=APP, user_id=USER)
        history = ToolHistory(APP, USER)
        plugin = HistoryPlugin(history, rejected=not cancelled)
        runner = Runner(
            app=App(
                name=APP,
                root_agent=LlmAgent(name=AGENT, model=model, tools=[probe]),
                plugins=[plugin],
            ),
            session_service=service,
        )

        async def drive(sdk_id, prompt):
            async with contextlib.aclosing(
                runner.run_async(
                    user_id=USER,
                    session_id=session.id,
                    invocation_id=sdk_id,
                    new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
                )
            ) as events:
                async for event in events:
                    history.persisted(event)

        try:
            if cancelled:
                task = asyncio.create_task(drive(LOGICAL, "original task"))
                await entered.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                with pytest.raises(RuntimeError, match="proof budget rejected"):
                    await drive(LOGICAL, "original task")
            await history.seal(service, session)
            await drive(LOGICAL + ":turn:1", "Runtime reminder: finish remaining work")
            payload = model.payloads[-1]
            calls = [
                call["id"]
                for message in payload["messages"]
                for call in message.get("tool_calls", [])
            ]
            results = {
                message["tool_call_id"]: json.loads(message["content"])
                for message in payload["messages"]
                if message["role"] == "tool"
            }
            assert calls == ["completed-id", "interrupted-id"]
            assert results["completed-id"] == {"ok": True, "exact": "completed", "ordinal": 1}
            assert results["interrupted-id"]["error"]["code"] == (
                "tool_call_cancelled" if cancelled else "tool_call_not_executed"
            )
            assert effects == (["completed", "waiting"] if cancelled else ["completed"])
            assert plugin.order[0] == "select"
            assert len(await stored_responses(service, session)) == 2
            assert "Runtime reminder" in model.requests[-1]["contentText"]
            assert not history.needs_seal
        finally:
            await runner.close()
            history.clear()

    asyncio.run(scenario())
