"""Test-local exact-ID request projection on the accepted ADK; no production mode."""

from __future__ import annotations

import asyncio
import json
from contextlib import aclosing
from dataclasses import dataclass, field

import httpx
import pytest
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.events import Event
from google.adk.events.event_actions import EventActions, EventCompaction
from google.adk.models.llm_request import LlmRequest
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext
from google.genai import types
from test_adk_gateway_conformance import chat, gateway_model
from test_adk_runtime import assert_balanced_tool_calls
from test_audit_completion_continuation import CountedState

from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.llm.client import new_gateway_client
from contractor_runtime.llm.openai import GatewayModelError, OpenAICompatibleGatewayLlm
from contractor_runtime.worker.budget import _InvocationBudget
from contractor_runtime.worker.function_tools import ContractorFunctionTool
from contractor_runtime.worker.instrumentation import WorkerInstrumentationPlugin
from contractor_runtime.worker.runtime import _latest_observed_exact_refs

TASK = "Exact objective; parameter=value; input=source@revision; subtask=7"


class Rejected(RuntimeError):
    pass


def groups(events):
    """Validate full ordered FC/FR groups before choosing any immutable prefix."""
    result, active, pending, seen = [], [], set(), set()
    for event in events:
        if event.id in seen or event.partial or event.error_code or event.branch:
            raise Rejected("identity/partial/error/foreign")
        seen.add(event.id)
        if event.isolation_scope or event.actions.compaction:
            raise Rejected("foreign/native marker")
        calls = event.get_function_calls()
        replies = event.get_function_responses()
        if calls:
            if active or any(not call.id or call.id in pending for call in calls):
                raise Rejected("interleaved")
            pending = {call.id for call in calls}
            if len(pending) != len(calls):
                raise Rejected("duplicate call")
            active = [event]
        elif replies:
            if not active:
                raise Rejected("orphan")
            for reply in replies:
                if reply.id not in pending:
                    raise Rejected("foreign response")
                pending.remove(reply.id)
            active.append(event)
            if not pending:
                result.append(active)
                active = []
        elif event.content:
            if active:
                raise Rejected("split group")
            result.append([event])
    if active:
        raise Rejected("incomplete")
    return result


@dataclass
class Projection:
    generation: int = 0
    source_ids: tuple = ()
    memory: str = ""
    markers: set = field(default_factory=set)
    owner: tuple | None = None

    def select(self, events, keep=1):
        if not 1 <= keep <= 16:
            raise Rejected("tail bound")
        eligible = [e for e in events if e.author != "user" and e.id not in self.markers]
        complete = groups(eligible)
        if len(complete) <= keep:
            raise Rejected("ineffective")
        return tuple(e.id for group in complete[:-keep] for e in group)

    async def checkpoint(self, sessions, session, model, budget, *, fault="", stop=None):
        if stop and stop.is_set():
            raise asyncio.CancelledError
        identity = (session.app_name, session.user_id, session.id)
        if self.owner is not None and self.owner != identity:
            raise Rejected("foreign session")
        self.owner = identity
        selected = self.select(session.events)
        generation = self.generation
        budget.before_model_call()
        prompt_count = budget.latest_prompt_tokens
        response = None
        try:
            async for response in model.generate_content_async(
                LlmRequest(
                    contents=[
                        types.Content(
                            role="user",
                            parts=[
                                types.Part(
                                    text=(
                                        TASK
                                        + self.memory
                                        + json.dumps(
                                            [
                                                e.model_dump(mode="json")
                                                for e in session.events
                                                if e.id in selected and e.id not in self.source_ids
                                            ]
                                        )
                                    )
                                )
                            ],
                        )
                    ]
                )
            ):
                budget.after_model_response(response.usage_metadata)
        except GatewayModelError as error:
            if error.response_received:
                budget.after_model_response(error.usage_metadata)
            raise
        finally:
            budget.latest_prompt_tokens = prompt_count
        if stop and stop.is_set():
            raise asyncio.CancelledError
        if response is None or response.content is None:
            raise Rejected("empty")
        if response.finish_reason == types.FinishReason.MAX_TOKENS:
            raise Rejected("output limited")
        if response.content.parts and any(p.function_call for p in response.content.parts):
            raise Rejected("tool-bearing")
        try:
            value = json.loads("".join(p.text or "" for p in response.content.parts))
            if set(value) != {"summary"} or not isinstance(value["summary"], str):
                raise ValueError
            memory = value["summary"].strip()
        except (ValueError, TypeError):
            raise Rejected("schema") from None
        source_bytes = sum(
            len(e.model_dump_json().encode())
            for e in session.events
            if e.id in selected and e.id not in self.source_ids
        )
        if not memory or len(memory.encode()) > 16 * 1024 or len(memory.encode()) >= source_bytes:
            raise Rejected("invalid/ineffective")
        if generation != self.generation or selected != self.select(session.events):
            raise Rejected("stale")
        marker = Event(id=f"checkpoint-{generation + 1}", author="worker")
        if fault == "before":
            raise Rejected("definite append failure")
        await sessions.append_event(session=session, event=marker)
        if fault in {"lost", "unknown"}:
            if fault == "unknown":
                raise Rejected("ambiguous: inspection unavailable")
            stored = await sessions.get_session(
                app_name=session.app_name, user_id=session.user_id, session_id=session.id
            )
            matches = [e for e in stored.events if e.id == marker.id]
            if len(matches) != 1 or matches[0] != marker:
                raise Rejected("ambiguous")
        if stop and stop.is_set():
            raise asyncio.CancelledError
        self.source_ids, self.memory = selected, memory
        self.markers.add(marker.id)
        self.generation += 1

    def apply(self, events):
        # Public BeforeModel callback owns only this request, never raw session history.
        retained = [
            e
            for e in events
            if e.author != "user" and e.id not in self.markers and e.id not in self.source_ids
        ]
        groups(retained)
        contents = [types.Content(role="user", parts=[types.Part(text=TASK)])]
        if self.memory:
            contents.append(types.Content(role="model", parts=[types.Part(text=self.memory)]))
        contents.extend(e.content for e in retained if e.content)
        return contents


def test_real_runner_two_checkpoints_preserve_logical_owner_and_gateway_requests():
    async def scenario():
        payloads, tools, observed, sdk_ids = [], [], [], []
        artifact_refs = []
        main_calls = 0
        state = CountedState()
        budget = _InvocationBudget(7, 8, 10000, state.metrics)
        budget.start()
        projection = Projection()
        deadline = object()  # Same admission authority for every callback/turn.
        admissions = []

        async def gateway(request):
            nonlocal main_calls
            payload = json.loads(request.content)
            payloads.append(payload)
            if payload["model"] == "worker-summary-model":
                assert "tools" not in payload
                memory = f"rolling-memory-{projection.generation + 1}"
                return httpx.Response(200, json=chat({"content": json.dumps({"summary": memory})}))
            main_calls += 1
            if main_calls == 5:
                return httpx.Response(200, json=chat({"content": "finished"}))
            return httpx.Response(
                200,
                json=chat(
                    {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"call-{main_calls}-{i}",
                                "type": "function",
                                "function": {
                                    "name": "probe",
                                    "arguments": json.dumps({"name": f"{main_calls}-{i}"}),
                                },
                            }
                            for i in range(2)
                        ],
                    },
                    finish_reason="tool_calls",
                ),
            )

        def admission():
            admissions.append(deadline)

        async def probe(name: str, tool_context: ToolContext) -> dict:
            """Record one exact observation for the integration proof."""
            tools.append((name, tool_context.invocation_id))
            artifact_refs.append(ArtifactRef(namespace="proof", name=name, revision="exact"))
            probe.artifact_observation_cursor = len(artifact_refs)
            return {"observation": name, "data": "x" * 2048}

        probe.artifact_observation_cursor = 0

        def collect(_owner, cursor):
            observed.extend(artifact_refs[cursor:])

        plugin = WorkerInstrumentationPlugin(
            state=state,
            execution=state.resources.execution,
            budget=lambda: budget,
            instrumentation=None,
            model_alias="scripted",
            observe_artifacts=collect,
            allowed_tool_names=frozenset({"probe"}),
            check_admission=admission,
        )

        async def project(callback_context, llm_request):
            llm_request.contents = projection.apply(callback_context.session.events)

        sessions = InMemorySessionService()
        current = await sessions.create_session(app_name="proof", user_id="user")
        async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as http:
            model, compactor = gateway_model(http), gateway_model(http, summary=True)
            runner = Runner(
                app=App(
                    name="proof",
                    root_agent=LlmAgent(
                        name="worker",
                        model=model,
                        instruction="Keep the original task.",
                        before_model_callback=project,
                        tools=[
                            ContractorFunctionTool(
                                probe, logical_invocation_id=plugin.logical_invocation_id_for_tool
                            )
                        ],
                    ),
                    plugins=[plugin],
                ),
                session_service=sessions,
            )
            plugin.prepare_invocation(invocation_id="logical", subtask_id="7")
            try:
                for turn in range(3):
                    sdk_id = f"logical-turn-{turn}"
                    if turn:
                        plugin.prepare_continuation(
                            invocation_id="logical", sdk_invocation_id=sdk_id
                        )
                    else:
                        sdk_id = "logical"
                    sdk_ids.append(sdk_id)
                    count = 0
                    async with aclosing(
                        runner.run_async(
                            user_id="user",
                            session_id=current.id,
                            invocation_id=sdk_id,
                            new_message=types.Content(role="user", parts=[types.Part(text=TASK)]),
                        )
                    ) as events:
                        async for event in events:
                            if event.get_function_responses():
                                count += 1
                                if count == 2:
                                    break
                    current = await sessions.get_session(
                        app_name="proof", user_id="user", session_id=current.id
                    )
                    if turn < 2:
                        await projection.checkpoint(
                            sessions, current, compactor, budget, fault="lost" if turn == 0 else ""
                        )
                assert projection.generation == 2 and state.begins == 1
                assert tools == [(f"{n}-{i}", "logical") for n in range(1, 5) for i in range(2)]
                assert _latest_observed_exact_refs(tuple(observed)) == artifact_refs
                assert len(artifact_refs) == 8
                assert budget.model_calls == 7 and budget.tool_calls == 8
                assert budget.total_tokens == 7 * 18 and budget.latest_prompt_tokens == 11
                assert all(value is deadline for value in admissions)
                for payload in payloads:
                    assert_balanced_tool_calls(payload)
                last = json.dumps(payloads[-1])
                assert TASK in last and "rolling-memory-2" in last
                assert "rolling-memory-1" not in last and "call-1-0" not in last
                assert "call-4-0" in last and "call-4-1" in last
                assert len(set(sdk_ids)) == 3
                await plugin.complete_invocation(invocation_id="logical", phase="succeeded")
                assert state.completions == 1
            finally:
                await plugin.close()
                await runner.close()
                await model.close()
                await compactor.close()

    asyncio.run(scenario())


def source_event(index, *, timestamp=None):
    return Event(
        id=f"source-{index}",
        author="worker",
        timestamp=timestamp or index + 1,
        content=types.Content(role="model", parts=[types.Part(text=f"raw-{index}" + "x" * 1024)]),
    )


@pytest.mark.parametrize("case", ["equal-start", "equal-end", "out-of-order", "overlap", "foreign"])
def test_native_timestamp_range_cannot_replace_exact_source_identity(case):
    async def scenario():
        sessions = InMemorySessionService()
        current = await sessions.create_session(app_name="native", user_id="user")
        source = [source_event(i) for i in range(3)]
        if case == "equal-start":
            source[0].timestamp = source[2].timestamp = 1
        elif case in {"equal-end", "foreign"}:
            source[1].timestamp = source[2].timestamp = 2
        elif case == "out-of-order":
            source[0].timestamp, source[1].timestamp, source[2].timestamp = 3, 1, 2
        if case == "foreign":
            source[2].branch = "other"
        for event in source:
            await sessions.append_event(session=current, event=event)
        if case == "overlap":
            await sessions.append_event(
                session=current,
                event=Event(
                    author="worker",
                    actions=EventActions(
                        compaction=EventCompaction(
                            start_timestamp=1,
                            end_timestamp=2,
                            compacted_content=types.Content(
                                role="model", parts=[types.Part(text="old-memory")]
                            ),
                        )
                    ),
                ),
            )
        await sessions.append_event(
            session=current,
            event=Event(
                author="worker",
                actions=EventActions(
                    compaction=EventCompaction(
                        start_timestamp=min(e.timestamp for e in source[:2]),
                        end_timestamp=max(e.timestamp for e in source[:2]),
                        compacted_content=types.Content(
                            role="model", parts=[types.Part(text="new-memory")]
                        ),
                    )
                ),
            ),
        )
        payloads = []

        async def gateway(request):
            payloads.append(json.loads(request.content))
            return httpx.Response(200, json=chat({"content": "done"}))

        async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as http:
            model = gateway_model(http)
            runner = Runner(
                agent=LlmAgent(name="worker", model=model),
                app_name="native",
                session_service=sessions,
            )
            try:
                async with aclosing(
                    runner.run_async(
                        user_id="user",
                        session_id=current.id,
                        new_message=types.Content(role="user", parts=[types.Part(text=TASK)]),
                    )
                ) as events:
                    async for _ in events:
                        pass
            finally:
                await runner.close()
                await model.close()
        serialized = json.dumps(payloads[0])
        # Native inclusive ranges absorb the retained neighbor. Overlap selects newer memory.
        assert ("raw-2" in serialized) is (case == "overlap")
        assert "new-memory" in serialized and "old-memory" not in serialized
        if case != "foreign":
            projection = Projection()
            selected = projection.select(source)
            assert selected == (source[0].id, source[1].id)
            projection.source_ids, projection.memory = selected, "new-memory"
            assert "raw-2" in str(projection.apply(source))

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "case",
    [
        "partial",
        "error",
        "branch",
        "isolation",
        "orphan",
        "incomplete",
        "interleaved",
        "duplicate",
        "native",
        "tail16",
    ],
)
def test_exact_projection_rejects_unsafe_groups_before_mutation(case):
    events = [source_event(i) for i in range(17)]
    if case == "tail16":
        assert Projection().select(events, keep=16) == (events[0].id,)
        return
    first = events[0]
    if case == "partial":
        first.partial = True
    elif case == "error":
        first.error_code = "fixture"
    elif case == "branch":
        first.branch = "foreign"
    elif case == "isolation":
        first.isolation_scope = "foreign"
    elif case == "duplicate":
        events[1].id = first.id
    elif case == "native":
        first.actions.compaction = EventCompaction(
            start_timestamp=1,
            end_timestamp=2,
            compacted_content=types.Content(role="model", parts=[types.Part(text="old")]),
        )
    elif case == "orphan":
        first.content = types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id="missing", name="probe", response={}
                    )
                )
            ],
        )
    else:
        first.content = types.Content(
            role="model",
            parts=[
                types.Part(function_call=types.FunctionCall(id="pending", name="probe", args={}))
            ],
        )
        if case == "incomplete":
            events = [first]
    before = [e.model_dump_json() for e in events]
    with pytest.raises(Rejected):
        Projection().select(events)
    assert before == [e.model_dump_json() for e in events]


@pytest.mark.parametrize(
    "case",
    [
        "empty",
        "none",
        "oversized",
        "limited",
        "schema",
        "ineffective",
        "exception",
        "before",
        "unknown",
        "cancel-before",
        "cancel-append",
    ],
)
def test_rejected_checkpoint_fences_continuation_and_keeps_charges(case):
    async def scenario():
        stopped = asyncio.Event()

        class Sessions(InMemorySessionService):
            async def append_event(self, *, session, event):
                value = await super().append_event(session=session, event=event)
                if event.id.startswith("checkpoint") and case == "cancel-append":
                    stopped.set()
                return value

        sessions = Sessions()
        current = await sessions.create_session(app_name="proof", user_id="user")
        for i in range(3):
            await sessions.append_event(session=current, event=source_event(i))
        current = await sessions.get_session(
            app_name="proof", user_id="user", session_id=current.id
        )
        requests = []

        async def gateway(request):
            requests.append(json.loads(request.content))
            if case == "exception":
                raise RuntimeError("controlled summarizer failure")
            memory = "" if case == "empty" else "m"
            if case == "oversized":
                memory = "m" * (16 * 1024 + 1)
            if case == "ineffective":
                memory = "m" * 10000
            content = json.dumps({"summary": memory})
            if case == "none":
                content = None
            if case == "schema":
                content = json.dumps({"summary": memory, "extra": True})
            return httpx.Response(
                200,
                json=chat(
                    {"content": content}, finish_reason="length" if case == "limited" else "stop"
                ),
            )

        state, projection = CountedState(), Projection()
        budget = _InvocationBudget(4, 4, 10000, state.metrics)
        budget.start()
        if case == "cancel-before":
            stopped.set()
        async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as http:
            handle = new_gateway_client(
                base_url="https://fixture.example/v1",
                api_key="",
                timeout_seconds=10,
                http_client=http,
            )
            handle.max_retries = 0
            model = OpenAICompatibleGatewayLlm(model="worker-summary-model", client_handle=handle)
            try:
                with pytest.raises((Rejected, RuntimeError, asyncio.CancelledError)):
                    await projection.checkpoint(
                        sessions,
                        current,
                        model,
                        budget,
                        stop=stopped,
                        fault=case if case in {"before", "unknown"} else "",
                    )
            finally:
                await model.close()
        stored = await sessions.get_session(app_name="proof", user_id="user", session_id=current.id)
        markers = [e for e in stored.events if e.id.startswith("checkpoint")]
        assert len(markers) == (1 if case in {"unknown", "cancel-append"} else 0)
        assert projection.generation == 0 and not projection.memory
        assert len(requests) == (0 if case == "cancel-before" else 1)
        assert all(p["model"] == "worker-summary-model" for p in requests)
        assert budget.model_calls == len(requests) and budget.tool_calls == 0
        assert budget.total_tokens == (0 if case in {"exception", "cancel-before"} else 18)

    asyncio.run(scenario())


def test_postcommit_ineffective_main_response_stops_before_tool_dispatch():
    async def scenario():
        sessions = InMemorySessionService()
        current = await sessions.create_session(app_name="proof", user_id="user")
        for i in range(3):
            await sessions.append_event(session=current, event=source_event(i))
        state, projection, effects, requests = CountedState(), Projection(), [], []
        budget = _InvocationBudget(4, 4, 10000, state.metrics)
        budget.start()

        async def gateway(request):
            payload = json.loads(request.content)
            requests.append(payload)
            if payload["model"] == "worker-summary-model":
                return httpx.Response(200, json=chat({"content": '{"summary":"memory"}'}))
            response = chat(
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "late",
                            "type": "function",
                            "function": {"name": "probe", "arguments": "{}"},
                        }
                    ],
                },
                finish_reason="tool_calls",
            )
            response["usage"]["prompt_tokens"] = 900
            response["usage"]["total_tokens"] = 907
            return httpx.Response(200, json=response)

        async def probe() -> dict:
            """Forbidden late side effect."""
            effects.append(True)
            return {}

        async def before(callback_context, llm_request):
            llm_request.contents = projection.apply(callback_context.session.events)
            budget.before_model_call()

        async def after(callback_context, llm_response):
            budget.after_model_response(llm_response.usage_metadata)
            if budget.latest_prompt_tokens >= 750:
                raise Rejected("ineffective after commit, no new eligible progress")

        async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as http:
            main, compactor = gateway_model(http), gateway_model(http, summary=True)
            await projection.checkpoint(sessions, current, compactor, budget)
            runner = Runner(
                agent=LlmAgent(
                    name="worker",
                    model=main,
                    tools=[probe],
                    before_model_callback=before,
                    after_model_callback=after,
                ),
                app_name="proof",
                session_service=sessions,
            )
            try:
                with pytest.raises(Rejected, match="after commit"):
                    async with aclosing(
                        runner.run_async(
                            user_id="user",
                            session_id=current.id,
                            new_message=types.Content(role="user", parts=[types.Part(text=TASK)]),
                        )
                    ) as events:
                        async for _ in events:
                            pass
                assert projection.generation == 1 and projection.memory == "memory"
                assert budget.model_calls == 2 and budget.total_tokens == 925
                assert not effects and len(requests) == 2
            finally:
                await runner.close()
                await main.close()
                await compactor.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("transition", ["provider", "append", "continuation"])
def test_cancelled_owner_joins_transition_before_cleanup(transition):
    async def scenario():
        entered, release, joined = asyncio.Event(), asyncio.Event(), asyncio.Event()
        effects, requests = [], []

        class Sessions(InMemorySessionService):
            async def append_event(self, *, session, event):
                if transition == "append" and event.id.startswith("checkpoint"):
                    entered.set()
                    try:
                        await release.wait()
                    finally:
                        joined.set()
                return await super().append_event(session=session, event=event)

        sessions = Sessions()
        current = await sessions.create_session(app_name="proof", user_id="user")
        for i in range(3):
            await sessions.append_event(session=current, event=source_event(i))

        async def gateway(request):
            payload = json.loads(request.content)
            requests.append(payload)
            if transition == "provider" or payload["model"] == "worker-model":
                entered.set()
                try:
                    await release.wait()
                finally:
                    joined.set()
            return httpx.Response(200, json=chat({"content": '{"summary":"memory"}'}))

        async def probe() -> dict:
            """Observe any wrongly dispatched continuation tool."""
            effects.append(True)
            return {}

        state, projection = CountedState(), Projection()
        budget = _InvocationBudget(4, 4, 10000, state.metrics)
        budget.start()
        async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as http:
            compactor, main = gateway_model(http, summary=True), gateway_model(http)
            runner = Runner(
                agent=LlmAgent(name="worker", model=main, tools=[probe]),
                app_name="proof",
                session_service=sessions,
            )

            async def owner():
                await projection.checkpoint(sessions, current, compactor, budget)
                if transition == "continuation":
                    async with aclosing(
                        runner.run_async(
                            user_id="user",
                            session_id=current.id,
                            new_message=types.Content(role="user", parts=[types.Part(text=TASK)]),
                        )
                    ) as events:
                        async for _ in events:
                            pass

            task = asyncio.create_task(owner())
            try:
                await entered.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert joined.is_set() and task.done() and not effects
                assert projection.generation == (1 if transition == "continuation" else 0)
                assert len(requests) == (2 if transition == "continuation" else 1)
            finally:
                release.set()
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                await runner.close()
                await main.close()
                await compactor.close()

    asyncio.run(scenario())


def test_adk_event_id_is_not_append_idempotency():
    async def scenario():
        sessions = InMemorySessionService()
        current = await sessions.create_session(app_name="proof", user_id="user")
        marker = Event(id="same-marker", author="worker")
        await sessions.append_event(session=current, event=marker)
        await sessions.append_event(session=current, event=marker)
        divergent = marker.model_copy(
            update={"content": types.Content(role="model", parts=[types.Part(text="other")])}
        )
        await sessions.append_event(session=current, event=divergent)
        stored = await sessions.get_session(app_name="proof", user_id="user", session_id=current.id)
        assert len([e for e in stored.events if e.id == marker.id]) == 2

    asyncio.run(scenario())
