from __future__ import annotations

import asyncio
from typing import Any

import pytest
from google.adk.events import Event, EventActions
from google.adk.sessions import InMemorySessionService
from google.genai import types

from contractor_runtime.contracts import WorkerSessionMode
from contractor_runtime.session_lifecycle import (
    WorkerSessionLifecycle,
    WorkerSessionLifecycleError,
)

APP = "worker"
USER = "control-plane"


def test_isolated_sessions_are_lazy_fresh_and_carry_only_eligible_state() -> None:
    async def scenario() -> None:
        service = RecordingSessionService()
        lifecycle = WorkerSessionLifecycle(
            mode=WorkerSessionMode.ISOLATED,
            app_name=APP,
            user_id=USER,
            service=service,
        )

        assert (await service.list_sessions(app_name=APP, user_id=USER)).sessions == []

        first_id = await lifecycle.begin_invocation({"stateRevision": 1})
        first = await lifecycle.active_session()
        carried_value = {"nested": [1]}
        await service.append_event(
            first,
            Event(
                invocationId="invocation-1",
                author="worker",
                content=types.Content(role="model", parts=[types.Part(text="first conversation")]),
                actions=EventActions(
                    stateDelta={
                        "ordinary": carried_value,
                        "app:mode": "debug",
                        "user:locale": "en",
                        "temp:scratch": "discard me",
                        "contractor": {"stateRevision": 999},
                    }
                ),
            ),
        )
        # ADK normally keeps temp State only on the invocation-local Session
        # object. Seed the stored value as well so Contractor's own filter is
        # independently exercised instead of relying on ADK's implementation.
        service.sessions[APP][USER][first_id].state["temp:forced"] = "discard me too"
        stored_ordinary = service.sessions[APP][USER][first_id].state["ordinary"]

        await lifecycle.finish_invocation()
        assert (await service.list_sessions(app_name=APP, user_id=USER)).sessions == []

        assert isinstance(stored_ordinary, dict)
        stored_ordinary["nested"].append(2)
        second_id = await lifecycle.begin_invocation({"stateRevision": 2})
        second = await lifecycle.active_session()

        assert first_id != second_id
        assert first_id not in {"allocation-1", "invocation-1"}
        assert second.events == []
        assert second.state["ordinary"] == {"nested": [1]}
        assert second.state["app:mode"] == "debug"
        assert second.state["user:locale"] == "en"
        assert not any(key.startswith("temp:") for key in second.state)
        assert second.state["contractor"] == {"stateRevision": 2}

        await lifecycle.finish_invocation()
        await lifecycle.close()
        assert APP not in service.sessions
        assert APP not in service.app_state
        assert APP not in service.user_state

    asyncio.run(scenario())


def test_shared_session_retains_events_and_is_deleted_once_on_close() -> None:
    async def scenario() -> None:
        service = RecordingSessionService()
        lifecycle = WorkerSessionLifecycle(
            mode=WorkerSessionMode.SHARED,
            app_name=APP,
            user_id=USER,
            service=service,
        )

        first_id = await lifecycle.begin_invocation({"stateRevision": 1})
        first = await lifecycle.active_session()
        await service.append_event(
            first,
            Event(
                invocationId="invocation-1",
                author="worker",
                content=types.Content(role="model", parts=[types.Part(text="retained")]),
                actions=EventActions(stateDelta={"ordinary": "value"}),
            ),
        )
        await lifecycle.finish_invocation()

        second_id = await lifecycle.begin_invocation({"stateRevision": 2})
        second = await lifecycle.active_session()
        assert second_id == first_id
        assert [event.content.parts[0].text for event in second.events] == ["retained"]
        assert second.state["ordinary"] == "value"

        await lifecycle.finish_invocation()
        await lifecycle.close()
        assert service.created == [first_id]
        assert service.deleted == [first_id]

    asyncio.run(scenario())


def test_oversized_carried_state_fences_allocation_after_deleting_isolated_session() -> None:
    async def scenario() -> None:
        service = RecordingSessionService()
        lifecycle = WorkerSessionLifecycle(
            mode=WorkerSessionMode.ISOLATED,
            app_name=APP,
            user_id=USER,
            service=service,
            max_carried_state_bytes=64,
        )
        session_id = await lifecycle.begin_invocation({"stateRevision": 1})
        service.sessions[APP][USER][session_id].state["large"] = "x" * 128

        with pytest.raises(WorkerSessionLifecycleError, match="state_too_large"):
            await lifecycle.finish_invocation()
        assert lifecycle.failed
        assert (await service.list_sessions(app_name=APP, user_id=USER)).sessions == []
        with pytest.raises(WorkerSessionLifecycleError, match="fenced"):
            await lifecycle.begin_invocation({"stateRevision": 2})
        await lifecycle.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("failure", "expected"),
    [("create", "create_failed"), ("snapshot", "snapshot_failed"), ("delete", "delete_failed")],
)
def test_session_service_failures_fence_and_release_retries_cleanup(
    failure: str, expected: str
) -> None:
    async def scenario() -> None:
        service = RecordingSessionService(failure=failure)
        lifecycle = WorkerSessionLifecycle(
            mode=WorkerSessionMode.ISOLATED,
            app_name=APP,
            user_id=USER,
            service=service,
        )

        if failure == "create":
            with pytest.raises(WorkerSessionLifecycleError, match=expected):
                await lifecycle.begin_invocation({"stateRevision": 1})
        else:
            await lifecycle.begin_invocation({"stateRevision": 1})
            with pytest.raises(WorkerSessionLifecycleError, match=expected):
                await lifecycle.finish_invocation()

        assert lifecycle.failed
        with pytest.raises(WorkerSessionLifecycleError, match="fenced"):
            await lifecycle.begin_invocation({"stateRevision": 2})

        # Each injected fault is one-shot. Allocation release must retry any
        # identity whose create/delete outcome was ambiguous and empty the
        # allocation-local service before slot reuse.
        await lifecycle.close()
        assert APP not in service.sessions
        assert APP not in service.app_state
        assert APP not in service.user_state

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["create_cancel", "delete_cancel"])
def test_cancelled_mutation_fences_and_release_reconciles_ambiguous_session(
    operation: str,
) -> None:
    async def scenario() -> None:
        service = RecordingSessionService(failure=operation)
        lifecycle = WorkerSessionLifecycle(
            mode=WorkerSessionMode.ISOLATED,
            app_name=APP,
            user_id=USER,
            service=service,
        )

        if operation == "create_cancel":
            with pytest.raises(asyncio.CancelledError):
                await lifecycle.begin_invocation({"stateRevision": 1})
        else:
            await lifecycle.begin_invocation({"stateRevision": 1})
            with pytest.raises(asyncio.CancelledError):
                await lifecycle.finish_invocation()

        assert lifecycle.failed
        with pytest.raises(WorkerSessionLifecycleError, match="fenced"):
            await lifecycle.begin_invocation({"stateRevision": 2})
        await lifecycle.close()
        assert APP not in service.sessions

    asyncio.run(scenario())


class RecordingSessionService(InMemorySessionService):
    def __init__(self, *, failure: str | None = None) -> None:
        super().__init__()
        self.failure = failure
        self.created: list[str] = []
        self.deleted: list[str] = []
        self._failed = False

    async def create_session(self, **kwargs: Any):  # type: ignore[no-untyped-def]
        session = await super().create_session(**kwargs)
        self.created.append(session.id)
        if self.failure == "create_cancel" and not self._failed:
            self._failed = True
            raise asyncio.CancelledError
        if self.failure == "create" and not self._failed:
            self._failed = True
            raise RuntimeError("create failed after commit")
        return session

    async def get_session(self, **kwargs: Any):  # type: ignore[no-untyped-def]
        if self.failure == "snapshot" and not self._failed:
            self._failed = True
            raise RuntimeError("snapshot failed")
        return await super().get_session(**kwargs)

    async def delete_session(self, **kwargs: Any) -> None:
        session_id = kwargs["session_id"]
        self.deleted.append(session_id)
        if self.failure == "delete_cancel" and not self._failed:
            self._failed = True
            await super().delete_session(**kwargs)
            raise asyncio.CancelledError
        if self.failure == "delete" and not self._failed:
            self._failed = True
            raise RuntimeError("delete failed")
        await super().delete_session(**kwargs)
