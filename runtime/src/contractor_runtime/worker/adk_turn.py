"""ADK turn execution and joined history settlement."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator, Callable

from google.adk.agents.run_config import RunConfig
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions.session import Session
from google.genai import types

from contractor_runtime.core.async_tasks import join_owned_task
from contractor_runtime.worker.compaction import (
    CompactionCoordinator,
)
from contractor_runtime.worker.failures import _chained_error
from contractor_runtime.worker.history import ToolHistory, ToolHistoryError
from contractor_runtime.worker.invocation import INVOCATION_CLEANUP_TIMEOUT_SECONDS
from contractor_runtime.worker.sessions import (
    WorkerSessionLifecycle,
    WorkerSessionLifecycleError,
)


class AdkTurnRunner:
    """Run one SDK turn and settle its tool history before returning control."""

    def __init__(
        self,
        *,
        runner: Runner,
        sessions: WorkerSessionLifecycle,
        history: ToolHistory,
        app_name: str,
        user_id: str,
        max_model_calls: int,
        compaction: CompactionCoordinator | None,
        observe_cancellation: Callable[[], None],
        fence: Callable[[], None],
    ) -> None:
        self._runner = runner
        self._session_lifecycle = sessions
        self._session_service = sessions.service
        self._tool_history = history
        self._app_name = app_name
        self._user_id = user_id
        self._max_model_calls = max_model_calls
        self._compaction = compaction
        self._raise_observed_tool_cancellation = observe_cancellation
        self._fence = fence

    async def events(
        self, logical_id: str, sdk_id: str, session_id: str, prompt: str
    ) -> AsyncGenerator[Event]:
        cancelled = False
        try:
            self._tool_history.begin_turn(logical_id, sdk_id, session_id)
            async with contextlib.aclosing(
                self._runner.run_async(
                    user_id=self._user_id,
                    session_id=session_id,
                    invocation_id=sdk_id,
                    run_config=RunConfig(max_llm_calls=self._max_model_calls),
                    new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
                )
            ) as events:
                async for event in events:
                    # Runner yields only after SessionService accepted this event.
                    self._tool_history.persisted(event)
                    if self._compaction is not None:
                        await self._compaction.observe(event)
                    yield event
            # Some SDK node runners absorb a child's CancelledError and end
            # normally. Observe it before selecting a result or entering cleanup.
            self._raise_observed_tool_cancellation()
        except asyncio.CancelledError:
            cancelled = True
            raise
        except Exception as error:
            if _chained_error(error, ToolHistoryError) is None:
                try:
                    self._raise_observed_tool_cancellation()
                except asyncio.CancelledError:
                    cancelled = True
                    raise
            raise
        finally:
            try:
                await self.seal()
            except (Exception, asyncio.CancelledError):
                if not cancelled:
                    raise

    async def seal(self) -> None:
        async def seal() -> None:
            self._tool_history.check()
            if self._tool_history.needs_seal:
                session_id = self._session_lifecycle.active_session_id
                if session_id is None:
                    raise WorkerSessionLifecycleError("session_absent")
                # Supply the owner's identity only. The ledger obtains its
                # bounded event suffix through the public SessionService itself.
                session = Session(app_name=self._app_name, user_id=self._user_id, id=session_id)
                await self._tool_history.seal(self._session_service, session)

        async def bounded_seal() -> None:
            try:
                await asyncio.wait_for(seal(), timeout=INVOCATION_CLEANUP_TIMEOUT_SECONDS)
            except ToolHistoryError as error:
                self._fence()
                raise WorkerSessionLifecycleError(error.code) from None
            except (Exception, asyncio.CancelledError):
                self._fence()
                raise WorkerSessionLifecycleError("tool_history_unavailable") from None

        task = asyncio.create_task(bounded_seal(), name="worker-tool-history-seal")
        if await join_owned_task(task):
            with contextlib.suppress(Exception, asyncio.CancelledError):
                task.result()
            raise asyncio.CancelledError
        task.result()
