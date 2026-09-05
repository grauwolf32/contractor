"""Allocation-local ADK Worker session ownership."""

from __future__ import annotations

import asyncio
import copy
import json
import uuid
from typing import Any

from google.adk.sessions import BaseSessionService, InMemorySessionService
from google.adk.sessions.session import Session
from google.adk.sessions.state import State

from contractor_runtime.contracts import WorkerSessionMode

MAX_CARRIED_STATE_BYTES = 4 * 1024 * 1024


class WorkerSessionLifecycleError(RuntimeError):
    """A safe allocation-local session lifecycle failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class WorkerSessionLifecycle:
    """Owns lazy shared/isolated sessions for exactly one allocation."""

    def __init__(
        self,
        *,
        mode: WorkerSessionMode,
        app_name: str,
        user_id: str,
        service: BaseSessionService | None = None,
        max_carried_state_bytes: int = MAX_CARRIED_STATE_BYTES,
    ) -> None:
        if not isinstance(mode, WorkerSessionMode):
            raise ValueError("Worker session mode must be isolated or shared")
        if max_carried_state_bytes <= 0:
            raise ValueError("max carried State bytes must be positive")
        self.mode = mode
        self.app_name = app_name
        self.user_id = user_id
        self.service = service if service is not None else InMemorySessionService()
        self._max_carried_state_bytes = max_carried_state_bytes
        self._carried_state: dict[str, Any] = {}
        self._active_session_id: str | None = None
        self._shared_session_id: str | None = None
        self._owned_session_ids: set[str] = set()
        self._failed = False
        self._closed = False

    @property
    def active_session_id(self) -> str | None:
        return self._active_session_id

    @property
    def shared_session_id(self) -> str | None:
        return self._shared_session_id

    @property
    def failed(self) -> bool:
        return self._failed

    async def begin_invocation(self, contractor_state: dict[str, Any]) -> str:
        if self._closed:
            raise WorkerSessionLifecycleError("closed")
        if self._failed:
            raise WorkerSessionLifecycleError("fenced")
        if self._active_session_id is not None:
            self._failed = True
            raise WorkerSessionLifecycleError("ownership_conflict")

        if self.mode == WorkerSessionMode.SHARED and self._shared_session_id is not None:
            session = await self._get(self._shared_session_id)
            self._active_session_id = session.id
            return session.id

        session_id = f"adk-session-{uuid.uuid4().hex}"
        initial_state = self._initial_state(contractor_state)
        # Creation is allowed to fail after committing in a remote/custom
        # SessionService. Remember the requested identity before the call so
        # allocation cleanup can always make an idempotent deletion attempt.
        self._owned_session_ids.add(session_id)
        try:
            session = await self.service.create_session(
                app_name=self.app_name,
                user_id=self.user_id,
                session_id=session_id,
                state=initial_state,
            )
        except asyncio.CancelledError:
            self._failed = True
            raise
        except Exception as error:
            self._failed = True
            raise WorkerSessionLifecycleError("create_failed") from error
        if session.id != session_id:
            self._failed = True
            if session.id:
                self._owned_session_ids.add(session.id)
            raise WorkerSessionLifecycleError("identity_mismatch")
        self._active_session_id = session_id
        if self.mode == WorkerSessionMode.SHARED:
            self._shared_session_id = session_id
        return session_id

    async def active_session(self) -> Session:
        session_id = self._active_session_id
        if session_id is None:
            raise WorkerSessionLifecycleError("session_absent")
        return await self._get(session_id)

    async def finish_invocation(self) -> None:
        session_id = self._active_session_id
        if session_id is None:
            self._failed = True
            raise WorkerSessionLifecycleError("session_absent")

        errors: list[WorkerSessionLifecycleError] = []
        try:
            session = await self._get(session_id)
            eligible_state = self._eligible_state(session.state)
            if self.mode == WorkerSessionMode.ISOLATED:
                self._carried_state = eligible_state
        except WorkerSessionLifecycleError as error:
            errors.append(error)

        if self.mode == WorkerSessionMode.ISOLATED:
            try:
                await self._delete(session_id)
            except WorkerSessionLifecycleError as error:
                errors.append(error)
            else:
                self._active_session_id = None
                self._owned_session_ids.discard(session_id)
        else:
            self._active_session_id = None

        if errors:
            self._failed = True
            raise errors[0]

    async def close(self) -> None:
        self._closed = True
        errors: list[WorkerSessionLifecycleError] = []
        for session_id in sorted(self._owned_session_ids):
            try:
                await self._delete(session_id)
            except WorkerSessionLifecycleError as error:
                errors.append(error)
            else:
                self._owned_session_ids.discard(session_id)
        if errors:
            self._failed = True
            raise errors[0]
        self._active_session_id = None
        self._shared_session_id = None
        self._carried_state.clear()
        self._clear_service_state()

    async def _get(self, session_id: str) -> Session:
        try:
            session = await self.service.get_session(
                app_name=self.app_name,
                user_id=self.user_id,
                session_id=session_id,
            )
        except asyncio.CancelledError:
            self._failed = True
            raise
        except Exception as error:
            self._failed = True
            raise WorkerSessionLifecycleError("snapshot_failed") from error
        if session is None:
            self._failed = True
            raise WorkerSessionLifecycleError("session_missing")
        return session

    async def _delete(self, session_id: str) -> None:
        try:
            await self.service.delete_session(
                app_name=self.app_name,
                user_id=self.user_id,
                session_id=session_id,
            )
        except asyncio.CancelledError:
            self._failed = True
            raise
        except Exception as error:
            raise WorkerSessionLifecycleError("delete_failed") from error

    def _initial_state(self, contractor_state: dict[str, Any]) -> dict[str, Any]:
        try:
            initial = copy.deepcopy(self._carried_state)
            initial["contractor"] = copy.deepcopy(contractor_state)
        except Exception as error:
            self._failed = True
            raise WorkerSessionLifecycleError("state_copy_failed") from error
        return initial

    def _eligible_state(self, state: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(state, dict):
            raise WorkerSessionLifecycleError("state_invalid")
        if any(not isinstance(key, str) for key in state):
            raise WorkerSessionLifecycleError("state_invalid")
        try:
            eligible = copy.deepcopy(
                {
                    key: value
                    for key, value in state.items()
                    if isinstance(key, str)
                    and key != "contractor"
                    and not key.startswith(State.TEMP_PREFIX)
                }
            )
            encoded = json.dumps(
                eligible,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        except WorkerSessionLifecycleError:
            raise
        except Exception as error:
            raise WorkerSessionLifecycleError("state_invalid") from error
        if len(encoded) > self._max_carried_state_bytes:
            raise WorkerSessionLifecycleError("state_too_large")
        return eligible

    def _clear_service_state(self) -> None:
        # Production deliberately uses one allocation-local in-memory service.
        # Clearing app/user maps is required because delete_session only removes
        # conversation state in ADK.
        if not isinstance(self.service, InMemorySessionService):
            return
        self.service.sessions.pop(self.app_name, None)
        self.service.user_state.pop(self.app_name, None)
        self.service.app_state.pop(self.app_name, None)
