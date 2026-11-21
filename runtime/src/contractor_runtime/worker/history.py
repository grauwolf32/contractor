"""Seal interrupted tool batches before ADK projects their next request history."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Never

from google.adk.events import Event
from google.adk.sessions import BaseSessionService
from google.adk.sessions.base_session_service import GetSessionConfig
from google.adk.sessions.session import Session
from google.genai import types

from contractor_runtime.llm.image_parts import image_result_metadata
from contractor_runtime.threads import to_thread_until_done

MAX_HISTORY_CALLS = 10_000
MAX_HISTORY_RETAINED_BYTES = 32 * 1024 * 1024
_MAX_TURN_IDS = 1001  # 1000 admitted model calls plus a refused final turn.


class ToolHistoryError(RuntimeError):
    """An interrupted history cannot safely continue; callers must fence it."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"Worker tool history failed ({code})")


@dataclass(slots=True)
class _Call:
    call_id: str
    name: str
    event_id: str
    timestamp: float
    branch: str | None
    isolation_scope: str | None
    node_path: str
    metadata_bytes: int
    persisted: bool = False
    admitted: bool = False
    cancelled: bool = False
    result: bytes | None = None


class ToolHistory:
    """Retain only unresolved calls/results of the current logical invocation.

    ``observe_event`` runs before the SDK appends a finalized event. The Runtime
    calls ``persisted`` only after Runner yields it, which acknowledges append.
    Once Runner and its tools have joined, ``seal`` uses the public SessionService
    to pair saved calls without repeating semantic work. Raw arguments already
    belong to the in-memory SDK session and are not copied into this ledger.
    """

    def __init__(
        self,
        app_name: str,
        user_id: str,
        agent_name: str = "contractor_worker",
        *,
        max_calls: int = MAX_HISTORY_CALLS,
        max_retained_bytes: int = MAX_HISTORY_RETAINED_BYTES,
    ) -> None:
        if not app_name or not user_id or not agent_name:
            raise ValueError("Tool history identities are required")
        if (
            type(max_calls) is not int
            or not 0 < max_calls <= MAX_HISTORY_CALLS
            or type(max_retained_bytes) is not int
            or not 0 < max_retained_bytes <= MAX_HISTORY_RETAINED_BYTES
        ):
            raise ValueError("Tool history bounds must be positive and supported")
        self._failure_code: str | None = None
        self._app_name = app_name
        self._user_id = user_id
        self._agent_name = agent_name
        self._max_calls = max_calls
        self._max_bytes = max_retained_bytes
        self._logical_id: str | None = None
        self._sdk_id: str | None = None
        self._session_id: str | None = None
        self._turn_ids: set[str] = set()
        self._observed_calls = 0
        self._calls: dict[str, _Call] = {}
        self._retained_bytes = 0
        self._snapshot_lock = asyncio.Lock()
        self._sealing = False
        self._pending_seals: dict[str, Event] = {}

    def check(self) -> None:
        """Refuse further work after the first history failure, even if SDK caught it."""
        if self._failure_code is not None:
            raise ToolHistoryError(self._failure_code)

    def _fail(self, code: str) -> Never:
        if self._failure_code is None:
            self._failure_code = code
        raise ToolHistoryError(self._failure_code)

    @property
    def needs_seal(self) -> bool:
        return bool(self._calls)

    def begin_turn(
        self,
        logical_invocation_id: str,
        sdk_invocation_id: str,
        session_id: str,
    ) -> None:
        self.check()
        if self._calls or self._pending_seals or self._snapshot_lock.locked() or self._sealing:
            self._fail("invalid_tool_history")
        if not logical_invocation_id or not sdk_invocation_id or not session_id:
            self._fail("invalid_tool_history")
        if logical_invocation_id != self._logical_id:
            self._turn_ids.clear()
            self._observed_calls = 0
        elif session_id != self._session_id:
            self._fail("invalid_tool_history")
        if sdk_invocation_id in self._turn_ids or len(self._turn_ids) >= _MAX_TURN_IDS:
            self._fail("invalid_tool_history")
        self._logical_id = logical_invocation_id
        self._sdk_id = sdk_invocation_id
        self._session_id = session_id
        self._turn_ids.add(sdk_invocation_id)

    def observe_event(self, event: Event) -> None:
        self._check_event(event)
        if event.partial:
            return
        calls = event.get_function_calls()
        if not calls:
            return
        if event.author != self._agent_name or event.get_function_responses() or not event.id:
            self._fail("invalid_tool_history")
        if self._observed_calls + len(calls) > self._max_calls:
            self._fail("tool_history_limit_exceeded")
        additions: dict[str, _Call] = {}
        added_bytes = 0
        for call in calls:
            if not call.id or not call.name or call.id in self._calls or call.id in additions:
                self._fail("invalid_tool_history")
            try:
                metadata_bytes = _metadata_size(
                    call.id,
                    call.name,
                    event.id,
                    event.branch,
                    event.isolation_scope,
                    event.node_info.path,
                )
            except ToolHistoryError as error:
                self._fail(error.code)
            added_bytes += metadata_bytes
            if self._retained_bytes + added_bytes > self._max_bytes:
                self._fail("tool_history_limit_exceeded")
            additions[call.id] = _Call(
                call_id=call.id,
                name=call.name,
                event_id=event.id,
                timestamp=event.timestamp,
                branch=event.branch,
                isolation_scope=event.isolation_scope,
                node_path=event.node_info.path,
                metadata_bytes=metadata_bytes,
            )
        self._calls.update(additions)
        self._retained_bytes += added_bytes
        self._observed_calls += len(additions)

    def persisted(self, event: Event) -> None:
        self._check_event(event)
        if event.partial:
            return
        for call in event.get_function_calls():
            pending = self._calls.get(call.id or "")
            if (
                pending is None
                or pending.event_id != event.id
                or pending.timestamp != event.timestamp
                or pending.name != call.name
            ):
                self._fail("invalid_tool_history")
            self._check_scope(pending, event.branch, event.isolation_scope)
            pending.persisted = True
        for response in event.get_function_responses():
            pending = self._calls.get(response.id or "")
            if pending is None or pending.name != response.name:
                self._fail("invalid_tool_history")
            self._check_scope(pending, event.branch, event.isolation_scope)
            self._remove(pending)

    def admitted(self, tool_context: Any) -> None:
        pending = self._for_context(tool_context)
        if not pending.persisted or pending.admitted:
            self._fail("invalid_tool_history")
        pending.admitted = True

    async def completed(self, tool_context: Any, result: Any) -> None:
        await self._record_result(tool_context, result, cancelled=False)

    async def failed(self, tool_context: Any, result: Any, cancelled: bool = False) -> None:
        """Record a safe result, or mark cancellation without inventing a result.

        A cancellation notification may pass ``None``; then seal uses the closed
        cancelled outcome. A previously completed result always takes precedence.
        """
        if result is None:
            pending = self._for_context(tool_context)
            pending.cancelled = pending.cancelled or cancelled
            return
        await self._record_result(tool_context, result, cancelled=cancelled)

    async def _record_result(self, tool_context: Any, result: Any, *, cancelled: bool) -> None:
        # Only one snapshot exists in flight: parallel tools cannot each reserve
        # the entire remaining byte budget while serialization runs off-loop.
        # A result is already known when this method begins. Cancellation must
        # join the preceding snapshot and ours even if it arrives while waiting
        # for the shared byte reservation, rather than discard that result.
        cancelled_while_waiting = False
        while True:
            try:
                await self._snapshot_lock.acquire()
                break
            except asyncio.CancelledError:
                cancelled_while_waiting = True
        try:
            pending = self._for_context(tool_context)
            if pending.result is not None:
                if cancelled:
                    pending.cancelled = True
                    return
                self._fail("invalid_tool_history")
            available = self._max_bytes - self._retained_bytes
            snapshot: list[bytes] = []

            def encode() -> None:
                payload = (
                    image_result_metadata(result, replay=True)
                    if isinstance(result, dict)
                    else {"result": result}
                )
                snapshot.append(_encode_bounded(payload, available))

            try:
                await to_thread_until_done(encode, name="worker-tool-history-snapshot")
            except ToolHistoryError as error:
                self._fail(error.code)
            except Exception:
                self._fail("invalid_tool_history")
            except asyncio.CancelledError:
                # The original tool already completed. Keep its known result if
                # the joined serializer finished, even though cancellation wins.
                if snapshot:
                    # A concurrent failure is already latched. Preserve the
                    # cancellation that owns this serializer cleanup.
                    with contextlib.suppress(ToolHistoryError):
                        self._store_result(tool_context, pending, snapshot[0], cancelled=cancelled)
                raise
            self._store_result(tool_context, pending, snapshot[0], cancelled=cancelled)
        finally:
            self._snapshot_lock.release()
            if cancelled_while_waiting:
                # Any encoding error has already latched the history failure.
                # The caller's cancellation still owns its semantic outcome.
                raise asyncio.CancelledError

    def _store_result(
        self, tool_context: Any, pending: _Call, encoded: bytes, *, cancelled: bool
    ) -> None:
        if self._for_context(tool_context) is not pending:
            self._fail("invalid_tool_history")
        if self._retained_bytes + len(encoded) > self._max_bytes:
            self._fail("tool_history_limit_exceeded")
        pending.result = encoded
        pending.cancelled = pending.cancelled or cancelled
        self._retained_bytes += len(encoded)

    async def seal(self, service: BaseSessionService, session: Session) -> None:
        self.check()
        self._check_session(session)
        if not self._calls:
            return
        if self._sealing or self._snapshot_lock.locked():
            self._fail("invalid_tool_history")
        self._sealing = True
        try:
            current = await self._reload(service)
            self._reconcile_source(current)
            groups: dict[str, list[_Call]] = {}
            for pending in self._calls.values():
                groups.setdefault(self._source_key(pending), []).append(pending)
            for source_id, group in groups.items():
                event = self._pending_seals.get(source_id)
                if event is None:
                    event = await to_thread_until_done(
                        self._seal_event, source_id, group, name="worker-tool-history-seal"
                    )
                    self._pending_seals[source_id] = event
                try:
                    await service.append_event(session=current, event=event)
                except asyncio.CancelledError:
                    # Keep the exact pending event for the cleanup owner. Never
                    # turn cancellation into success or retry an uncertain append.
                    raise
                except Exception:
                    current = await self._reload(service)
                    if not any(item.id == event.id and item == event for item in current.events):
                        self._fail("tool_history_unavailable")
                for pending in group:
                    self._remove(pending)
                self._pending_seals.pop(source_id, None)
        except ToolHistoryError as error:
            self._fail(error.code)
        except Exception:
            self._fail("invalid_tool_history")
        finally:
            self._sealing = False

    def clear(self) -> None:
        if self._snapshot_lock.locked() or self._sealing:
            self._fail("invalid_tool_history")
        self._failure_code = None
        self._calls.clear()
        self._pending_seals.clear()
        self._retained_bytes = 0
        self._observed_calls = 0
        self._turn_ids.clear()
        self._logical_id = self._sdk_id = self._session_id = None

    async def _reload(self, service: BaseSessionService) -> Session:
        maximum_events = 2 * len(self._calls) + 3
        try:
            current = await service.get_session(
                app_name=self._app_name,
                user_id=self._user_id,
                session_id=self._session_id or "",
                config=GetSessionConfig(num_recent_events=maximum_events),
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            self._fail("tool_history_unavailable")
        if current is None or len(current.events) > maximum_events:
            self._fail("tool_history_unavailable")
        self._check_session(current)
        return current

    def _reconcile_source(self, session: Session) -> None:
        # One bounded suffix pass handles reused IDs correctly: a result belongs
        # to the latest preceding call, never an older occurrence with that ID.
        active: dict[str, _Call] = {}
        found: set[str] = set()
        resolved: set[str] = set()
        for event in session.events:
            if event.invocation_id != self._sdk_id:
                continue
            for call in event.get_function_calls():
                pending = self._calls.get(call.id or "")
                if (
                    pending is not None
                    and pending.event_id == event.id
                    and pending.timestamp == event.timestamp
                ):
                    if pending.name != call.name:
                        self._fail("invalid_tool_history")
                    self._check_scope(pending, event.branch, event.isolation_scope)
                    active[pending.call_id] = pending
                    found.add(pending.call_id)
                    resolved.discard(pending.call_id)
                else:
                    active.pop(call.id or "", None)
            for response in event.get_function_responses():
                pending = active.pop(response.id or "", None)
                if pending is not None:
                    if pending.name != response.name:
                        self._fail("invalid_tool_history")
                    self._check_scope(pending, event.branch, event.isolation_scope)
                    resolved.add(pending.call_id)
        for pending in tuple(self._calls.values()):
            if pending.call_id in resolved:
                self._remove(pending)
            elif pending.call_id in found:
                pending.persisted = True
            elif pending.persisted:
                # A caller that acknowledged persistence cannot silently lose
                # the call or fetch an arbitrary unbounded history to find it.
                self._fail("tool_history_unavailable")
            else:
                # No persisted selecting event: SDK never resumed to dispatch.
                self._remove(pending)
        unresolved_sources = {self._source_key(pending) for pending in self._calls.values()}
        for source_id in tuple(self._pending_seals):
            if source_id not in unresolved_sources:
                self._pending_seals.pop(source_id)

    def _seal_event(self, source_id: str, group: list[_Call]) -> Event:
        parts = []
        for pending in group:
            if pending.result is not None:
                response = json.loads(pending.result)
            else:
                code = (
                    "tool_call_not_executed"
                    if not pending.admitted
                    else "tool_call_cancelled"
                    if pending.cancelled
                    else "tool_call_outcome_unknown"
                )
                response = {"ok": False, "error": {"code": code}}
            parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        id=pending.call_id, name=pending.name, response=response
                    )
                )
            )
        first = group[0]
        identity = json.dumps(
            [self._session_id, self._sdk_id, source_id, [p.call_id for p in group]],
            ensure_ascii=True,
            separators=(",", ":"),
        )
        return Event(
            id="worker-history-" + hashlib.sha256(identity.encode()).hexdigest(),
            invocation_id=self._sdk_id or "",
            author=self._agent_name,
            branch=first.branch,
            isolation_scope=first.isolation_scope,
            node_info={"path": first.node_path},
            content=types.Content(role="user", parts=parts),
        )

    def _check_event(self, event: Event) -> None:
        self.check()
        if (
            self._sdk_id is None
            or event.invocation_id != self._sdk_id
            or self._sealing
            or self._pending_seals
        ):
            self._fail("invalid_tool_history")

    def _check_session(self, session: Session) -> None:
        if (
            self._session_id is None
            or getattr(session, "app_name", None) != self._app_name
            or getattr(session, "user_id", None) != self._user_id
            or getattr(session, "id", None) != self._session_id
        ):
            self._fail("invalid_tool_history")

    def _for_context(self, context: Any) -> _Call:
        self.check()
        if (
            self._sdk_id is None
            or getattr(context, "invocation_id", None) != self._sdk_id
            or self._sealing
            or self._pending_seals
        ):
            self._fail("invalid_tool_history")
        session = getattr(context, "session", None)
        if session is None:
            self._fail("invalid_tool_history")
        self._check_session(session)
        pending = self._calls.get(getattr(context, "function_call_id", None))
        if pending is None:
            self._fail("invalid_tool_history")
        self._check_scope(
            pending, getattr(context, "branch", None), getattr(context, "isolation_scope", None)
        )
        return pending

    @staticmethod
    def _source_key(pending: _Call) -> str:
        # SDK timestamps distinguish distinct events even under a test UUID
        # provider that stamps more than one event with the same ID.
        identity = json.dumps(
            [
                pending.event_id,
                pending.timestamp,
                pending.branch,
                pending.isolation_scope,
                pending.node_path,
            ],
            ensure_ascii=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(identity.encode()).hexdigest()

    def _check_scope(self, pending: _Call, branch: str | None, isolation_scope: str | None) -> None:
        if pending.branch != branch or pending.isolation_scope != isolation_scope:
            self._fail("invalid_tool_history")

    def _remove(self, pending: _Call) -> None:
        if self._calls.pop(pending.call_id, None) is pending:
            self._retained_bytes -= pending.metadata_bytes + len(pending.result or b"")


def _metadata_size(*values: str | None) -> int:
    try:
        return sum(len(value.encode("utf-8")) for value in values if value is not None)
    except (AttributeError, UnicodeError):
        raise ToolHistoryError("invalid_tool_history") from None


def _encode_bounded(value: Any, maximum: int) -> bytes:
    """Preflight JSON sizes so a single escaped string cannot allocate past cap."""
    try:
        _json_size(value, maximum, set())
        encoded = json.dumps(
            value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
    except ToolHistoryError:
        raise
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise ToolHistoryError("invalid_tool_history") from None
    if len(encoded) > maximum:
        raise ToolHistoryError("tool_history_limit_exceeded")
    return encoded


def _json_size(value: Any, maximum: int, active: set[int]) -> int:
    if maximum < 0:
        raise ToolHistoryError("tool_history_limit_exceeded")
    if isinstance(value, str):
        size = 2
        for char in value:
            ordinal = ord(char)
            if char in {'"', "\\", "\b", "\f", "\n", "\r", "\t"}:
                size += 2
            elif ordinal < 32:
                size += 6
            elif 0xD800 <= ordinal <= 0xDFFF:
                raise ToolHistoryError("invalid_tool_history")
            else:
                size += 1 if ordinal < 128 else 2 if ordinal < 2048 else 3 if ordinal < 65536 else 4
            if size > maximum:
                raise ToolHistoryError("tool_history_limit_exceeded")
        return size
    if isinstance(value, dict | list):
        identity = id(value)
        if identity in active:
            raise ToolHistoryError("invalid_tool_history")
        active.add(identity)
        size = 2
        try:
            for index, item in enumerate(value.items() if isinstance(value, dict) else value):
                size += int(index != 0)
                if isinstance(value, dict):
                    key, item = item
                    if not isinstance(key, str):
                        raise ToolHistoryError("invalid_tool_history")
                    size += _json_size(key, maximum - size, active) + 1
                size += _json_size(item, maximum - size, active)
                if size > maximum:
                    raise ToolHistoryError("tool_history_limit_exceeded")
        finally:
            active.remove(identity)
        return size
    if value is None or isinstance(value, bool | int | float):
        size = len(json.dumps(value, allow_nan=False))
        if size > maximum:
            raise ToolHistoryError("tool_history_limit_exceeded")
        return size
    raise ToolHistoryError("invalid_tool_history")
