"""Bounded model-facing tools over one allocation-scoped Caido client."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import math
import secrets
import time
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any

from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.adapters.caido_graphql import CaidoClientError, CaidoGraphQLClient
from contractor_runtime.adapters.host import EMPTY_ADAPTER_HANDLES
from contractor_runtime.artifacts import MAX_ARTIFACT_BYTES, ArtifactClient, ArtifactTransportError
from contractor_runtime.contracts import ArtifactRef, RuntimeSettings, RuntimeSettingsV2
from contractor_runtime.toolsets.artifact_visibility import model_visible_exact_refs
from contractor_runtime.toolsets.run_artifacts import ArtifactClientFactory, ToolMetrics
from contractor_runtime.workspace import AllocationWorkspace

CAIDO_EXCHANGE_MEDIA_TYPE = "application/vnd.contractor.caido-exchange+json"
CAIDO_EXCHANGE_ARTIFACT_PREFIX = "caido.exchange."
CAIDO_EXCHANGE_SCHEMA_VERSION = "1.0"
MAX_PAGE_SIZE = 100
MAX_OFFSET = 1_000_000_000
MAX_ID_BYTES = 256
MAX_SHORT_TEXT_BYTES = 8192
MAX_TERM_BYTES = 2048
MAX_SCOPE_TERMS = 256
MAX_PREVIEW_CHARACTERS = 8192
MAX_BINARY_PREVIEW_BYTES = 6144
MAX_TOOL_RESULT_BYTES = 1024 * 1024
MAX_WORKFLOWS = 100
MAX_SITEMAP_ENTRIES = 100
MAX_AUTOMATE_ENTRIES = 100
MAX_PAYLOADS_PER_RESULT = 32

CAIDO_TOOL_NAMES = frozenset(
    {
        "caido_automate_results",
        "caido_automate_run",
        "caido_history",
        "caido_replay",
        "caido_request_detail",
        "caido_scope",
        "caido_sitemap",
        "caido_workflow_findings",
        "caido_workflow_list",
        "caido_workflow_run",
    }
)
CAIDO_READ_TOOL_NAMES = frozenset(
    {
        "caido_automate_results",
        "caido_history",
        "caido_request_detail",
        "caido_scope",
        "caido_sitemap",
        "caido_workflow_findings",
        "caido_workflow_list",
    }
)
_AUTOMATE_SORT_FIELDS = frozenset(
    {"RESP_STATUS_CODE", "RESP_LENGTH", "RESP_ROUNDTRIP_TIME", "POSITION", "PAYLOAD_0"}
)
_WORKFLOW_KINDS = frozenset({"convert", "active", "passive"})
_SITEMAP_DEPTHS = frozenset({"DIRECT", "ALL"})
_ERROR_RETRYABILITY = MappingProxyType(
    {
        "caido_not_configured": False,
        "caido_request_invalid": False,
        "caido_request_failed": True,
        "caido_response_invalid": False,
        "caido_response_too_large": False,
    }
)


class CaidoToolError(RuntimeError):
    """Stable content-free failure returned through the Worker tool boundary."""

    def __init__(self, code: str, *, retryable: bool | None = None) -> None:
        normalized = code if code in _ERROR_RETRYABILITY else "caido_request_failed"
        self.code = normalized
        self.retryable = _ERROR_RETRYABILITY[normalized] if retryable is None else bool(retryable)
        super().__init__(f"Caido operation failed ({normalized})")


class CaidoToolsetFactory:
    ref = "caido@1"
    exported_tools = CAIDO_TOOL_NAMES
    infrastructure_channels = MappingProxyType(
        {tool: frozenset({"caido-graphql-client"}) for tool in sorted(exported_tools)}
    )

    def __init__(self, artifact_client_factory: ArtifactClientFactory | None = None) -> None:
        self._artifact_client_factory = artifact_client_factory or _unconfigured_client

    async def probe(self) -> frozenset[str]:
        return CAIDO_READ_TOOL_NAMES

    async def create_selected(
        self,
        *,
        selected: Sequence[str],
        allocation_id: str,
        run_id: str,
        namespace: str,
        runtime_settings: RuntimeSettings,
        workspace: AllocationWorkspace,
        state: Any,
        adapter_handles: AdapterHandles = EMPTY_ADAPTER_HANDLES,
        project_workspace: Any = None,
    ) -> Mapping[str, Any]:
        del run_id, workspace, project_workspace
        unavailable = sorted(set(selected) - CAIDO_READ_TOOL_NAMES)
        if unavailable:
            raise ValueError(f"unavailable selected Caido tools: {', '.join(unavailable)}")
        metrics = getattr(state, "metrics", None)
        if metrics is None or not callable(getattr(metrics, "record_tool_call", None)):
            raise TypeError("caido@1 requires State.metrics")
        handle = adapter_handles.caido_graphql
        if not isinstance(handle, CaidoGraphQLClient):
            raise CaidoToolError("caido_not_configured")
        session = _CaidoSession(
            handle=handle,
            artifact_client=self._artifact_client_factory(allocation_id, runtime_settings),
            namespace=namespace,
            metric_secrets=_runtime_secrets(runtime_settings),
        )
        builders: dict[str, Callable[[], _CaidoTool]] = {
            "caido_scope": lambda: CaidoScopeTool(session, metrics),
            "caido_history": lambda: CaidoHistoryTool(session, metrics),
            "caido_request_detail": lambda: CaidoRequestDetailTool(session, metrics),
            "caido_automate_results": lambda: CaidoAutomateResultsTool(session, metrics),
            "caido_sitemap": lambda: CaidoSitemapTool(session, metrics),
            "caido_workflow_list": lambda: CaidoWorkflowListTool(session, metrics),
            "caido_workflow_findings": lambda: CaidoWorkflowFindingsTool(session, metrics),
        }
        return {name: builders[name]() for name in selected}


class _CaidoSession:
    def __init__(
        self,
        *,
        handle: CaidoGraphQLClient,
        artifact_client: ArtifactClient,
        namespace: str,
        metric_secrets: tuple[str, ...],
    ) -> None:
        self._handle: CaidoGraphQLClient | None = handle
        self._artifact_client: ArtifactClient | None = artifact_client
        self._namespace = namespace
        self._metric_secrets = metric_secrets
        self._nonce = secrets.token_hex(8)
        self._next_artifact = 1
        self._lock = asyncio.Lock()
        self._closed = False

    @property
    def metric_secrets(self) -> tuple[str, ...]:
        return self._metric_secrets

    @property
    def known_exact_refs(self) -> tuple[ArtifactRef, ...]:
        client = self._artifact_client
        return model_visible_exact_refs(getattr(client, "known_exact_refs", ()))

    async def scopes(self) -> dict[str, Any]:
        async with self._lock:
            data = await self._execute("scopes")
            _exact_object(data, {"scopes"})
            raw_scopes = _list(data["scopes"], maximum=MAX_PAGE_SIZE)
            scopes: list[dict[str, Any]] = []
            for raw in raw_scopes:
                scope = _exact_object(raw, {"id", "name", "allowlist", "denylist"})
                allowlist = _string_list(
                    scope["allowlist"], maximum=MAX_SCOPE_TERMS, item_bytes=MAX_TERM_BYTES
                )
                denylist = _string_list(
                    scope["denylist"], maximum=MAX_SCOPE_TERMS, item_bytes=MAX_TERM_BYTES
                )
                if len(allowlist) + len(denylist) > MAX_SCOPE_TERMS:
                    raise CaidoToolError("caido_response_invalid")
                scopes.append(
                    {
                        "id": _identifier(scope["id"]),
                        "name": _text(scope["name"]),
                        "allowlist": allowlist,
                        "denylist": denylist,
                    }
                )
            return _bounded_result({"scopes": scopes})

    async def history(self, filter_text: str, limit: int, offset: int) -> dict[str, Any]:
        selected_filter = _filter(filter_text)
        selected_limit, selected_offset = _page(limit, offset)
        variables: dict[str, Any] = {
            "limit": selected_limit,
            "offset": selected_offset,
            "order": {"by": "ID", "ordering": "DESC"},
        }
        if selected_filter:
            variables["filter"] = selected_filter
        async with self._lock:
            data = await self._execute("requests_by_offset", variables)
            connection = _connection(data, "requestsByOffset", selected_limit)
            requests = [_request_summary(node) for node in connection["nodes"]]
            return _bounded_result(
                {
                    "count": connection["count"],
                    "offset": selected_offset,
                    "requests": requests,
                }
            )

    async def request_detail(self, request_id: str) -> dict[str, Any]:
        selected_id = _request_identifier(request_id)
        async with self._lock:
            data = await self._execute("request_detail", {"id": selected_id})
            _exact_object(data, {"request"})
            if data["request"] is None:
                return {"request_id": selected_id, "status": "not_found"}
            request = _request_detail(data["request"])
            raw_request = _blob(request.pop("_raw"))
            response = request.get("response")
            raw_response = b""
            if isinstance(response, dict):
                raw_response = _blob(response.pop("_raw"))
            envelope = _exchange_envelope(raw_request, raw_response)
            artifact: ArtifactRef | None = None
            if envelope is not None:
                client = self._artifact_client
                if client is None:
                    raise CaidoToolError("caido_request_failed")
                encoded = json.dumps(
                    envelope,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode()
                if len(encoded) > MAX_ARTIFACT_BYTES:
                    raise CaidoToolError("caido_response_too_large")
                artifact_number = self._next_artifact
                self._next_artifact += 1
                try:
                    written = await client.write_artifact(
                        ArtifactRef(
                            namespace=self._namespace,
                            name=(
                                f"{CAIDO_EXCHANGE_ARTIFACT_PREFIX}{self._nonce}."
                                f"{artifact_number:06d}"
                            ),
                        ),
                        data=encoded,
                        media_type=CAIDO_EXCHANGE_MEDIA_TYPE,
                        expected_revision=None,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    raise CaidoToolError("caido_request_failed") from None
                artifact = written.artifact.require_exact()
            request["raw"] = _raw_preview(raw_request)
            if isinstance(response, dict):
                response["raw"] = _raw_preview(raw_response)
            request["raw_artifact"] = (
                None if artifact is None else artifact.model_dump(by_alias=True)
            )
            return _bounded_result(request)

    async def automate_results(
        self,
        session_id: str,
        entry_id: str,
        limit: int,
        offset: int,
        sort_by: str,
        ascending: bool,
    ) -> dict[str, Any]:
        selected_session = _request_identifier(session_id)
        selected_entry = _optional_identifier(entry_id)
        selected_limit, selected_offset = _page(limit, offset)
        if not isinstance(sort_by, str) or sort_by not in _AUTOMATE_SORT_FIELDS:
            raise CaidoToolError("caido_request_invalid")
        if type(ascending) is not bool:
            raise CaidoToolError("caido_request_invalid")
        async with self._lock:
            if not selected_entry:
                data = await self._execute("automate_session", {"id": selected_session})
                _exact_object(data, {"automateSession"})
                if data["automateSession"] is None:
                    return {"session_id": selected_session, "status": "not_found"}
                session = _automate_session(data["automateSession"])
                entries = session["entries"]
                if not entries:
                    return {"session_id": selected_session, "status": "no_entries"}
                selected_entry = entries[-1]["id"]
            data = await self._execute(
                "automate_entry_requests",
                {
                    "id": selected_entry,
                    "limit": selected_limit,
                    "offset": selected_offset,
                    "order": {
                        "by": sort_by,
                        "ordering": "ASC" if ascending else "DESC",
                    },
                },
            )
            _exact_object(data, {"automateEntry"})
            if data["automateEntry"] is None:
                return {
                    "session_id": selected_session,
                    "entry_id": selected_entry,
                    "status": "not_found",
                }
            entry = _automate_entry(data["automateEntry"], selected_limit)
            return _bounded_result(
                {
                    "session_id": selected_session,
                    "entry_id": selected_entry,
                    "total_results": entry["count"],
                    "offset": selected_offset,
                    "results": entry["results"],
                }
            )

    async def sitemap(self, parent_id: str, scope_id: str, depth: str) -> dict[str, Any]:
        selected_parent = _optional_identifier(parent_id)
        selected_scope = _optional_identifier(scope_id)
        if not isinstance(depth, str) or depth not in _SITEMAP_DEPTHS:
            raise CaidoToolError("caido_request_invalid")
        async with self._lock:
            if selected_parent:
                data = await self._execute(
                    "sitemap_descendants",
                    {"parentId": selected_parent, "depth": depth},
                )
                field = "sitemapDescendantEntries"
                include_parent = True
            else:
                variables = {} if not selected_scope else {"scopeId": selected_scope}
                data = await self._execute("sitemap_root", variables)
                field = "sitemapRootEntries"
                include_parent = False
            _exact_object(data, {field})
            connection = _exact_object(data[field], {"nodes"})
            nodes = _list(connection["nodes"], maximum=MAX_SITEMAP_ENTRIES)
            entries = [_sitemap_entry(node, include_parent=include_parent) for node in nodes]
            return _bounded_result({"entries": entries})

    async def workflow_list(self, kind: str) -> dict[str, Any]:
        if not isinstance(kind, str):
            raise CaidoToolError("caido_request_invalid")
        selected_kind = kind.strip().lower()
        if selected_kind and selected_kind not in _WORKFLOW_KINDS:
            raise CaidoToolError("caido_request_invalid")
        async with self._lock:
            data = await self._execute("workflows")
            _exact_object(data, {"workflows"})
            raw_workflows = _list(data["workflows"], maximum=MAX_WORKFLOWS)
            workflows = [_workflow(item) for item in raw_workflows]
            if selected_kind:
                workflows = [item for item in workflows if item["kind"] == selected_kind]
            return _bounded_result({"workflows": workflows})

    async def workflow_findings(self, limit: int, offset: int) -> dict[str, Any]:
        selected_limit, selected_offset = _page(limit, offset)
        async with self._lock:
            data = await self._execute(
                "findings_by_offset",
                {
                    "limit": selected_limit,
                    "offset": selected_offset,
                    "order": {"by": "ID", "ordering": "DESC"},
                },
            )
            connection = _connection(data, "findingsByOffset", selected_limit)
            findings = [_finding(item) for item in connection["nodes"]]
            return _bounded_result(
                {
                    "count": connection["count"],
                    "offset": selected_offset,
                    "findings": findings,
                }
            )

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._handle = None
            self._artifact_client = None
            self._namespace = ""
            self._nonce = ""
            self._metric_secrets = ()
            self._closed = True

    async def _execute(
        self, operation: str, variables: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        handle = self._handle
        if self._closed or handle is None:
            raise CaidoToolError("caido_request_failed")
        try:
            return await handle.execute(operation, variables)
        except asyncio.CancelledError:
            raise
        except CaidoClientError as error:
            raise CaidoToolError(error.code, retryable=error.retryable) from None
        except Exception:
            raise CaidoToolError("caido_request_failed") from None


class _CaidoTool:
    name: str
    description: str

    def __init__(self, session: _CaidoSession, metrics: ToolMetrics) -> None:
        self._session = session
        self._metrics = metrics
        self.__name__ = self.name
        self.__doc__ = self.description

    @property
    def known_exact_refs(self) -> tuple[ArtifactRef, ...]:
        return self._session.known_exact_refs

    async def close(self) -> None:
        await self._session.close()

    async def _call(
        self,
        arguments: Mapping[str, Any],
        summary: Callable[[Any], Mapping[str, Any]],
        operation: Callable[[], Any],
    ) -> Any:
        started_ns = time.perf_counter_ns()
        try:
            result = await operation()
            self._metrics.record_tool_call(
                self.name,
                arguments=arguments,
                result=summary(result),
                secrets=self._session.metric_secrets,
                duration_ms=_elapsed_ms(started_ns),
            )
            return result
        except asyncio.CancelledError:
            raise
        except Exception as error:
            bounded = (
                error
                if isinstance(error, CaidoToolError)
                else CaidoToolError("caido_request_failed")
            )
            self._metrics.record_tool_call(
                self.name,
                arguments=arguments,
                error=bounded,
                secrets=self._session.metric_secrets,
                duration_ms=_elapsed_ms(started_ns),
            )
            if isinstance(error, CaidoToolError):
                raise
            raise bounded from None


class CaidoScopeTool(_CaidoTool):
    name = "caido_scope"
    description = "List Caido proxy scopes; create becomes available only with the action slice."

    async def __call__(
        self,
        action: str = "list",
        name: str = "",
        allowlist: list[str] | None = None,
        denylist: list[str] | None = None,
    ) -> dict[str, Any]:
        arguments = {
            "action": (
                action if isinstance(action, str) and action in {"list", "create"} else "invalid"
            ),
            "allowCount": len(allowlist) if isinstance(allowlist, list) else 0,
            "denyCount": len(denylist) if isinstance(denylist, list) else 0,
            "hasName": bool(name) if isinstance(name, str) else False,
        }

        async def operation() -> dict[str, Any]:
            if action != "list" or name or allowlist is not None or denylist is not None:
                raise CaidoToolError("caido_request_invalid")
            return await self._session.scopes()

        return await self._call(
            arguments, lambda result: {"count": len(result["scopes"])}, operation
        )


class CaidoHistoryTool(_CaidoTool):
    name = "caido_history"
    description = "Query one bounded page of Caido proxy history using HTTPQL."

    async def __call__(self, filter: str = "", limit: int = 20, offset: int = 0) -> dict[str, Any]:
        arguments = {
            "hasFilter": bool(filter) if isinstance(filter, str) else False,
            "limit": limit if type(limit) is int else -1,
            "offset": offset if type(offset) is int else -1,
        }
        return await self._call(
            arguments,
            lambda result: {"count": len(result["requests"])},
            lambda: self._session.history(filter, limit, offset),
        )


class CaidoRequestDetailTool(_CaidoTool):
    name = "caido_request_detail"
    description = "Read bounded raw previews and an exact exchange artifact for one Caido request."

    async def __call__(self, request_id: str) -> dict[str, Any]:
        arguments = {"hasRequestId": bool(request_id) if isinstance(request_id, str) else False}
        return await self._call(
            arguments,
            lambda result: {
                "status": result.get("status", "found"),
                "hasArtifact": result.get("raw_artifact") is not None,
            },
            lambda: self._session.request_detail(request_id),
        )


class CaidoAutomateResultsTool(_CaidoTool):
    name = "caido_automate_results"
    description = "Read one bounded page of results from a Caido Automate session."

    async def __call__(
        self,
        session_id: str,
        entry_id: str = "",
        limit: int = 50,
        offset: int = 0,
        sort_by: str = "RESP_STATUS_CODE",
        ascending: bool = True,
    ) -> dict[str, Any]:
        arguments = {
            "hasSessionId": bool(session_id) if isinstance(session_id, str) else False,
            "hasEntryId": bool(entry_id) if isinstance(entry_id, str) else False,
            "limit": limit if type(limit) is int else -1,
            "offset": offset if type(offset) is int else -1,
            "sortBy": (
                sort_by
                if isinstance(sort_by, str) and sort_by in _AUTOMATE_SORT_FIELDS
                else "invalid"
            ),
            "ascending": ascending if type(ascending) is bool else False,
        }
        return await self._call(
            arguments,
            lambda result: {
                "status": result.get("status", "found"),
                "count": len(result.get("results", [])),
            },
            lambda: self._session.automate_results(
                session_id, entry_id, limit, offset, sort_by, ascending
            ),
        )


class CaidoSitemapTool(_CaidoTool):
    name = "caido_sitemap"
    description = "Browse one bounded root or descendant page of Caido's passive sitemap."

    async def __call__(
        self, parent_id: str = "", scope_id: str = "", depth: str = "DIRECT"
    ) -> dict[str, Any]:
        arguments = {
            "hasParentId": bool(parent_id) if isinstance(parent_id, str) else False,
            "hasScopeId": bool(scope_id) if isinstance(scope_id, str) else False,
            "depth": (depth if isinstance(depth, str) and depth in _SITEMAP_DEPTHS else "invalid"),
        }
        return await self._call(
            arguments,
            lambda result: {"count": len(result["entries"])},
            lambda: self._session.sitemap(parent_id, scope_id, depth),
        )


class CaidoWorkflowListTool(_CaidoTool):
    name = "caido_workflow_list"
    description = "List the bounded installed Caido workflow inventory, optionally by kind."

    async def __call__(self, kind: str = "") -> dict[str, Any]:
        selected = kind if isinstance(kind, str) and kind.lower() in _WORKFLOW_KINDS else ""
        arguments = {"kind": selected or ("all" if kind == "" else "invalid")}
        return await self._call(
            arguments,
            lambda result: {"count": len(result["workflows"])},
            lambda: self._session.workflow_list(kind),
        )


class CaidoWorkflowFindingsTool(_CaidoTool):
    name = "caido_workflow_findings"
    description = "Read one bounded newest-first page of findings reported in Caido."

    async def __call__(self, limit: int = 20, offset: int = 0) -> dict[str, Any]:
        arguments = {
            "limit": limit if type(limit) is int else -1,
            "offset": offset if type(offset) is int else -1,
        }
        return await self._call(
            arguments,
            lambda result: {"count": len(result["findings"])},
            lambda: self._session.workflow_findings(limit, offset),
        )


def _request_summary(value: object) -> dict[str, Any]:
    node = _exact_object(
        value,
        {
            "id",
            "method",
            "host",
            "path",
            "port",
            "query",
            "isTls",
            "source",
            "createdAt",
            "response",
        },
    )
    response = _request_response_summary(node["response"])
    return {
        "id": _identifier(node["id"]),
        "method": _text(node["method"]),
        "host": _text(node["host"]),
        "path": _text(node["path"]),
        "port": _nullable_integer(node["port"], minimum=1, maximum=65535),
        "query": _text(node["query"]),
        "is_tls": _boolean(node["isTls"]),
        "source": _text(node["source"]),
        "created_at": _text(node["createdAt"]),
        "status_code": None if response is None else response["status_code"],
        "response_length": None if response is None else response["length"],
        "roundtrip_ms": None if response is None else response["roundtrip_ms"],
    }


def _request_response_summary(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    response = _exact_object(value, {"statusCode", "length", "roundtripTime"})
    return {
        "status_code": _nullable_integer(response["statusCode"], minimum=100, maximum=999),
        "length": _nullable_integer(response["length"], minimum=0),
        "roundtrip_ms": _nullable_number(response["roundtripTime"], minimum=0),
    }


def _request_detail(value: object) -> dict[str, Any]:
    request = _exact_object(
        value,
        {
            "id",
            "method",
            "host",
            "path",
            "port",
            "query",
            "isTls",
            "raw",
            "createdAt",
            "source",
            "response",
        },
    )
    response: dict[str, Any] | None = None
    if request["response"] is not None:
        raw_response = _exact_object(
            request["response"],
            {"id", "statusCode", "length", "roundtripTime", "raw"},
        )
        response = {
            "id": _nullable_identifier(raw_response["id"]),
            "status_code": _nullable_integer(raw_response["statusCode"], minimum=100, maximum=999),
            "length": _nullable_integer(raw_response["length"], minimum=0),
            "roundtrip_ms": _nullable_number(raw_response["roundtripTime"], minimum=0),
            "_raw": raw_response["raw"],
        }
    return {
        "id": _identifier(request["id"]),
        "method": _text(request["method"]),
        "host": _text(request["host"]),
        "path": _text(request["path"]),
        "port": _nullable_integer(request["port"], minimum=1, maximum=65535),
        "query": _text(request["query"]),
        "is_tls": _boolean(request["isTls"]),
        "created_at": _text(request["createdAt"]),
        "source": _text(request["source"]),
        "response": response,
        "_raw": request["raw"],
    }


def _automate_session(value: object) -> dict[str, Any]:
    session = _exact_object(value, {"id", "name", "entries", "settings"})
    entries: list[dict[str, Any]] = []
    for raw in _list(session["entries"], maximum=MAX_AUTOMATE_ENTRIES):
        entry = _exact_object(raw, {"id", "name", "createdAt"})
        entries.append(
            {
                "id": _identifier(entry["id"]),
                "name": _text(entry["name"]),
                "created_at": _text(entry["createdAt"]),
            }
        )
    settings = _exact_object(session["settings"], {"strategy", "placeholders"})
    _text(settings["strategy"])
    for raw in _list(settings["placeholders"], maximum=32):
        placeholder = _exact_object(raw, {"start", "end"})
        start = _integer(placeholder["start"], minimum=0)
        end = _integer(placeholder["end"], minimum=0)
        if end < start:
            raise CaidoToolError("caido_response_invalid")
    return {"id": _identifier(session["id"]), "entries": entries}


def _automate_entry(value: object, limit: int) -> dict[str, Any]:
    entry = _exact_object(value, {"id", "name", "requestsByOffset"})
    _identifier(entry["id"])
    _text(entry["name"])
    connection = _raw_connection(entry["requestsByOffset"], limit)
    results: list[dict[str, Any]] = []
    for raw in connection["nodes"]:
        node = _exact_object(raw, {"sequenceId", "error", "payloads", "request"})
        error = node["error"]
        if error is not None and not isinstance(error, (str, bool)):
            raise CaidoToolError("caido_response_invalid")
        payloads: list[str] = []
        for raw_payload in _list(node["payloads"], maximum=MAX_PAYLOADS_PER_RESULT):
            payload = _exact_object(raw_payload, {"position", "raw"})
            _integer(payload["position"], minimum=0)
            decoded = _blob(payload["raw"])
            try:
                text = decoded.decode()
            except UnicodeDecodeError:
                raise CaidoToolError("caido_response_invalid") from None
            if len(text.encode()) > MAX_SHORT_TEXT_BYTES:
                raise CaidoToolError("caido_response_invalid")
            payloads.append(text)
        request: dict[str, Any] = {}
        if node["request"] is not None:
            raw_request = _exact_object(
                node["request"], {"id", "method", "host", "path", "query", "response"}
            )
            response = _request_response_summary(raw_request["response"])
            request = {
                "request_id": _identifier(raw_request["id"]),
                "method": _text(raw_request["method"]),
                "host": _text(raw_request["host"]),
                "path": _text(raw_request["path"]),
                "query": _text(raw_request["query"]),
                "status_code": None if response is None else response["status_code"],
                "response_length": None if response is None else response["length"],
                "roundtrip_ms": None if response is None else response["roundtrip_ms"],
            }
        results.append(
            {
                "sequence_id": _nullable_integer(node["sequenceId"], minimum=0),
                "payloads": payloads,
                "failed": error is not None and error is not False,
                **request,
            }
        )
    return {"count": connection["count"], "results": results}


def _sitemap_entry(value: object, *, include_parent: bool) -> dict[str, Any]:
    expected = {"id", "label", "kind", "hasDescendants", "metadata"}
    if include_parent:
        expected.add("parentId")
    node = _exact_object(value, expected)
    metadata = node["metadata"]
    if metadata is not None:
        selected = _object(metadata)
        if selected:
            _exact_object(selected, {"isTls", "port"})
            _boolean(selected["isTls"])
            _nullable_integer(selected["port"], minimum=1, maximum=65535)
    result = {
        "id": _identifier(node["id"]),
        "label": _text(node["label"]),
        "kind": _text(node["kind"]),
        "has_children": _boolean(node["hasDescendants"]),
    }
    if include_parent:
        result["parent_id"] = _nullable_identifier(node["parentId"])
    return result


def _workflow(value: object) -> dict[str, Any]:
    item = _exact_object(value, {"id", "name", "kind", "enabled", "global"})
    kind = _text(item["kind"]).lower()
    if kind not in _WORKFLOW_KINDS:
        raise CaidoToolError("caido_response_invalid")
    return {
        "id": _identifier(item["id"]),
        "name": _text(item["name"]),
        "kind": kind,
        "enabled": _boolean(item["enabled"]),
        "global": _boolean(item["global"]),
    }


def _finding(value: object) -> dict[str, Any]:
    item = _exact_object(
        value,
        {"id", "title", "description", "host", "path", "reporter", "createdAt", "request"},
    )
    request_id = None
    if item["request"] is not None:
        request = _exact_object(item["request"], {"id", "method", "host", "path"})
        request_id = _identifier(request["id"])
        _text(request["method"])
        _text(request["host"])
        _text(request["path"])
    return {
        "id": _identifier(item["id"]),
        "title": _text(item["title"]),
        "description": _text(item["description"]),
        "reporter": _text(item["reporter"]),
        "host": _text(item["host"]),
        "path": _text(item["path"]),
        "created_at": _text(item["createdAt"]),
        "request_id": request_id,
    }


def _connection(data: object, field: str, limit: int) -> dict[str, Any]:
    selected = _exact_object(data, {field})
    return _raw_connection(selected[field], limit)


def _raw_connection(value: object, limit: int) -> dict[str, Any]:
    connection = _exact_object(value, {"count", "nodes"})
    count_object = _exact_object(connection["count"], {"value"})
    count = _integer(count_object["value"], minimum=0)
    nodes = _list(connection["nodes"], maximum=limit)
    if count < len(nodes):
        raise CaidoToolError("caido_response_invalid")
    return {"count": count, "nodes": nodes}


def _exchange_envelope(request: bytes, response: bytes) -> dict[str, Any] | None:
    if not request and not response:
        return None
    return {
        "schemaVersion": CAIDO_EXCHANGE_SCHEMA_VERSION,
        "request": _raw_envelope_value(request),
        "response": _raw_envelope_value(response),
    }


def _raw_envelope_value(value: bytes) -> dict[str, Any] | None:
    if not value:
        return None
    try:
        text = value.decode()
    except UnicodeDecodeError:
        return {"kind": "binary", "dataBase64": base64.b64encode(value).decode("ascii")}
    return {"kind": "text", "text": text}


def _raw_preview(value: bytes) -> dict[str, Any] | None:
    if not value:
        return None
    try:
        text = value.decode()
    except UnicodeDecodeError:
        selected = value[:MAX_BINARY_PREVIEW_BYTES]
        return {
            "kind": "binary",
            "size": len(value),
            "data_b64": base64.b64encode(selected).decode("ascii"),
            "truncated": len(selected) < len(value),
        }
    selected_text = text[:MAX_PREVIEW_CHARACTERS]
    return {
        "kind": "text",
        "size": len(value),
        "data": selected_text,
        "truncated": len(selected_text) < len(text),
    }


def _blob(value: object) -> bytes:
    if value is None or value == "":
        return b""
    if not isinstance(value, str) or len(value) > ((MAX_ARTIFACT_BYTES + 2) // 3) * 4:
        raise CaidoToolError("caido_response_invalid")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise CaidoToolError("caido_response_invalid") from None
    if len(decoded) > MAX_ARTIFACT_BYTES or base64.b64encode(decoded).decode("ascii") != value:
        raise CaidoToolError("caido_response_invalid")
    return decoded


def _page(limit: object, offset: object) -> tuple[int, int]:
    selected_limit = _integer(limit, minimum=1, maximum=MAX_PAGE_SIZE, request=True)
    selected_offset = _integer(offset, minimum=0, maximum=MAX_OFFSET, request=True)
    return selected_limit, selected_offset


def _filter(value: object) -> str:
    if not isinstance(value, str) or len(value.encode()) > MAX_SHORT_TEXT_BYTES:
        raise CaidoToolError("caido_request_invalid")
    if any(ord(character) < 32 for character in value):
        raise CaidoToolError("caido_request_invalid")
    return value


def _request_identifier(value: object) -> str:
    try:
        return _identifier(value)
    except CaidoToolError:
        raise CaidoToolError("caido_request_invalid") from None


def _identifier(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode()) > MAX_ID_BYTES
        or any(ord(character) < 32 for character in value)
    ):
        raise CaidoToolError("caido_response_invalid")
    return value


def _optional_identifier(value: object) -> str:
    if value == "":
        return ""
    return _request_identifier(value)


def _nullable_identifier(value: object) -> str | None:
    return None if value is None else _identifier(value)


def _text(value: object) -> str:
    if not isinstance(value, str) or len(value.encode()) > MAX_SHORT_TEXT_BYTES:
        raise CaidoToolError("caido_response_invalid")
    return value


def _string_list(value: object, *, maximum: int, item_bytes: int) -> list[str]:
    result = _list(value, maximum=maximum)
    if any(not isinstance(item, str) or len(item.encode()) > item_bytes for item in result):
        raise CaidoToolError("caido_response_invalid")
    return list(result)


def _object(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise CaidoToolError("caido_response_invalid")
    return value


def _exact_object(value: object, keys: set[str]) -> dict[str, Any]:
    result = _object(value)
    if set(result) != keys:
        raise CaidoToolError("caido_response_invalid")
    return result


def _list(value: object, *, maximum: int) -> list[Any]:
    if not isinstance(value, list) or len(value) > maximum:
        raise CaidoToolError("caido_response_invalid")
    return value


def _boolean(value: object) -> bool:
    if type(value) is not bool:
        raise CaidoToolError("caido_response_invalid")
    return value


def _integer(
    value: object,
    *,
    minimum: int,
    maximum: int | None = None,
    request: bool = False,
) -> int:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise CaidoToolError("caido_request_invalid" if request else "caido_response_invalid")
    return value


def _nullable_integer(value: object, *, minimum: int, maximum: int | None = None) -> int | None:
    return None if value is None else _integer(value, minimum=minimum, maximum=maximum)


def _nullable_number(value: object, *, minimum: float) -> int | float | None:
    if value is None:
        return None
    if type(value) not in {int, float} or not math.isfinite(value) or value < minimum:
        raise CaidoToolError("caido_response_invalid")
    return value


def _bounded_result(value: dict[str, Any]) -> dict[str, Any]:
    try:
        encoded = json.dumps(
            value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode()
    except (TypeError, ValueError):
        raise CaidoToolError("caido_response_invalid") from None
    if len(encoded) > MAX_TOOL_RESULT_BYTES:
        raise CaidoToolError("caido_response_too_large")
    return value


def _runtime_secrets(settings: RuntimeSettings) -> tuple[str, ...]:
    values: list[str] = []
    token = settings.llm_gateway_token
    if token is not None:
        values.append(token.get_secret_value())
    if (
        isinstance(settings, RuntimeSettingsV2)
        and settings.caido is not None
        and settings.caido.bearer_token is not None
    ):
        values.append(settings.caido.bearer_token.get_secret_value())
    return tuple(value for value in values if value)


def _elapsed_ms(started_ns: int) -> int:
    return max(0, (time.perf_counter_ns() - started_ns) // 1_000_000)


class _UnavailableArtifactTransport:
    async def request(self, *_args: Any, **_kwargs: Any) -> Any:
        raise ArtifactTransportError("Artifact transport is not configured")


def _unconfigured_client(allocation_id: str, runtime_settings: RuntimeSettings) -> ArtifactClient:
    del runtime_settings
    return ArtifactClient(allocation_id, _UnavailableArtifactTransport())
