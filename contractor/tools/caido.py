"""Caido proxy integration tools.

Provides LLM agents with programmatic access to Caido's web security
proxy for automated fuzzing (Automate), HTTP history analysis, and
endpoint discovery via sitemap.

Caido exposes a GraphQL API at ``/graphql``; this module wraps the key
operations into LLM-friendly tool functions following the same factory
pattern as ``http.py`` and ``vuln.py``.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx
from google.adk.tools.tool_context import ToolContext

logger = logging.getLogger(__name__)

CaidoStrategy = Literal["SEQUENTIAL", "PARALLEL", "MATRIX", "ALL"]

_REQUESTS_QUERY = """
query RequestsByOffset($limit: Int, $offset: Int, $filter: HTTPQL, $order: RequestResponseOrderInput) {
  requestsByOffset(limit: $limit, offset: $offset, filter: $filter, order: $order) {
    count { value }
    nodes {
      id
      method
      host
      path
      port
      query
      isTls
      source
      createdAt
      response {
        statusCode
        length
        roundtripTime
      }
    }
  }
}
"""

_REQUEST_DETAIL_QUERY = """
query RequestDetail($id: ID!) {
  request(id: $id) {
    id
    method
    host
    path
    port
    query
    isTls
    raw
    createdAt
    source
    response {
      id
      statusCode
      length
      roundtripTime
      raw
    }
  }
}
"""

_SCOPES_QUERY = """
query Scopes {
  scopes {
    id
    name
    allowlist
    denylist
  }
}
"""

_CREATE_SCOPE = """
mutation CreateScope($input: CreateScopeInput!) {
  createScope(input: $input) {
    error { ... on InvalidGlobTermsUserError { code } ... on OtherUserError { code } }
    scope { id name allowlist denylist }
  }
}
"""

_CREATE_AUTOMATE_SESSION = """
mutation CreateAutomateSession($input: CreateAutomateSessionInput!) {
  createAutomateSession(input: $input) {
    session { id name raw settings { strategy } }
  }
}
"""

_UPDATE_AUTOMATE_SESSION = """
mutation UpdateAutomateSession($id: ID!, $input: UpdateAutomateSessionInput!) {
  updateAutomateSession(id: $id, input: $input) {
    error {
      ... on PermissionDeniedUserError { code }
      ... on OtherUserError { code }
    }
    session { id name settings { placeholders { start end } strategy } }
  }
}
"""

_START_AUTOMATE_TASK = """
mutation StartAutomateTask($automateSessionId: ID!) {
  startAutomateTask(automateSessionId: $automateSessionId) {
    automateTask { id paused entry { id name } }
  }
}
"""

_AUTOMATE_SESSION_QUERY = """
query AutomateSession($id: ID!) {
  automateSession(id: $id) {
    id
    name
    raw
    entries { id name createdAt }
    settings { strategy placeholders { start end } }
  }
}
"""

_AUTOMATE_ENTRY_REQUESTS = """
query AutomateEntryRequests($id: ID!, $limit: Int, $offset: Int, $order: AutomateEntryRequestOrderInput) {
  automateEntry(id: $id) {
    id
    name
    requestsByOffset(limit: $limit, offset: $offset, order: $order) {
      count { value }
      nodes {
        sequenceId
        error
        payloads { position raw }
        request {
          id
          method
          host
          path
          query
          response {
            statusCode
            length
            roundtripTime
          }
        }
      }
    }
  }
}
"""

_AUTOMATE_TASKS_QUERY = """
query AutomateTasks {
  automateTasks(first: 50) {
    nodes {
      id
      paused
      entry { id name session { id name } }
    }
  }
}
"""

_SITEMAP_ROOT_QUERY = """
query SitemapRoot($scopeId: ID) {
  sitemapRootEntries(scopeId: $scopeId) {
    nodes {
      id
      label
      kind
      hasDescendants
      metadata { ... on SitemapEntryMetadataDomain { isTls port } }
    }
  }
}
"""

_SITEMAP_DESCENDANTS_QUERY = """
query SitemapDescendants($parentId: ID!, $depth: SitemapDescendantsDepth!) {
  sitemapDescendantEntries(parentId: $parentId, depth: $depth) {
    nodes {
      id
      label
      kind
      hasDescendants
      parentId
      metadata { ... on SitemapEntryMetadataDomain { isTls port } }
    }
  }
}
"""

_CREATE_REPLAY_SESSION = """
mutation CreateReplaySession($input: CreateReplaySessionInput!) {
  createReplaySession(input: $input) {
    session {
      id
      name
      activeEntry { id }
    }
  }
}
"""

_START_REPLAY_TASK = """
mutation StartReplayTask($sessionId: ID!, $input: StartReplayTaskInput!) {
  startReplayTask(sessionId: $sessionId, input: $input) {
    error {
      ... on TaskInProgressUserError { code }
      ... on OtherUserError { code }
    }
    task { id replayEntry { id } }
  }
}
"""

_REPLAY_ENTRY_QUERY = """
query ReplayEntry($id: ID!) {
  replayEntry(id: $id) {
    id
    raw
    error
    request {
      id
      method
      host
      path
      query
      response {
        statusCode
        length
        roundtripTime
        raw
      }
    }
  }
}
"""


class CaidoError(Exception):
    pass


@dataclass
class CaidoClient:
    """Thin GraphQL client for a local Caido instance.

    ``auth_token`` is a bearer access token (not a PAT).  Use
    ``scripts/caido_auth.py`` to exchange a PAT for an access token,
    or enable guest access on the instance to skip auth entirely.
    """
    url: str
    auth_token: str | None = None
    timeout: float = 30.0
    _http: httpx.AsyncClient | None = field(default=None, repr=False, init=False)

    @property
    def graphql_url(self) -> str:
        return f"{self.url.rstrip('/')}/graphql"

    async def _client(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            headers: dict[str, str] = {}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            self._http = httpx.AsyncClient(
                headers=headers,
                timeout=httpx.Timeout(self.timeout),
                verify=False,
            )
        return self._http

    async def execute(
        self,
        query: str,
        variables: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        client = await self._client()
        payload: dict[str, Any] = {"query": query}
        if variables:
            payload["variables"] = variables
        try:
            resp = await client.post(self.graphql_url, json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise CaidoError(f"HTTP error talking to Caido at {self.graphql_url}: {exc}") from exc
        body = resp.json()
        if "errors" in body and body["errors"]:
            msgs = "; ".join(e.get("message", str(e)) for e in body["errors"])
            raise CaidoError(f"GraphQL error: {msgs}")
        return body.get("data", {})

    async def close(self) -> None:
        if self._http and not self._http.is_closed:
            await self._http.aclose()


def _decode_blob(blob: str | None) -> str:
    """Decode a Caido Blob (base64-encoded) to UTF-8 text, lossy."""
    if not blob:
        return ""
    try:
        return base64.b64decode(blob).decode("utf-8", errors="replace")
    except Exception:
        return blob


def _find_placeholder_offsets(raw: str, target: str) -> list[tuple[int, int]]:
    """Find all occurrences of *target* in the raw HTTP request and return (start, end) byte pairs."""
    offsets: list[tuple[int, int]] = []
    start = 0
    while True:
        idx = raw.find(target, start)
        if idx == -1:
            break
        offsets.append((idx, idx + len(target)))
        start = idx + len(target)
    return offsets


def _format_request_node(node: dict[str, Any]) -> dict[str, Any]:
    resp = node.get("response") or {}
    return {
        "id": node["id"],
        "method": node.get("method", ""),
        "host": node.get("host", ""),
        "path": node.get("path", ""),
        "port": node.get("port"),
        "query": node.get("query", ""),
        "is_tls": node.get("isTls", False),
        "source": node.get("source", ""),
        "created_at": node.get("createdAt", ""),
        "status_code": resp.get("statusCode"),
        "response_length": resp.get("length"),
        "roundtrip_ms": resp.get("roundtripTime"),
    }


def caido_tools(
    name: str,
    *,
    caido_url: str = "http://127.0.0.1:8080",
    auth_token: str | None = None,
    timeout: float = 30.0,
) -> list[Any]:
    """Build Caido proxy interaction tools.

    Returns a list of async tool functions that share a single
    ``CaidoClient`` instance (same pattern as ``http_tools``).
    """
    cli = CaidoClient(url=caido_url, auth_token=auth_token, timeout=timeout)

    # ------------------------------------------------------------------
    # 1. caido_scope
    # ------------------------------------------------------------------
    async def caido_scope(
        action: Literal["list", "create"] = "list",
        name: str = "",
        allowlist: list[str] | None = None,
        denylist: list[str] | None = None,
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """Manage Caido proxy scopes (target allowlists).

        **Always set a scope before running automate** to prevent fuzzing
        non-target hosts.

        Args:
            action: "list" to view existing scopes, "create" to add one.
            name: scope name (required for "create").
            allowlist: glob patterns for allowed hosts, e.g. ["*.example.com", "api.target.io"].
            denylist: glob patterns to exclude.

        Returns:
            {"scopes": [...]} for list, or {"scope": {...}} for create.
        """
        try:
            if action == "list":
                data = await cli.execute(_SCOPES_QUERY)
                return {"scopes": data.get("scopes", [])}

            if not name:
                return {"error": "name is required when action='create'"}
            data = await cli.execute(
                _CREATE_SCOPE,
                variables={
                    "input": {
                        "name": name,
                        "allowlist": allowlist or [],
                        "denylist": denylist or [],
                    }
                },
            )
            result = data.get("createScope", {})
            if result.get("error"):
                return {"error": f"failed to create scope: {result['error']}"}
            return {"scope": result.get("scope")}
        except CaidoError as exc:
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # 2. caido_history
    # ------------------------------------------------------------------
    async def caido_history(
        filter: str = "",
        limit: int = 20,
        offset: int = 0,
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """Query Caido's HTTP proxy history.

        Returns requests that passed through the Caido proxy, newest first.
        Use HTTPQL filter syntax to narrow results.

        HTTPQL examples:
          - ``req.host.cont:"example.com"`` — host contains
          - ``resp.code.eq:200`` — status code equals
          - ``req.method.eq:"POST"`` — POST requests
          - ``req.path.cont:"/api/"`` — path contains
          - Combine with AND/OR: ``req.host.eq:"target" AND resp.code.gte:400``

        Args:
            filter: HTTPQL filter string (empty = all requests).
            limit: max results to return (1-100).
            offset: pagination offset.

        Returns:
            {"count": N, "requests": [{id, method, host, path, ...}, ...]}.
        """
        try:
            variables: dict[str, Any] = {
                "limit": min(max(limit, 1), 100),
                "offset": max(offset, 0),
                "order": {"by": "ID", "ordering": "DESC"},
            }
            if filter:
                variables["filter"] = filter
            data = await cli.execute(_REQUESTS_QUERY, variables=variables)
            conn = data.get("requestsByOffset", {})
            nodes = conn.get("nodes", [])
            return {
                "count": conn.get("count", {}).get("value", 0),
                "requests": [_format_request_node(n) for n in nodes],
            }
        except CaidoError as exc:
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # 3. caido_request_detail
    # ------------------------------------------------------------------
    async def caido_request_detail(
        request_id: str,
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """Get the full raw request and response for a Caido history entry.

        Use this to inspect an interesting request found via ``caido_history``
        before deciding what to fuzz. The ``raw`` fields contain the full
        HTTP request/response text.

        Args:
            request_id: the ID from caido_history results.

        Returns:
            Full request details including raw HTTP text and response.
        """
        try:
            data = await cli.execute(
                _REQUEST_DETAIL_QUERY,
                variables={"id": request_id},
            )
            req = data.get("request")
            if not req:
                return {"error": f"request {request_id} not found"}
            resp = req.get("response") or {}
            return {
                "id": req["id"],
                "method": req.get("method", ""),
                "host": req.get("host", ""),
                "path": req.get("path", ""),
                "port": req.get("port"),
                "query": req.get("query", ""),
                "is_tls": req.get("isTls", False),
                "raw": _decode_blob(req.get("raw")),
                "source": req.get("source", ""),
                "response": {
                    "id": resp.get("id"),
                    "status_code": resp.get("statusCode"),
                    "length": resp.get("length"),
                    "roundtrip_ms": resp.get("roundtripTime"),
                    "raw": _decode_blob(resp.get("raw")),
                } if resp else None,
            }
        except CaidoError as exc:
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # 4. caido_replay
    # ------------------------------------------------------------------
    async def caido_replay(
        request_id: str = "",
        raw_request: str = "",
        host: str = "",
        port: int = 80,
        is_tls: bool = False,
        wait: bool = True,
        timeout_seconds: float = 15.0,
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """Send an HTTP request through Caido's Replay feature.

        Either provide ``request_id`` to re-send an existing request from
        history, or provide ``raw_request`` + connection details to send
        a new one. The request and response appear in Caido's history.

        Args:
            request_id: ID of an existing request to replay (from caido_history).
            raw_request: raw HTTP request text (alternative to request_id).
                Example: "GET /api/users HTTP/1.1\\r\\nHost: target.com\\r\\n\\r\\n"
            host: target host (required when using raw_request).
            port: target port (default 80).
            is_tls: whether to use TLS/HTTPS.
            wait: if True, wait for the response before returning.
            timeout_seconds: max seconds to wait for response.

        Returns:
            Replay result with request/response details, or session info
            if wait=False.
        """
        try:
            # Resolve connection info and raw request (as base64 Blob)
            if request_id:
                detail_data = await cli.execute(
                    _REQUEST_DETAIL_QUERY,
                    variables={"id": request_id},
                )
                req = detail_data.get("request")
                if not req:
                    return {"error": f"request {request_id} not found"}
                raw_b64 = req.get("raw", "")
                conn_info = {
                    "host": req["host"],
                    "port": req["port"],
                    "isTLS": req.get("isTls", False),
                }
                source: dict[str, Any] = {"id": request_id}
            elif raw_request and host:
                raw_b64 = base64.b64encode(raw_request.encode()).decode()
                conn_info = {"host": host, "port": port, "isTLS": is_tls}
                source = {
                    "raw": {
                        "connectionInfo": conn_info,
                        "raw": raw_b64,
                    }
                }
            else:
                return {"error": "provide either request_id or (raw_request + host)"}

            # Create replay session
            data = await cli.execute(
                _CREATE_REPLAY_SESSION,
                variables={"input": {"requestSource": source}},
            )
            session = data.get("createReplaySession", {}).get("session")
            if not session:
                return {"error": "failed to create replay session"}
            session_id = session["id"]
            entry_id = (session.get("activeEntry") or {}).get("id")

            # Start the replay task (v0.55: requires input with raw + connection + settings)
            start_data = await cli.execute(
                _START_REPLAY_TASK,
                variables={
                    "sessionId": session_id,
                    "input": {
                        "raw": raw_b64,
                        "connection": conn_info,
                        "settings": {
                            "placeholders": [],
                            "updateContentLength": True,
                            "connectionClose": False,
                        },
                    },
                },
            )
            start_result = start_data.get("startReplayTask", {})
            if start_result.get("error"):
                return {"error": f"failed to start replay: {start_result['error']}"}
            task = start_result.get("task", {})
            replay_entry_id = (task.get("replayEntry") or {}).get("id") or entry_id

            if not wait or not replay_entry_id:
                return {
                    "session_id": session_id,
                    "entry_id": replay_entry_id,
                    "task_id": task.get("id"),
                    "status": "started",
                }

            deadline = asyncio.get_event_loop().time() + timeout_seconds
            while asyncio.get_event_loop().time() < deadline:
                await asyncio.sleep(0.5)
                entry_data = await cli.execute(
                    _REPLAY_ENTRY_QUERY,
                    variables={"id": replay_entry_id},
                )
                entry = entry_data.get("replayEntry")
                if not entry:
                    continue
                if entry.get("error"):
                    return {
                        "session_id": session_id,
                        "entry_id": replay_entry_id,
                        "error": entry["error"],
                    }
                req_node = entry.get("request")
                if req_node and req_node.get("response"):
                    resp = req_node["response"]
                    return {
                        "session_id": session_id,
                        "entry_id": replay_entry_id,
                        "request_id": req_node.get("id"),
                        "method": req_node.get("method", ""),
                        "host": req_node.get("host", ""),
                        "path": req_node.get("path", ""),
                        "status_code": resp.get("statusCode"),
                        "response_length": resp.get("length"),
                        "roundtrip_ms": resp.get("roundtripTime"),
                        "response_raw": _decode_blob(resp.get("raw")),
                    }

            return {
                "session_id": session_id,
                "entry_id": replay_entry_id,
                "status": "timeout",
                "message": f"no response within {timeout_seconds}s — query caido_request_detail later",
            }
        except CaidoError as exc:
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # 5. caido_automate_run
    # ------------------------------------------------------------------
    async def caido_automate_run(
        request_id: str,
        targets: list[str],
        payloads: list[str],
        strategy: CaidoStrategy = "ALL",
        workers: int = 5,
        delay_ms: int = 0,
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """Create and start a Caido Automate (fuzzing) session.

        Automate replaces marked positions in a request with payload values
        and sends all variants concurrently. This is much faster than
        issuing individual HTTP requests for parameter fuzzing.

        Workflow:
        1. Call ``caido_request_detail(request_id)`` first to see the raw
           request and pick target values.
        2. Call this tool with the target strings and payloads.
        3. Poll ``caido_automate_results(session_id)`` for results.

        Args:
            request_id: Caido request ID (from caido_history) to use as
                the fuzzing template.
            targets: literal substrings in the raw request to mark as
                injection points. Example: if the raw request contains
                ``id=42``, pass ``["42"]`` to fuzz that value.
                Only the **first occurrence** of each target is used.
            payloads: values to inject at each target position.
                Example: ``["' OR 1=1--", "1 UNION SELECT null--", "1;ls"]``.
            strategy: how to combine payloads across multiple targets:
                - "ALL" — try every payload in every position (default).
                - "SEQUENTIAL" — one payload list applied sequentially.
                - "PARALLEL" — payload[i] goes to target[i] in lockstep.
                - "MATRIX" — cartesian product of all payload lists.
            workers: number of concurrent requests (1-50).
            delay_ms: milliseconds delay between requests.

        Returns:
            ``{"session_id": ..., "entry_id": ..., "task_id": ...}`` on
            success. Use ``caido_automate_results(session_id)`` to fetch
            the results once the task completes.
        """
        try:
            # Fetch the original request to get raw content and connection info
            detail_data = await cli.execute(
                _REQUEST_DETAIL_QUERY,
                variables={"id": request_id},
            )
            req = detail_data.get("request")
            if not req:
                return {"error": f"request {request_id} not found in Caido history"}

            raw_b64: str = req.get("raw", "")
            if not raw_b64:
                return {"error": f"request {request_id} has no raw content"}

            raw_text = _decode_blob(raw_b64)

            # Compute placeholder offsets on the decoded raw text
            placeholders: list[dict[str, int]] = []
            for target in targets:
                offsets = _find_placeholder_offsets(raw_text, target)
                if not offsets:
                    return {
                        "error": f"target {target!r} not found in the raw request. "
                        f"Raw request starts with: {raw_text[:200]!r}"
                    }
                start, end = offsets[0]
                placeholders.append({"start": start, "end": end})

            # Create session from the request
            create_data = await cli.execute(
                _CREATE_AUTOMATE_SESSION,
                variables={"input": {"requestSource": {"id": request_id}}},
            )
            session = create_data.get("createAutomateSession", {}).get("session")
            if not session:
                return {"error": "failed to create automate session"}
            session_id = session["id"]

            # Configure the session with placeholders, payloads, and strategy
            settings: dict[str, Any] = {
                "closeConnection": False,
                "updateContentLength": True,
                "strategy": strategy,
                "concurrency": {
                    "workers": min(max(workers, 1), 50),
                    "delay": max(delay_ms, 0),
                },
                "placeholders": placeholders,
                "payloads": [
                    {
                        "options": {"simpleList": {"list": payloads}},
                        "preprocessors": [],
                    }
                ],
                "extractors": [],
                "redirect": {"strategy": "NEVER", "max": 0},
                "retryOnFailure": {"maximumRetries": 0, "backoff": 0},
            }

            update_data = await cli.execute(
                _UPDATE_AUTOMATE_SESSION,
                variables={
                    "id": session_id,
                    "input": {
                        "raw": raw_b64,
                        "connection": {
                            "host": req["host"],
                            "port": req["port"],
                            "isTLS": req.get("isTls", False),
                        },
                        "settings": settings,
                    },
                },
            )
            update_result = update_data.get("updateAutomateSession", {})
            if update_result.get("error"):
                return {"error": f"failed to configure automate session: {update_result['error']}"}

            # Start the fuzzing task
            start_data = await cli.execute(
                _START_AUTOMATE_TASK,
                variables={"automateSessionId": session_id},
            )
            task = start_data.get("startAutomateTask", {}).get("automateTask")
            entry = task.get("entry", {}) if task else {}

            return {
                "session_id": session_id,
                "task_id": task.get("id") if task else None,
                "entry_id": entry.get("id"),
                "entry_name": entry.get("name"),
                "targets": targets,
                "payload_count": len(payloads),
                "strategy": strategy,
                "status": "started",
                "message": "Fuzzing started. Use caido_automate_results(session_id) to check progress.",
            }
        except CaidoError as exc:
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # 6. caido_automate_results
    # ------------------------------------------------------------------
    async def caido_automate_results(
        session_id: str,
        entry_id: str = "",
        limit: int = 50,
        offset: int = 0,
        sort_by: str = "RESP_STATUS_CODE",
        ascending: bool = True,
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """Retrieve results from a Caido Automate (fuzzing) session.

        Call this after ``caido_automate_run`` to see which payloads
        triggered interesting responses. Look for anomalies in status
        codes, response lengths, and roundtrip times.

        Args:
            session_id: automate session ID from caido_automate_run.
            entry_id: specific entry ID (if empty, uses latest entry).
            limit: max results per page (1-100).
            offset: pagination offset.
            sort_by: field to sort by. Options: RESP_STATUS_CODE,
                RESP_LENGTH, RESP_ROUNDTRIP_TIME, POSITION, PAYLOAD_0.
            ascending: sort direction.

        Returns:
            Fuzzing results with payload values, status codes, and
            response metadata for each request sent.
        """
        try:
            if not entry_id:
                session_data = await cli.execute(
                    _AUTOMATE_SESSION_QUERY,
                    variables={"id": session_id},
                )
                session = session_data.get("automateSession")
                if not session:
                    return {"error": f"automate session {session_id} not found"}
                entries = session.get("entries", [])
                if not entries:
                    return {
                        "session_id": session_id,
                        "status": "no_entries",
                        "message": "No entries yet — the task may still be starting.",
                    }
                entry_id = entries[-1]["id"]

            data = await cli.execute(
                _AUTOMATE_ENTRY_REQUESTS,
                variables={
                    "id": entry_id,
                    "limit": min(max(limit, 1), 100),
                    "offset": max(offset, 0),
                    "order": {
                        "by": sort_by,
                        "ordering": "ASC" if ascending else "DESC",
                    },
                },
            )
            entry = data.get("automateEntry")
            if not entry:
                return {"error": f"automate entry {entry_id} not found"}

            conn = entry.get("requestsByOffset", {})
            total = conn.get("count", {}).get("value", 0)
            results: list[dict[str, Any]] = []
            for node in conn.get("nodes", []):
                req = node.get("request", {})
                resp = (req.get("response") or {})
                payload_values = [
                    p.get("raw", "") for p in node.get("payloads", [])
                ]
                results.append({
                    "sequence_id": node.get("sequenceId"),
                    "payloads": payload_values,
                    "error": node.get("error"),
                    "method": req.get("method", ""),
                    "host": req.get("host", ""),
                    "path": req.get("path", ""),
                    "query": req.get("query", ""),
                    "status_code": resp.get("statusCode"),
                    "response_length": resp.get("length"),
                    "roundtrip_ms": resp.get("roundtripTime"),
                    "request_id": req.get("id"),
                })

            return {
                "session_id": session_id,
                "entry_id": entry_id,
                "total_results": total,
                "offset": offset,
                "results": results,
            }
        except CaidoError as exc:
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # 7. caido_sitemap
    # ------------------------------------------------------------------
    async def caido_sitemap(
        parent_id: str = "",
        scope_id: str = "",
        depth: str = "DIRECT_CHILDREN",
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """Browse Caido's passive sitemap built from proxied traffic.

        The sitemap organizes discovered URLs into a tree by host and path
        segments. Call with no arguments to get root entries (hosts), then
        drill down by passing ``parent_id``.

        Note: the sitemap is populated from traffic that flows through the
        proxy. Send requests via http_request (routed through Caido proxy)
        or caido_replay first to populate it.

        Args:
            parent_id: ID of a parent entry to expand (empty = root).
            scope_id: limit to a specific scope (empty = all).
            depth: "DIRECT_CHILDREN" or "ALL_DESCENDANTS".

        Returns:
            {"entries": [{id, label, kind, has_children}, ...]}.
        """
        try:
            if parent_id:
                data = await cli.execute(
                    _SITEMAP_DESCENDANTS_QUERY,
                    variables={"parentId": parent_id, "depth": depth},
                )
                conn = data.get("sitemapDescendantEntries", {})
            else:
                variables: dict[str, Any] = {}
                if scope_id:
                    variables["scopeId"] = scope_id
                data = await cli.execute(_SITEMAP_ROOT_QUERY, variables=variables)
                conn = data.get("sitemapRootEntries", {})

            entries: list[dict[str, Any]] = []
            for node in conn.get("nodes", []):
                entries.append({
                    "id": node["id"],
                    "label": node.get("label", ""),
                    "kind": node.get("kind", ""),
                    "has_children": node.get("hasDescendants", False),
                    "parent_id": node.get("parentId"),
                })
            return {"entries": entries}
        except CaidoError as exc:
            return {"error": str(exc)}

    return [
        caido_scope,
        caido_history,
        caido_request_detail,
        caido_replay,
        caido_automate_run,
        caido_automate_results,
        caido_sitemap,
    ]
