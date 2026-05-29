"""Integration tests for Caido proxy tools.

Requires a running Caido instance. Tests are skipped with a warning
when Caido is unreachable or auth is not configured.

Run:
    poetry run pytest tests/integrational/contractor/tools/test_caido.py -v
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from contractor.tools.caido import CaidoClient, CaidoError, CaidoTools

# ---------------------------------------------------------------------------
# Env / settings helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _load_settings() -> tuple[str | None, str | None]:
    """Read CAIDO_URL and CAIDO_AUTH_TOKEN from env vars / cli/.env."""
    import os

    url = os.environ.get("CAIDO_URL")
    token = os.environ.get("CAIDO_AUTH_TOKEN")

    env_path = _REPO_ROOT / "cli" / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if key == "CAIDO_URL" and not url:
                url = value
            elif key == "CAIDO_AUTH_TOKEN" and not token:
                token = value
    return url, token


CAIDO_URL, CAIDO_AUTH_TOKEN = _load_settings()


def _caido_reachable() -> bool:
    if not CAIDO_URL:
        return False
    try:
        resp = httpx.get(f"{CAIDO_URL}/health", timeout=3, verify=False)
        return resp.status_code == 200
    except Exception:
        return False


_reachable = _caido_reachable()

skip_reason = "Caido not reachable" if not _reachable else ""
if _reachable and not CAIDO_AUTH_TOKEN:
    # Check if guests are allowed
    try:
        resp = httpx.post(
            f"{CAIDO_URL}/graphql",
            json={"query": "{ scopes { id } }"},
            timeout=5,
            verify=False,
        )
        body = resp.json()
        errors = body.get("errors", [])
        auth_err = any(
            e.get("extensions", {}).get("CAIDO", {}).get("code") == "AUTHORIZATION"
            for e in errors
        )
        if auth_err:
            skip_reason = "Caido requires auth but CAIDO_AUTH_TOKEN not set in cli/.env"
    except Exception:
        skip_reason = "Caido auth check failed"

if not _reachable:
    pytest.skip(
        "Caido proxy not running — skipping integration tests",
        allow_module_level=True,
    )
elif skip_reason:
    pytest.skip(skip_reason, allow_module_level=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client() -> CaidoClient:
    return CaidoClient(url=CAIDO_URL, auth_token=CAIDO_AUTH_TOKEN)


@pytest.fixture(scope="module")
def backend(client: CaidoClient) -> CaidoTools:
    # Tests exercise the backend directly (raw bare/error dicts); the
    # agent-facing caido_tools() wraps these methods with aguard().
    return CaidoTools(client)


# Convenience: bind individual backend methods
@pytest.fixture(scope="module")
def scope_tool(backend):
    return backend.scope


@pytest.fixture(scope="module")
def history_tool(backend):
    return backend.history


@pytest.fixture(scope="module")
def detail_tool(backend):
    return backend.request_detail


@pytest.fixture(scope="module")
def replay_tool(backend):
    return backend.replay


@pytest.fixture(scope="module")
def automate_run_tool(backend):
    return backend.automate_run


@pytest.fixture(scope="module")
def automate_results_tool(backend):
    return backend.automate_results


@pytest.fixture(scope="module")
def sitemap_tool(backend):
    return backend.sitemap


# ---------------------------------------------------------------------------
# CaidoClient tests
# ---------------------------------------------------------------------------


class TestCaidoClient:
    @pytest.mark.asyncio
    async def test_execute_simple_query(self, client: CaidoClient):
        data = await client.execute("{ scopes { id name } }")
        assert "scopes" in data

    @pytest.mark.asyncio
    async def test_execute_bad_query_raises(self, client: CaidoClient):
        with pytest.raises(CaidoError, match="GraphQL error"):
            await client.execute("{ nonExistentField }")


# ---------------------------------------------------------------------------
# caido_scope
# ---------------------------------------------------------------------------


class TestCaidoScope:
    @pytest.mark.asyncio
    async def test_list_scopes(self, scope_tool):
        result = await scope_tool(action="list")
        assert "scopes" in result
        assert isinstance(result["scopes"], list)

    @pytest.mark.asyncio
    async def test_create_scope(self, scope_tool):
        import time

        name = f"test-scope-{int(time.time())}"
        result = await scope_tool(
            action="create",
            name=name,
            allowlist=["*.test-target.local"],
        )
        assert "error" not in result, result
        assert result["scope"]["name"] == name
        assert "*.test-target.local" in result["scope"]["allowlist"]

    @pytest.mark.asyncio
    async def test_create_scope_missing_name(self, scope_tool):
        result = await scope_tool(action="create")
        assert "error" in result


# ---------------------------------------------------------------------------
# caido_history
# ---------------------------------------------------------------------------


class TestCaidoHistory:
    @pytest.mark.asyncio
    async def test_history_returns_count_and_requests(self, history_tool):
        result = await history_tool(limit=5)
        assert "count" in result
        assert "requests" in result
        assert isinstance(result["requests"], list)
        assert len(result["requests"]) <= 5

    @pytest.mark.asyncio
    async def test_history_with_filter(self, history_tool):
        result = await history_tool(filter='req.method.eq:"GET"', limit=3)
        assert "error" not in result, result
        for req in result["requests"]:
            assert req["method"] == "GET"

    @pytest.mark.asyncio
    async def test_history_request_shape(self, history_tool):
        result = await history_tool(limit=1)
        if result["requests"]:
            req = result["requests"][0]
            assert "id" in req
            assert "method" in req
            assert "host" in req
            assert "path" in req
            assert "status_code" in req

    @pytest.mark.asyncio
    async def test_history_pagination(self, history_tool):
        page1 = await history_tool(limit=2, offset=0)
        page2 = await history_tool(limit=2, offset=2)
        assert "error" not in page1
        assert "error" not in page2
        ids1 = {r["id"] for r in page1["requests"]}
        ids2 = {r["id"] for r in page2["requests"]}
        assert ids1.isdisjoint(ids2), "pages should not overlap"


# ---------------------------------------------------------------------------
# caido_replay
# ---------------------------------------------------------------------------


class TestCaidoReplay:
    @pytest.mark.asyncio
    async def test_replay_raw_request(self, replay_tool):
        result = await replay_tool(
            raw_request="GET /get?test=caido HTTP/1.1\r\nHost: httpbin.org\r\nAccept: */*\r\n\r\n",
            host="httpbin.org",
            port=443,
            is_tls=True,
            timeout_seconds=20,
        )
        assert "error" not in result, result
        assert result["status_code"] == 200
        assert result["response_length"] > 0
        assert result["roundtrip_ms"] > 0
        assert "httpbin.org" in result["response_raw"]

    @pytest.mark.asyncio
    async def test_replay_by_request_id(self, replay_tool, history_tool):
        hist = await history_tool(limit=1)
        assert hist["requests"], "need at least one request in history"
        rid = hist["requests"][0]["id"]

        result = await replay_tool(request_id=rid, timeout_seconds=20)
        # May fail if original target is down — that's ok, just check structure
        assert "session_id" in result
        assert "entry_id" in result

    @pytest.mark.asyncio
    async def test_replay_missing_args(self, replay_tool):
        result = await replay_tool()
        assert "error" in result

    @pytest.mark.asyncio
    async def test_replay_nowait(self, replay_tool):
        result = await replay_tool(
            raw_request="GET /get HTTP/1.1\r\nHost: httpbin.org\r\n\r\n",
            host="httpbin.org",
            port=443,
            is_tls=True,
            wait=False,
        )
        assert "error" not in result, result
        assert result["status"] == "started"
        assert "session_id" in result


# ---------------------------------------------------------------------------
# caido_request_detail
# ---------------------------------------------------------------------------


class TestCaidoRequestDetail:
    @pytest.mark.asyncio
    async def test_detail_decodes_raw(self, detail_tool, replay_tool):
        r = await replay_tool(
            raw_request="GET /headers HTTP/1.1\r\nHost: httpbin.org\r\n\r\n",
            host="httpbin.org",
            port=443,
            is_tls=True,
            timeout_seconds=20,
        )
        assert r.get("request_id"), f"replay failed: {r}"

        detail = await detail_tool(request_id=r["request_id"])
        assert "error" not in detail, detail
        assert "GET /headers" in detail["raw"]
        assert detail["host"] == "httpbin.org"
        assert detail["response"] is not None
        assert detail["response"]["status_code"] == 200
        assert "HTTP/1.1 200" in detail["response"]["raw"]

    @pytest.mark.asyncio
    async def test_detail_not_found(self, detail_tool):
        result = await detail_tool(request_id="99999999")
        assert "error" in result


# ---------------------------------------------------------------------------
# caido_sitemap
# ---------------------------------------------------------------------------


class TestCaidoSitemap:
    @pytest.mark.asyncio
    async def test_sitemap_root(self, sitemap_tool):
        result = await sitemap_tool()
        assert "entries" in result
        assert isinstance(result["entries"], list)

    @pytest.mark.asyncio
    async def test_sitemap_entry_shape(self, sitemap_tool):
        result = await sitemap_tool()
        if result["entries"]:
            entry = result["entries"][0]
            assert "id" in entry
            assert "label" in entry
            assert "kind" in entry
            assert "has_children" in entry

    @pytest.mark.asyncio
    async def test_sitemap_drill_down(self, sitemap_tool):
        root = await sitemap_tool()
        parents = [e for e in root.get("entries", []) if e.get("has_children")]
        if not parents:
            pytest.skip("no sitemap entries with children")
        children = await sitemap_tool(parent_id=parents[0]["id"])
        assert "error" not in children, children
        assert isinstance(children["entries"], list)


# ---------------------------------------------------------------------------
# caido_automate (run + results)
# ---------------------------------------------------------------------------


class TestCaidoAutomate:
    @pytest_asyncio.fixture
    async def base_request_id(self, replay_tool):
        """Send a base request to httpbin that we can fuzz."""
        result = await replay_tool(
            raw_request="GET /get?id=1 HTTP/1.1\r\nHost: httpbin.org\r\nAccept: */*\r\n\r\n",
            host="httpbin.org",
            port=443,
            is_tls=True,
            timeout_seconds=20,
        )
        rid = result.get("request_id")
        assert rid, f"failed to create base request: {result}"
        return rid

    @pytest.mark.asyncio
    async def test_automate_run_and_results(
        self, automate_run_tool, automate_results_tool, base_request_id
    ):
        fuzz = await automate_run_tool(
            request_id=base_request_id,
            targets=["1"],
            payloads=["1", "2", "test"],
            strategy="ALL",
            workers=2,
            delay_ms=100,
        )
        assert "error" not in fuzz, fuzz
        assert fuzz["status"] == "started"
        assert fuzz["session_id"]
        assert fuzz["payload_count"] == 3

        await asyncio.sleep(10)

        results = await automate_results_tool(
            session_id=fuzz["session_id"], limit=10
        )
        assert "error" not in results, results
        assert results["total_results"] == 3
        assert len(results["results"]) == 3

        for entry in results["results"]:
            assert entry["status_code"] == 200
            assert entry["response_length"] > 0
            assert len(entry["payloads"]) == 1
            assert entry["payloads"][0] in ("1", "2", "test")

    @pytest.mark.asyncio
    async def test_automate_target_not_found(
        self, automate_run_tool, base_request_id
    ):
        result = await automate_run_tool(
            request_id=base_request_id,
            targets=["NONEXISTENT_VALUE_xyz"],
            payloads=["a"],
        )
        assert "error" in result
        assert "not found in the raw request" in result["error"]

    @pytest.mark.asyncio
    async def test_automate_bad_request_id(self, automate_run_tool):
        result = await automate_run_tool(
            request_id="99999999",
            targets=["x"],
            payloads=["y"],
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_automate_results_no_session(self, automate_results_tool):
        result = await automate_results_tool(session_id="99999999")
        assert "error" in result
