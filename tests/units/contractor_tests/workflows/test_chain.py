"""Unit tests for the exploitability workflow's HTTP-chain collection.

Covers the opaque run-unique tag, raw-chain serialization, deterministic
Caido collection (cited-id path + full-sequence fallback), and the
anomaly-capture safety net.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml
from google.genai import types

import contractor.workflows.exploitability.workflow as exploit
from contractor.workflows.exploitability.workflow import (
    ExploitabilityWorkflow, _finding_tag_prefix, _serialize_chain)


def test_tag_prefix_is_opaque_and_deterministic():
    name = "sql-injection-in-login"
    p1 = _finding_tag_prefix(name)
    p2 = _finding_tag_prefix(name)
    # Deterministic: precalculated in code (no LLM, no randomness), so it is
    # reproducible and derivable independently. Cross-run staleness is handled
    # by since_ms at collection time, not by perturbing the prefix.
    assert p1 == p2
    assert p1.startswith("r")
    # Vuln identity must not leak into the live header value.
    assert "sql" not in p1 and "login" not in p1


def test_serialize_chain_renders_request_and_response():
    exchanges = [
        {
            "tag": "rABC-h000001",
            "source": "cited",
            "detail": {
                "method": "GET",
                "host": "t",
                "path": "/x",
                "is_tls": True,
                "raw": "GET /x HTTP/1.1\r\nHost: t\r\n\r\n",
                "response": {"raw": "HTTP/1.1 200 OK\r\n\r\nbody"},
            },
        }
    ]
    out = _serialize_chain(finding_name="f", exchanges=exchanges, fallback=False)
    assert "rABC-h000001" in out
    assert "[cited]" in out
    assert "https://t/x" in out
    assert "--- REQUEST ---" in out and "--- RESPONSE ---" in out
    assert "HTTP/1.1 200 OK" in out
    assert "full tagged sequence" not in out


def test_serialize_chain_notes_fallback():
    out = _serialize_chain(finding_name="f", exchanges=[], fallback=True)
    assert "full tagged sequence" in out


class FakeBackend:
    """Stands in for CaidoTools — maps filter->history, id->detail."""

    def __init__(self, history_map, detail_map):
        self.history_map = history_map
        self.detail_map = detail_map
        self.history_calls: list[str] = []

        async def _close():
            return None

        self.cli = SimpleNamespace(close=_close)

    async def history(self, filter="", limit=20, offset=0):
        self.history_calls.append(filter)
        return self.history_map.get(filter, {"requests": []})

    async def request_detail(self, request_id):
        return self.detail_map.get(request_id, {"error": "not found"})


def _node(id, status=200, length=100, created_at=10_000):
    return {
        "id": id,
        "status_code": status,
        "response_length": length,
        "created_at": created_at,
    }


def _detail(path, raw="r", resp=None):
    return {"method": "GET", "host": "t", "path": path, "raw": raw,
            "response": resp or {}}


def _bare_workflow(artifact_service):
    p = object.__new__(ExploitabilityWorkflow)
    p.caido_url = "http://caido"
    p.caido_auth_token = None
    p.ctx = SimpleNamespace(artifact_service=artifact_service, app_name="app")
    return p


@pytest.mark.asyncio
async def test_fetch_exchanges_proof_ids_dedup_and_order():
    tag = "rABC-h000001"
    flt = f'req.raw.cont:"X-Request-Id: {tag}"'
    seq = 'req.raw.cont:"X-Request-Id: rABC-"'
    backend = FakeBackend(
        history_map={flt: {"requests": [_node("2"), _node("1")]}},
        detail_map={"1": _detail("/a"), "2": _detail("/b")},
    )
    exchanges = await exploit._fetch_exchanges(
        backend, request_ids=[tag], tag_prefix="rABC"
    )
    # History is newest-first; chain should read oldest-first (id 1 then 2).
    assert [e["detail"]["path"] for e in exchanges] == ["/a", "/b"]
    assert all(e["source"] == "cited" for e in exchanges)
    # Full-sequence scan runs first (for anomaly detection), then the cited tag.
    assert backend.history_calls == [seq, flt]


@pytest.mark.asyncio
async def test_fetch_exchanges_includes_anomaly_not_cited():
    # Agent cites a benign 200; a 500 sits in the tagged sequence uncited.
    tag = "rABC-h000001"
    cited_flt = f'req.raw.cont:"X-Request-Id: {tag}"'
    seq = 'req.raw.cont:"X-Request-Id: rABC-"'
    backend = FakeBackend(
        history_map={
            seq: {"requests": [_node("9", 500, 44000), _node("1", 200, 100)]},
            cited_flt: {"requests": [_node("1", 200, 100)]},
        },
        detail_map={
            "1": _detail("/users/admin", resp={"raw": "HTTP/1.1 200 OK"}),
            "9": _detail("/users/admin'", resp={"raw": "HTTP/1.1 500 ERR\n\nSQL syntax"}),
        },
    )
    exchanges = await exploit._fetch_exchanges(
        backend, request_ids=[tag], tag_prefix="rABC"
    )
    by_path = {e["detail"]["path"]: e["source"] for e in exchanges}
    assert by_path["/users/admin"] == "cited"
    assert by_path["/users/admin'"] == "anomaly"  # 500 auto-added


@pytest.mark.asyncio
async def test_fetch_exchanges_since_ms_drops_stale_runs():
    # Two requests share the deterministic tag (a prior run + this run);
    # since_ms keeps only the one created during the current run.
    tag = "rABC-h000001"
    cited_flt = f'req.raw.cont:"X-Request-Id: {tag}"'
    seq = 'req.raw.cont:"X-Request-Id: rABC-"'
    stale = _node("1", created_at=1_000)   # earlier run
    fresh = _node("2", created_at=9_000)   # this run
    backend = FakeBackend(
        history_map={seq: {"requests": [fresh, stale]},
                     cited_flt: {"requests": [fresh, stale]}},
        detail_map={"1": _detail("/stale"), "2": _detail("/fresh")},
    )
    exchanges = await exploit._fetch_exchanges(
        backend, request_ids=[tag], tag_prefix="rABC", since_ms=5_000
    )
    assert [e["detail"]["path"] for e in exchanges] == ["/fresh"]


@pytest.mark.asyncio
async def test_fetch_exchanges_fallback_uses_prefix_filter():
    prefix = "rABC"
    seq = f'req.raw.cont:"X-Request-Id: {prefix}-"'
    backend = FakeBackend(
        history_map={seq: {"requests": [_node("1")]}},
        detail_map={"1": _detail("/a")},
    )
    exchanges = await exploit._fetch_exchanges(
        backend, request_ids=[], tag_prefix=prefix
    )
    assert len(exchanges) == 1
    assert exchanges[0]["source"] == "sequence"
    assert backend.history_calls == [seq]


class FakeArtifactService:
    def __init__(self, store=None):
        self.store = store or {}

    async def load_artifact(self, *, app_name, user_id, filename):
        return self.store.get(filename)

    async def save_artifact(self, *, app_name, user_id, filename, artifact):
        self.store[filename] = artifact


@pytest.mark.asyncio
async def test_collect_http_chain_writes_artifact(monkeypatch):
    finding = "finding-1"
    ns = f"exploitability:{finding}"
    prefix = _finding_tag_prefix(finding)
    tag = f"{prefix}-h000001"

    verification = {
        finding: {"verdict": "exploitable", "evidence_request_ids": [tag]}
    }
    svc = FakeArtifactService(
        {
            f"user:vulnerability-verifications/{ns}": types.Part.from_text(
                text=yaml.safe_dump(verification)
            )
        }
    )

    flt = f'req.raw.cont:"X-Request-Id: {tag}"'
    backend = FakeBackend(
        history_map={flt: {"requests": [_node("9")]}},
        detail_map={"9": _detail("/p", raw="raw-req", resp={"raw": "raw-resp"})},
    )
    monkeypatch.setattr(exploit, "CaidoClient", lambda **kw: object())
    monkeypatch.setattr(exploit, "CaidoTools", lambda client: backend)

    p = _bare_workflow(artifact_service=svc)
    await p._collect_http_chain(
        finding_name=finding,
        source_namespace=ns,
        tag_prefix=prefix,
        user_id="u",
    )

    chain = svc.store.get(f"user:exploit-http-chains/{finding}")
    assert chain is not None
    assert "raw-req" in chain.text and "raw-resp" in chain.text
    assert tag in chain.text


@pytest.mark.asyncio
async def test_collect_http_chain_noop_without_caido(monkeypatch):
    svc = FakeArtifactService()
    p = _bare_workflow(artifact_service=svc)
    p.caido_url = None
    await p._collect_http_chain(
        finding_name="f", source_namespace="ns", tag_prefix="r", user_id="u"
    )
    assert svc.store == {}
