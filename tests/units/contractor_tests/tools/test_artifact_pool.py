from __future__ import annotations

import pytest
import pytest_asyncio
import yaml
from google.adk.artifacts import FileArtifactService
from google.genai import types

from contractor.tools.artifact_pool import (
    ArtifactPool,
    KeywordPoolBackend,
    PoolKey,
    artifact_pool_tools,
)

APP = "contractor"
USER = "cli-user"


# ── PoolKey.parse / matches ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "key, namespace, kind",
    [
        ("user:memory/trace_annotation:openapi:users", "trace_annotation:openapi:users", "memory"),
        ("trace_annotation/openapi/users/result", "trace_annotation/openapi/users", "result"),
        ("oas/summary", "oas", "summary"),
        ("foo/bar/records", "foo/bar", "records"),
        ("oas-openapi-building", "oas-openapi-building", "raw"),
    ],
)
def test_parse_classifies_keyspace(key: str, namespace: str, kind: str):
    pk = PoolKey.parse(key)
    assert pk.namespace == namespace
    assert pk.kind == kind


def test_matches_namespace_first_then_raw():
    mem = PoolKey.parse("user:memory/trace_annotation:openapi:users")
    assert mem.matches("trace_annotation:*")  # namespace match
    rec = PoolKey.parse("foo/bar/records")
    assert rec.matches("*/records")  # raw fallback
    assert not rec.matches("trace_annotation:*")


# ── ArtifactPool over a real FileArtifactService ──────────────────────────────
# Uses FileArtifactService (not InMemory) because the pool relies on its
# session_id=None == user-scoped semantics, which is what the runner saves with.


async def _seed(svc: FileArtifactService) -> None:
    async def save(filename: str, text: str) -> None:
        await svc.save_artifact(
            app_name=APP,
            user_id=USER,
            session_id=None,
            filename=filename,
            artifact=types.Part.from_text(text=text),
        )

    notes = {
        "login_handler": {
            "name": "login_handler",
            "memory": "POST /login authenticates the user via JWT",
            "description": "auth entrypoint",
            "tags": ["authentication"],
        },
        "orders_bola": {
            "name": "orders_bola",
            "memory": "GET /orders/{id} has no ownership check (BOLA)",
            "description": "broken object level auth",
            "tags": ["bola"],
        },
        # Injected reference body + cross-task plumbing — must be excluded from
        # documents()/search (they are not run-specific knowledge).
        "trace/references/sinks": {
            "name": "trace/references/sinks",
            "memory": "SQL injection BOLA JWT — a big reference dump of keywords",
            "description": "skill body",
            "tags": ["skill"],
        },
        "previous-task-result": {
            "name": "previous-task-result",
            "memory": "upstream BOLA result injected as inbox",
            "description": "inbox",
            "tags": ["inbox"],
        },
    }
    await save("user:memory/trace_annotation:api", yaml.safe_dump(notes))
    await save("trace_annotation/api/result", "BOLA on /orders/{id}: no ownership check")
    await save("oas-openapi-building", "openapi: 3.0.0 paths: /login")


@pytest_asyncio.fixture()
async def pool(tmp_path) -> ArtifactPool:
    svc = FileArtifactService(root_dir=str(tmp_path))
    await _seed(svc)
    return ArtifactPool(artifact_service=svc, app_name=APP, user_id=USER)


@pytest.mark.asyncio
async def test_keys_lists_all_kinds(pool: ArtifactPool):
    kinds = {pk.kind for pk in await pool.keys()}
    assert {"memory", "result", "raw"} <= kinds


@pytest.mark.asyncio
async def test_masks_fence_visibility(tmp_path):
    svc = FileArtifactService(root_dir=str(tmp_path))
    await _seed(svc)
    fenced = ArtifactPool(
        artifact_service=svc, app_name=APP, user_id=USER, masks=("oas-*",)
    )
    keys = await fenced.keys()
    assert [pk.raw for pk in keys] == ["oas-openapi-building"]
    # A masked-out artifact cannot be read either.
    assert await fenced.load_text("trace_annotation/api/result") is None


@pytest.mark.asyncio
async def test_load_notes_parses_memory_store(pool: ArtifactPool):
    notes = await pool.load_notes("trace_annotation:api")
    assert "login_handler" in notes
    assert "JWT" in notes["login_handler"]["memory"]


@pytest.mark.asyncio
async def test_keyword_backend_ranks_and_snippets(pool: ArtifactPool):
    hits = await KeywordPoolBackend().search(pool, "BOLA ownership", mask="*", k=5)
    assert hits
    assert hits[0].namespace == "trace_annotation/api"
    assert "BOLA" in hits[0].snippet


@pytest.mark.asyncio
async def test_documents_expand_per_note_and_drop_reserved(pool: ArtifactPool):
    docs = await pool.documents()
    mem = [d for d in docs if d.kind == "memory"]
    # The store has 4 notes but 2 are skill/inbox — only the 2 real ones expand.
    assert {d.note_name for d in mem} == {"login_handler", "orders_bola"}
    # include_reserved brings the skill/inbox notes back.
    with_reserved = await pool.documents(include_reserved=True)
    assert any(d.note_name == "trace/references/sinks" for d in with_reserved)


@pytest.mark.asyncio
async def test_search_skips_reserved_notes_and_carries_note_name(pool: ArtifactPool):
    # The skill note is keyword-stuffed with BOLA/JWT/SQL but must never surface.
    hits = await KeywordPoolBackend().search(pool, "JWT authenticates", mask="*", k=5)
    assert all(h.note_name != "trace/references/sinks" for h in hits)
    mem_hits = [h for h in hits if h.kind == "memory"]
    assert mem_hits and mem_hits[0].note_name == "login_handler"


# ── Frontend tools envelope ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tools_return_envelopes(tmp_path):
    svc = FileArtifactService(root_dir=str(tmp_path))
    await _seed(svc)
    tools = {
        t.__name__: t
        for t in artifact_pool_tools(
            artifact_service=svc, app_name=APP, user_id=USER
        )
    }

    ns = await tools["pool_namespaces"](tool_context=None)
    assert "result" in ns and ns["total_items"] >= 3

    read = await tools["pool_read"](
        key="trace_annotation/api/result", offset=0, limit=0, tool_context=None
    )
    assert "BOLA" in read["result"]

    missing = await tools["pool_read"](
        key="does/not/exist", offset=0, limit=0, tool_context=None
    )
    assert "error" in missing

    mem = await tools["pool_read_memory"](
        namespace="trace_annotation:api", name="login_handler", tool_context=None
    )
    assert mem["result"]["name"] == "login_handler"
