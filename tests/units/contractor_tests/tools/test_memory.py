from __future__ import annotations

import pytest
import yaml

from contractor.tools.memory import (
    MemoryFormat,
    MemoryNote,
    MemoryTools,
    _resolve_skill_reference,
)

# ---------------------------------------------------------------------------
# Mock ctx with artifact load/save support
# ---------------------------------------------------------------------------


class _Artifact:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeArtifactCtx:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def load_artifact(self, *, filename: str) -> _Artifact | None:
        text = self.store.get(filename)
        if text is None:
            return None
        return _Artifact(text)

    async def save_artifact(self, *, filename: str, artifact) -> None:
        # The real artifact is types.Part.from_text(text=...). We pull text out.
        text = getattr(artifact, "text", None)
        if text is None and hasattr(artifact, "to_dict"):
            text = artifact.to_dict().get("text", "")
        self.store[filename] = text or ""


# ---------------------------------------------------------------------------
# MemoryFormat — pure formatting tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def note() -> MemoryNote:
    return MemoryNote(
        name="finding",
        memory="vulnerable handler at /api/x",
        description="auth bypass",
        tags=["security", "auth"],
        ordinal=3,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-02T00:00:00+00:00",
    )


def test_format_memory_json(note: MemoryNote):
    fmt = MemoryFormat(_format="json")
    out = fmt.format_memory(note)
    assert isinstance(out, dict)
    assert out["name"] == "finding"
    assert out["memory"] == "vulnerable handler at /api/x"
    assert out["tags"] == ["security", "auth"]


def test_format_memory_yaml_round_trips(note: MemoryNote):
    fmt = MemoryFormat(_format="yaml")
    out = fmt.format_memory(note)
    assert isinstance(out, str)
    parsed = yaml.safe_load(out)
    # YAML output wraps the note under a "memory_<name>" key.
    inner = parsed["memory_finding"]
    assert inner["name"] == "finding"
    assert inner["memory"] == "vulnerable handler at /api/x"


def test_format_memory_markdown_includes_fields(note: MemoryNote):
    fmt = MemoryFormat(_format="markdown")
    out = fmt.format_memory(note)
    assert isinstance(out, str)
    assert "finding" in out
    assert "auth bypass" in out


def test_format_memory_xml_escapes_special_characters():
    note = MemoryNote(
        name="weird",
        memory="<script>alert('x')</script>",
        description="injection & such",
    )
    fmt = MemoryFormat(_format="xml")
    out = fmt.format_memory(note)
    assert isinstance(out, str)
    assert "<script>" not in out  # raw tags must be escaped
    assert "&lt;script&gt;" in out
    assert "&amp;" in out


def test_format_memory_preview_omits_body(note: MemoryNote):
    fmt = MemoryFormat(_format="json")
    out = fmt.format_memory_preview(note)
    assert "memory" not in out  # preview hides the body
    assert out["name"] == "finding"
    assert out["description"] == "auth bypass"


def test_format_memories_xml_wraps_in_root(note: MemoryNote):
    fmt = MemoryFormat(_format="xml")
    out = fmt.format_memories([note, note])
    assert isinstance(out, str)
    assert out.startswith("<memories>")
    assert out.endswith("</memories>")


def test_format_tags_json():
    fmt = MemoryFormat(_format="json")
    out = fmt.format_tags(["a", "b"])
    assert out == {"tags": ["a", "b"]}


def test_format_tags_markdown_empty_list():
    fmt = MemoryFormat(_format="markdown")
    out = fmt.format_tags([])
    assert isinstance(out, str)
    assert "none" in out


def test_format_tags_xml_self_closing_when_empty():
    fmt = MemoryFormat(_format="xml")
    out = fmt.format_tags([])
    assert out == "<tags />"


def test_type_hint_wraps_string_output():
    fmt = MemoryFormat(_format="markdown")
    note = MemoryNote(name="n", memory="m", description="d")
    out = fmt.format_memory(note, type_hint=True)
    assert isinstance(out, str)
    assert out.startswith("```markdown")
    assert out.endswith("```")


def test_type_hint_does_not_wrap_dict_output():
    fmt = MemoryFormat(_format="json")
    note = MemoryNote(name="n", memory="m", description="d")
    out = fmt.format_memory(note, type_hint=True)
    assert isinstance(out, dict)


# ---------------------------------------------------------------------------
# MemoryTools — write / read / append cycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_then_read_round_trip():
    tools = MemoryTools(name="agent")
    ctx = FakeArtifactCtx()

    await tools.write_memory(
        name="n1",
        memory="content",
        description="desc",
        tags=["a"],
        ctx=ctx,
    )

    note = await tools.read_memory("n1", ctx)
    assert note is not None
    assert note.memory == "content"
    assert note.tags == ["a"]
    assert note.ordinal == 1
    assert note.created_at and note.updated_at


@pytest.mark.asyncio
async def test_read_missing_returns_none():
    tools = MemoryTools(name="agent")
    ctx = FakeArtifactCtx()

    assert await tools.read_memory("absent", ctx) is None


@pytest.mark.asyncio
async def test_write_truncates_tags_to_three():
    tools = MemoryTools(name="agent")
    ctx = FakeArtifactCtx()

    await tools.write_memory(
        name="n",
        memory="m",
        description="d",
        tags=["t1", "t2", "t3", "t4", "t5"],
        ctx=ctx,
    )
    note = await tools.read_memory("n", ctx)
    assert note is not None
    assert note.tags == ["t1", "t2", "t3"]


@pytest.mark.asyncio
async def test_write_overwrites_preserves_ordinal_and_created_at():
    tools = MemoryTools(name="agent")
    ctx = FakeArtifactCtx()

    await tools.write_memory(
        name="n", memory="v1", description="d", tags=None, ctx=ctx
    )
    first = await tools.read_memory("n", ctx)
    assert first is not None

    await tools.write_memory(
        name="n", memory="v2", description="d2", tags=None, ctx=ctx
    )
    second = await tools.read_memory("n", ctx)

    assert second is not None
    assert second.memory == "v2"
    assert second.description == "d2"
    assert second.ordinal == first.ordinal
    assert second.created_at == first.created_at


@pytest.mark.asyncio
async def test_append_concatenates_with_newline():
    tools = MemoryTools(name="agent")
    ctx = FakeArtifactCtx()

    await tools.write_memory(
        name="n", memory="line1", description="d", tags=None, ctx=ctx
    )
    appended = await tools.append_memory("n", "line2", ctx)
    assert appended is not None
    assert appended.memory == "line1\nline2"


@pytest.mark.asyncio
async def test_append_returns_none_when_target_missing():
    tools = MemoryTools(name="agent")
    ctx = FakeArtifactCtx()

    result = await tools.append_memory("absent", "x", ctx)
    assert result is None


@pytest.mark.asyncio
async def test_ordinal_increments_for_new_memories():
    tools = MemoryTools(name="agent")
    ctx = FakeArtifactCtx()

    await tools.write_memory(name="a", memory="m", description="d", tags=None, ctx=ctx)
    await tools.write_memory(name="b", memory="m", description="d", tags=None, ctx=ctx)
    await tools.write_memory(name="c", memory="m", description="d", tags=None, ctx=ctx)

    notes = await tools.list_memories(ctx)
    ordinals = [n.ordinal for n in notes]
    assert ordinals == [1, 2, 3]


# ---------------------------------------------------------------------------
# MemoryTools — listing, search, tag filters
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_memories_sorts_by_ordinal_then_name():
    tools = MemoryTools(name="agent")
    ctx = FakeArtifactCtx()

    await tools.write_memory(name="zeta", memory="m", description="d", tags=None, ctx=ctx)
    await tools.write_memory(name="alpha", memory="m", description="d", tags=None, ctx=ctx)

    notes = await tools.list_memories(ctx)
    names = [n.name for n in notes]
    # ordinal 1 was assigned first → 'zeta', ordinal 2 → 'alpha'
    assert names == ["zeta", "alpha"]


@pytest.mark.asyncio
async def test_list_tags_returns_unique_sorted():
    tools = MemoryTools(name="agent")
    ctx = FakeArtifactCtx()

    await tools.write_memory(
        name="a", memory="m", description="d", tags=["x", "y"], ctx=ctx
    )
    await tools.write_memory(
        name="b", memory="m", description="d", tags=["y", "z"], ctx=ctx
    )

    tags = await tools.list_tags(ctx)
    assert tags == ["x", "y", "z"]


@pytest.mark.asyncio
async def test_search_memory_returns_any_matching_tag():
    tools = MemoryTools(name="agent")
    ctx = FakeArtifactCtx()

    await tools.write_memory(
        name="a", memory="m", description="d", tags=["red"], ctx=ctx
    )
    await tools.write_memory(
        name="b", memory="m", description="d", tags=["blue"], ctx=ctx
    )
    await tools.write_memory(
        name="c", memory="m", description="d", tags=["red", "green"], ctx=ctx
    )

    matches = await tools.search_memory(["red"], ctx)
    assert {m.name for m in matches} == {"a", "c"}


@pytest.mark.asyncio
async def test_reserved_tags_hidden_from_generic_surface():
    """skill/inbox notes are reachable only via their dedicated tools."""
    tools = MemoryTools(name="agent")
    ctx = FakeArtifactCtx()

    await tools.write_memory(name="note", memory="m", description="d",
                             tags=["topic"], ctx=ctx)
    await tools.write_memory(name="trace/refs/sinks", memory="SKILL BODY",
                             description="d", tags=["skill", "trace"], ctx=ctx)
    await tools.write_memory(name="prev-task", memory="INBOX", description="d",
                             tags=["inbox"], ctx=ctx)

    # list_memories excludes reserved
    assert {n.name for n in await tools.list_memories(ctx)} == {"note"}

    # read_memory hard-refuses reserved (treated as absent)
    assert await tools.read_memory("note", ctx) is not None
    assert await tools.read_memory("trace/refs/sinks", ctx) is None
    assert await tools.read_memory("prev-task", ctx) is None

    # search_memory excludes reserved notes AND drops reserved tags from query
    await tools.write_memory(name="note2", memory="m", description="d",
                             tags=["topic"], ctx=ctx)
    assert {n.name for n in await tools.search_memory(["topic"], ctx)} == {"note", "note2"}
    assert await tools.search_memory(["skill"], ctx) == []
    # a query mixing reserved + real tag still excludes the reserved skill note
    assert {n.name for n in await tools.search_memory(["skill", "trace"], ctx)} == set()

    # dedicated paths STILL reach reserved notes (skills_read / inbox_read)
    assert (await tools.read_memory_by_tag("trace/refs/sinks", "skill", ctx)).memory == "SKILL BODY"
    assert (await tools.read_memory_by_tag("prev-task", "inbox", ctx)).memory == "INBOX"
    assert {n.name for n in await tools.memories_by_tag("skill", ctx)} == {"trace/refs/sinks"}


@pytest.mark.asyncio
async def test_memories_by_tag_filters_by_single_tag():
    tools = MemoryTools(name="agent")
    ctx = FakeArtifactCtx()

    await tools.write_memory(
        name="a", memory="m", description="d", tags=["x"], ctx=ctx
    )
    await tools.write_memory(
        name="b", memory="m", description="d", tags=["y"], ctx=ctx
    )

    matches = await tools.memories_by_tag("x", ctx)
    assert [m.name for m in matches] == ["a"]


@pytest.mark.asyncio
async def test_read_memory_by_tag_requires_tag_match():
    tools = MemoryTools(name="agent")
    ctx = FakeArtifactCtx()

    await tools.write_memory(
        name="n", memory="m", description="d", tags=["t"], ctx=ctx
    )

    assert (await tools.read_memory_by_tag("n", "t", ctx)) is not None
    assert (await tools.read_memory_by_tag("n", "other", ctx)) is None
    assert (await tools.read_memory_by_tag("absent", "t", ctx)) is None


# ---------------------------------------------------------------------------
# MemoryTools — persistence: dump / load / cross-instance read
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persistence_across_instances():
    ctx = FakeArtifactCtx()

    writer = MemoryTools(name="agent")
    await writer.write_memory(
        name="n", memory="payload", description="d", tags=["t"], ctx=ctx
    )

    reader = MemoryTools(name="agent")
    note = await reader.read_memory("n", ctx)

    assert note is not None
    assert note.memory == "payload"
    assert note.tags == ["t"]


@pytest.mark.asyncio
async def test_memory_key_is_namespaced_by_name():
    ctx = FakeArtifactCtx()

    a = MemoryTools(name="alice")
    b = MemoryTools(name="bob")

    await a.write_memory(name="n", memory="A", description="d", tags=None, ctx=ctx)
    await b.write_memory(name="n", memory="B", description="d", tags=None, ctx=ctx)

    note_a = await a.read_memory("n", ctx)
    note_b = await b.read_memory("n", ctx)
    assert note_a is not None and note_a.memory == "A"
    assert note_b is not None and note_b.memory == "B"
    assert "alice" in a.memory_key()
    assert "bob" in b.memory_key()


# ---------------------------------------------------------------------------
# _resolve_skill_reference — tolerant skill-reference name matching
# ---------------------------------------------------------------------------


def _skill_note(name: str) -> MemoryNote:
    return MemoryNote(name=name, memory="m", description="d", tags=["skill"])


@pytest.fixture()
def trace_skill_notes() -> list[MemoryNote]:
    return [
        _skill_note("trace"),
        _skill_note("trace/references/sinks"),
        _skill_note("trace/references/sources"),
    ]


@pytest.mark.parametrize(
    "query",
    [
        "trace/references/sinks",  # canonical stored form
        "references/sinks",  # bare form copied from index tables
        "references/sinks.md",  # bare form with extension
        "sinks",  # basename only
        "sinks.md",
    ],
)
def test_resolve_skill_reference_tolerates_citation_forms(trace_skill_notes, query):
    note = _resolve_skill_reference(query, trace_skill_notes)
    assert note is not None
    assert note.name == "trace/references/sinks"


def test_resolve_skill_reference_returns_none_when_absent(trace_skill_notes):
    assert _resolve_skill_reference("references/nope", trace_skill_notes) is None


def test_resolve_skill_reference_returns_none_when_ambiguous():
    # idor lives under two loaded skills → basename match is ambiguous, bail out.
    notes = [
        _skill_note("exploit/references/idor"),
        _skill_note("vulns/references/idor"),
    ]
    assert _resolve_skill_reference("idor", notes) is None
    # but a fully-qualified form still resolves unambiguously.
    resolved = _resolve_skill_reference("vulns/references/idor", notes)
    assert resolved is not None and resolved.name == "vulns/references/idor"


@pytest.mark.asyncio
async def test_dump_yaml_preserves_ordinal_order():
    tools = MemoryTools(name="agent")
    ctx = FakeArtifactCtx()

    await tools.write_memory(name="a", memory="m", description="d", tags=None, ctx=ctx)
    await tools.write_memory(name="b", memory="m", description="d", tags=None, ctx=ctx)

    raw = tools.dump()
    parsed = yaml.safe_load(raw)
    assert list(parsed.keys()) == ["a", "b"]
