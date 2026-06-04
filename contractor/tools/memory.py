from __future__ import annotations

import asyncio
import contextlib
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, TypeVar
from xml.sax.saxutils import escape as xml_escape

import yaml
from google.adk.agents.callback_context import CallbackContext
from google.adk.artifacts import BaseArtifactService
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from contractor.tools.observations import SKILLS_READ_STATE_KEY
from contractor.tools.result import aguard, err
from contractor.utils import utc_now_iso

# Passthrough type for _type_hint: non-str payloads return unchanged, a str may
# be code-fence-wrapped — preserving the caller's narrower type.
_T = TypeVar("_T")


@dataclass
class MemoryNote:
    name: str
    memory: str
    description: str
    tags: list[str] = field(default_factory=list)
    ordinal: int = 0
    created_at: str = ""
    updated_at: str = ""


@dataclass
class MemoryFormat:
    _format: Literal["json", "markdown", "yaml", "xml"] = "json"

    @staticmethod
    def _type_hint(
        output: _T,
        fmt: str,
        type_hint: bool = False,
    ) -> _T | str:
        if not type_hint or not isinstance(output, str):
            return output
        return f"```{fmt}\n{output}\n```"

    @staticmethod
    def _memory_to_json(memory: MemoryNote, **kwargs) -> dict[str, Any]:
        return asdict(memory)

    @staticmethod
    def _memory_preview_to_json(memory: MemoryNote, **kwargs) -> dict[str, Any]:
        return {
            "name": memory.name,
            "description": memory.description,
            "tags": memory.tags,
            "ordinal": memory.ordinal,
            "created_at": memory.created_at,
            "updated_at": memory.updated_at,
        }

    @staticmethod
    def _memory_to_markdown(memory: MemoryNote, **kwargs) -> str:
        tags = ", ".join(memory.tags) if memory.tags else "-"
        return (
            f"### {memory.name}\n"
            f"**Description**: {memory.description}\n"
            f"**Tags**: {tags}\n"
            f"**Ordinal**: {memory.ordinal}\n"
            f"**Created At**: {memory.created_at or '-'}\n"
            f"**Updated At**: {memory.updated_at or '-'}\n"
            f"**Memory**:\n{memory.memory}\n"
        )

    @staticmethod
    def _memory_preview_to_markdown(memory: MemoryNote, **kwargs) -> str:
        tags = ", ".join(memory.tags) if memory.tags else "-"
        return (
            f"### {memory.name}\n"
            f"**Description**: {memory.description}\n"
            f"**Tags**: {tags}\n"
            f"**Ordinal**: {memory.ordinal}\n"
            f"**Created At**: {memory.created_at or '-'}\n"
            f"**Updated At**: {memory.updated_at or '-'}\n"
        )

    @staticmethod
    def _memory_to_yaml(memory: MemoryNote, **kwargs) -> str:
        payload = {
            f"memory_{memory.name}": {
                "name": memory.name,
                "memory": memory.memory,
                "description": memory.description,
                "tags": memory.tags,
                "ordinal": memory.ordinal,
                "created_at": memory.created_at,
                "updated_at": memory.updated_at,
            }
        }
        return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)

    @staticmethod
    def _memory_preview_to_yaml(memory: MemoryNote, **kwargs) -> str:
        payload = {
            f"memory_{memory.name}": {
                "name": memory.name,
                "description": memory.description,
                "tags": memory.tags,
                "ordinal": memory.ordinal,
                "created_at": memory.created_at,
                "updated_at": memory.updated_at,
            }
        }
        return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)

    @staticmethod
    def _memory_to_xml(memory: MemoryNote, indent: int = 0, **kwargs) -> str:
        pad = " " * (indent * 4)
        pad2 = " " * ((indent + 1) * 4)

        name = xml_escape(memory.name)
        description = xml_escape(memory.description)
        memory_text = xml_escape(memory.memory)
        created_at = xml_escape(memory.created_at)
        updated_at = xml_escape(memory.updated_at)
        tags = "\n".join(
            f"{pad2}    <tag>{xml_escape(tag)}</tag>" for tag in memory.tags
        )

        if tags:
            tags_block = f"\n{pad2}<tags>\n{tags}\n{pad2}</tags>"
        else:
            tags_block = f"\n{pad2}<tags />"

        return (
            f'{pad}<memory name="{name}">\n'
            f"{pad2}<description>{description}</description>\n"
            f"{pad2}<ordinal>{memory.ordinal}</ordinal>\n"
            f"{pad2}<created_at>{created_at}</created_at>\n"
            f"{pad2}<updated_at>{updated_at}</updated_at>\n"
            f"{pad2}<content>{memory_text}</content>"
            f"{tags_block}\n"
            f"{pad}</memory>"
        )

    @staticmethod
    def _memory_preview_to_xml(memory: MemoryNote, indent: int = 0, **kwargs) -> str:
        pad = " " * (indent * 4)
        pad2 = " " * ((indent + 1) * 4)

        name = xml_escape(memory.name)
        description = xml_escape(memory.description)
        created_at = xml_escape(memory.created_at)
        updated_at = xml_escape(memory.updated_at)
        tags = "\n".join(
            f"{pad2}    <tag>{xml_escape(tag)}</tag>" for tag in memory.tags
        )

        if tags:
            tags_block = f"\n{pad2}<tags>\n{tags}\n{pad2}</tags>"
        else:
            tags_block = f"\n{pad2}<tags />"

        return (
            f'{pad}<memory name="{name}">\n'
            f"{pad2}<description>{description}</description>\n"
            f"{pad2}<ordinal>{memory.ordinal}</ordinal>\n"
            f"{pad2}<created_at>{created_at}</created_at>\n"
            f"{pad2}<updated_at>{updated_at}</updated_at>"
            f"{tags_block}\n"
            f"{pad}</memory>"
        )

    def format_memory(
        self,
        memory: MemoryNote,
        *,
        type_hint: bool = False,
    ) -> str | dict[str, Any]:
        formatters = {
            "json": self._memory_to_json,
            "markdown": self._memory_to_markdown,
            "yaml": self._memory_to_yaml,
            "xml": self._memory_to_xml,
        }
        formatter = formatters.get(self._format, self._memory_to_json)
        output = formatter(memory)
        return self._type_hint(output, self._format, type_hint)

    def format_memory_preview(
        self,
        memory: MemoryNote,
        *,
        type_hint: bool = False,
    ) -> str | dict[str, Any]:
        formatters = {
            "json": self._memory_preview_to_json,
            "markdown": self._memory_preview_to_markdown,
            "yaml": self._memory_preview_to_yaml,
            "xml": self._memory_preview_to_xml,
        }
        formatter = formatters.get(self._format, self._memory_preview_to_json)
        output = formatter(memory)
        return self._type_hint(output, self._format, type_hint)

    def format_memories(
        self,
        memories: list[MemoryNote],
        *,
        type_hint: bool = False,
        preview: bool = False,
    ) -> str | list[dict[str, Any]]:
        if self._format == "json":
            if preview:
                return [self._memory_preview_to_json(m) for m in memories]
            return [self._memory_to_json(m) for m in memories]

        if self._format in {"markdown", "yaml"}:
            formatter = self.format_memory_preview if preview else self.format_memory
            output = "\n".join(
                item
                for item in (formatter(m, type_hint=False) for m in memories)
                if isinstance(item, str)
            )
            return self._type_hint(output, self._format, type_hint)

        if self._format == "xml":
            formatter = self._memory_preview_to_xml if preview else self._memory_to_xml
            output = (
                "<memories>\n"
                + "\n".join(formatter(m, indent=1) for m in memories)
                + "\n</memories>"
            )
            return self._type_hint(output, self._format, type_hint)

        if preview:
            return [self._memory_preview_to_json(m) for m in memories]
        return [self._memory_to_json(m) for m in memories]

    def format_tags(
        self,
        tags: list[str],
        *,
        type_hint: bool = False,
    ) -> str | list[str] | dict[str, Any]:
        if self._format == "json":
            return {"tags": tags}

        if self._format == "markdown":
            output = "\n".join(f"- {tag}" for tag in tags) if tags else "- none"
            return self._type_hint(output, self._format, type_hint)

        if self._format == "yaml":
            output = yaml.safe_dump({"tags": tags}, sort_keys=False, allow_unicode=True)
            return self._type_hint(output, self._format, type_hint)

        if self._format == "xml":
            if tags:
                output = (
                    "<tags>\n"
                    + "\n".join(f"    <tag>{xml_escape(tag)}</tag>" for tag in tags)
                    + "\n</tags>"
                )
            else:
                output = "<tags />"
            return self._type_hint(output, self._format, type_hint)

        return {"tags": tags}


# Tags reserved for system-managed memory. Notes carrying any of these are
# reachable only through their dedicated tools (skills_list/skills_read,
# inbox_list/inbox_read) — they are hidden from the generic memory surface
# (list_memories/search_memory/read_memory) so worker notes aren't polluted
# by skill bodies and the agent doesn't pull skills via read_memory.
_RESERVED_TAGS: frozenset[str] = frozenset({"skill", "inbox"})


def _is_reserved(note: MemoryNote) -> bool:
    return bool(_RESERVED_TAGS.intersection(note.tags))


@dataclass
class MemoryTools:
    name: str
    fmt: MemoryFormat = field(default_factory=MemoryFormat)
    notes: dict[str, MemoryNote] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def memory_key(self) -> str:
        return f"user:memory/{self.name}"

    def _normalize_note(
        self, name: str, item: dict[str, Any], fallback_ordinal: int
    ) -> MemoryNote:
        return MemoryNote(
            name=item.get("name", name),
            memory=item.get("memory", ""),
            description=item.get("description", ""),
            tags=item.get("tags", []) or [],
            ordinal=item.get("ordinal", fallback_ordinal),
            created_at=item.get("created_at", ""),
            updated_at=item.get("updated_at", ""),
        )

    def _next_ordinal(self) -> int:
        if not self.notes:
            return 1
        return max(note.ordinal for note in self.notes.values()) + 1

    async def inject(
        self,
        memories: list[MemoryNote],
        artifact_service: BaseArtifactService,
        app_name: str,
        user_id: str,
    ):
        """
        Inject memories from the outer world.
        """
        artifact = await artifact_service.load_artifact(
            filename=self.memory_key(),
            app_name=app_name,
            user_id=user_id,
        )
        raw: dict[str, Any] = {}
        if artifact is not None:
            raw = yaml.safe_load(artifact.text or "") or {}

        next_ordinal = 1
        if raw:
            existing_ordinals = [
                item.get("ordinal", 0)
                for item in raw.values()
                if isinstance(item, dict)
            ]
            next_ordinal = max(existing_ordinals, default=0) + 1

        for memory in memories:
            existing = raw.get(memory.name)
            if existing:
                memory.ordinal = existing.get("ordinal", next_ordinal)
                memory.created_at = existing.get(
                    "created_at", memory.created_at or utc_now_iso()
                )
                memory.updated_at = utc_now_iso()
            else:
                if memory.ordinal <= 0:
                    memory.ordinal = next_ordinal
                    next_ordinal += 1
                now = utc_now_iso()
                memory.created_at = memory.created_at or now
                memory.updated_at = memory.updated_at or now

            raw[memory.name] = asdict(memory)

        dump = yaml.safe_dump(
            raw,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )
        artifact_part = types.Part.from_text(text=dump)
        await artifact_service.save_artifact(
            filename=self.memory_key(),
            artifact=artifact_part,
            app_name=app_name,
            user_id=user_id,
        )

    async def load(self, ctx: ToolContext | CallbackContext):
        async with self._lock:
            artifact = await ctx.load_artifact(filename=self.memory_key())
            if artifact is None:
                self.notes = {}
                return

            raw = yaml.safe_load(artifact.text or "") or {}
            notes: dict[str, MemoryNote] = {}
            for index, (name, item) in enumerate(raw.items(), start=1):
                if not isinstance(item, dict):
                    continue
                note = self._normalize_note(
                    name=name, item=item, fallback_ordinal=index
                )
                notes[note.name] = note
            self.notes = notes

    def dump(self) -> str:
        notes = {
            name: asdict(memory)
            for name, memory in sorted(
                self.notes.items(),
                key=lambda pair: (pair[1].ordinal, pair[0]),
            )
        }
        return yaml.safe_dump(
            notes,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )

    async def save(self, ctx: ToolContext | CallbackContext):
        async with self._lock:
            artifact = types.Part.from_text(text=self.dump())
            await ctx.save_artifact(filename=self.memory_key(), artifact=artifact)

    async def list_memories(
        self,
        ctx: ToolContext | CallbackContext,
    ) -> list[MemoryNote]:
        await self.load(ctx)
        return sorted(
            (m for m in self.notes.values() if not _is_reserved(m)),
            key=lambda m: (m.ordinal, m.name),
        )

    async def list_tags(self, ctx: ToolContext | CallbackContext) -> list[str]:
        await self.load(ctx)
        tags = set()
        for memory in self.notes.values():
            tags.update(memory.tags)
        return sorted(tags)

    async def write_memory(
        self,
        name: str,
        memory: str,
        description: str,
        tags: list[str] | None,
        ctx: ToolContext | CallbackContext,
    ):
        await self.load(ctx)

        normalized_tags = (tags or [])[:3]
        now = utc_now_iso()
        existing = self.notes.get(name)

        if existing is not None:
            note = MemoryNote(
                name=name,
                memory=memory,
                description=description,
                tags=normalized_tags,
                ordinal=existing.ordinal,
                created_at=existing.created_at or now,
                updated_at=now,
            )
        else:
            note = MemoryNote(
                name=name,
                memory=memory,
                description=description,
                tags=normalized_tags,
                ordinal=self._next_ordinal(),
                created_at=now,
                updated_at=now,
            )

        self.notes[name] = note
        await self.save(ctx)

    async def append_memory(
        self,
        name: str,
        text: str,
        ctx: ToolContext | CallbackContext,
    ) -> MemoryNote | None:
        note = await self.read_memory(name, ctx)
        if note is None:
            return None

        appended = "\n".join(part for part in (note.memory, text) if part)
        await self.write_memory(
            name=note.name,
            memory=appended,
            description=note.description,
            tags=note.tags,
            ctx=ctx,
        )
        return await self.read_memory(name, ctx)

    async def read_memory(
        self,
        name: str,
        ctx: ToolContext | CallbackContext,
    ) -> MemoryNote | None:
        await self.load(ctx)
        note = self.notes.get(name)
        # Reserved notes (skills/inbox) are not readable here — steer the agent
        # to skills_read / inbox_read. Treated as absent.
        if note is not None and _is_reserved(note):
            return None
        return note

    async def search_memory(
        self,
        tags: list[str],
        ctx: ToolContext | CallbackContext,
    ) -> list[MemoryNote]:
        await self.load(ctx)
        # Drop reserved tags from the query and exclude reserved notes, so this
        # generic surface never returns skills/inbox.
        normalized = set(tags) - _RESERVED_TAGS
        if not normalized:
            return []
        return sorted(
            [
                memory
                for memory in self.notes.values()
                if not _is_reserved(memory)
                and any(tag in memory.tags for tag in normalized)
            ],
            key=lambda m: (m.ordinal, m.name),
        )

    async def memories_by_tag(
        self,
        tag: str,
        ctx: ToolContext | CallbackContext,
    ) -> list[MemoryNote]:
        await self.load(ctx)
        return sorted(
            [memory for memory in self.notes.values() if tag in memory.tags],
            key=lambda m: (m.ordinal, m.name),
        )

    async def read_memory_by_tag(
        self,
        name: str,
        tag: str,
        ctx: ToolContext | CallbackContext,
    ) -> MemoryNote | None:
        # Dedicated path for reserved categories (skills_read / inbox_read):
        # bypass the reserved-tag filter in read_memory and read self.notes
        # directly, then enforce the requested tag.
        await self.load(ctx)
        note = self.notes.get(name)
        if note is None:
            return None
        if tag not in note.tags:
            return None
        return note


def _push_skill_read(tool_context: ToolContext | None, skill_name: str) -> None:
    """Record a successful skill read into session state (deterministic).

    Mirrors ``_push_fs_coverage`` in the fs tools: the tool itself writes the
    *canonical resolved* skill name (so aliases like ``"sinks"`` collapse to
    ``"trace/references/sinks"``), and only on a confirmed hit — never a
    not-found query. Dedupes, preserves read order. The write becomes an ADK
    state delta forwarded to the planner by ``AgentTool``.
    """
    if tool_context is None:
        return
    state = getattr(tool_context, "state", None)
    if state is None:
        return
    with contextlib.suppress(Exception):
        seen = list(state.get(SKILLS_READ_STATE_KEY) or [])
        if skill_name not in seen:
            seen.append(skill_name)
            state[SKILLS_READ_STATE_KEY] = seen


def _resolve_skill_reference(
    name: str,
    skill_memories: list[MemoryNote],
) -> MemoryNote | None:
    """Best-effort match of a skill-reference name against loaded skill notes.

    Skill references are stored as ``<skill>/references/<topic>`` with no
    extension, but index files (and agents copying from them) tend to cite the
    bare ``references/<topic>.md`` form. Normalize the query and fall back to a
    unique suffix / basename match so either form resolves. Returns None when
    the match is absent or ambiguous (the caller then lists the real names).
    """
    query = name[:-3] if name.endswith(".md") else name
    query = query.strip("/")
    if not query:
        return None

    for tier in (
        lambda n: n.name == query,
        lambda n: n.name.endswith(f"/{query}"),
        lambda n: n.name.rsplit("/", 1)[-1] == query.rsplit("/", 1)[-1],
    ):
        matches = [n for n in skill_memories if tier(n)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return None
    return None


def memory_tools(name: str, fmt: MemoryFormat | None = None):
    m = MemoryTools(name=name, fmt=fmt or MemoryFormat("json"))

    async def write_memory(
        name: str,
        memory: str,
        description: str,
        tags: list[str] | None,
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        """
        Writes a memory to the shared memory store.

        OVERWRITES any existing entry with the same name, including its tags
        and description. To extend without losing metadata, use append_memory.

        Before writing, check list_memories or search_memory to avoid
        duplicating an entry under a slightly different name. If unsure which
        tag to use, call list_tags first — tags are case-sensitive.

        Args:
            name: A concise, meaningful title that clearly reflects the content of the memory.
            memory: The full content to store.
            description: A brief summary explaining what the memory contains and why it matters.
            tags: A list of 1–3 short tags representing the main topics of the memory.

        Guidelines:
            - Use clear, specific, and descriptive names in snake_case; avoid vague titles like "note1".
            - Ensure the description adds context, not just repetition of the name.
            - The tags "skill" and "inbox" are reserved for system use.
        """

        async def _impl() -> Any:
            await m.write_memory(name, memory, description, tags, tool_context)
            return "ok"

        return await aguard(_impl)

    async def append_memory(
        name: str,
        text: str,
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        """
        Appends text to an existing memory, preserving its description and tags.

        Prefer this over write_memory when adding incremental findings to a
        known entry — no need to re-specify the description or tags.

        Args:
            name: The name of the memory to append to.
            text: The text to append to the memory.
        """

        async def _impl() -> Any:
            memory = await m.append_memory(name, text, tool_context)
            if memory is None:
                return err(f"memory {name} not found")
            return m.fmt.format_memory(memory)

        return await aguard(_impl)

    async def read_memory(
        name: str,
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        """
        Reads a memory by name.

        Use this only when the name is known and the full content is needed.
        If the name is unknown, use search_memory or list_memories first.
        Memory results persist in context — reuse the previous result if you
        have already read this entry since the last write_memory/append_memory.

        Args:
            name: The exact name of the memory to read.

        Returns:
            The full stored memory, including content, tags, insertion order, and timestamps.

        Behavior:
            - Skills and inbox entries are NOT readable here — use skills_read
              or inbox_read for those (they are treated as not found).
        """

        async def _impl() -> Any:
            memory = await m.read_memory(name, tool_context)
            if memory is None:
                return err(f"memory {name} not found")
            return m.fmt.format_memory(memory)

        return await aguard(_impl)

    async def search_memory(
        tags: list[str],
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        """
        Searches for memories matching any of the provided tags.

        Prefer this over multiple read_memory calls when collecting context by
        topic — one call returns previews of all matches in insertion order.
        Previews only (no full body); follow up with read_memory only when the
        full content of a specific entry is needed.

        Args:
            tags: One or more tags to match.

        Returns:
            A preview list of matching memories.
        """

        async def _impl() -> Any:
            memories = await m.search_memory(tags, tool_context)
            return m.fmt.format_memories(memories, preview=True)

        return await aguard(_impl)

    async def list_tags(tool_context: ToolContext) -> dict[str, Any]:
        """
        Lists all tags in the memory store.

        Returns:
            A list of all tags.
        """

        async def _impl() -> Any:
            tags = await m.list_tags(tool_context)
            return m.fmt.format_tags(tags)

        return await aguard(_impl)

    async def list_memories(tool_context: ToolContext) -> dict[str, Any]:
        """
        Lists all memories in insertion order (preview only).

        Call once to discover what is available. Results persist in context —
        reuse the previous result unless a write_memory/append_memory has
        happened since.

        Returns:
            A preview list of all memories.
        """

        async def _impl() -> Any:
            memories = await m.list_memories(tool_context)
            return m.fmt.format_memories(memories, preview=True)

        return await aguard(_impl)

    async def skills_list(tool_context: ToolContext) -> dict[str, Any]:
        """
        Lists all agent skills.

        Skills are reusable operating instructions for the agent, such as workflows,
        procedures, tool-usage guidance, style rules, and domain-specific task patterns.

        Returns:
            A preview list of memories tagged with "skill".

        Behavior:
            - Only returns memories tagged with "skill".
            - Results are ordered by insertion order.
            - Use this to discover what reusable capabilities the agent has available.
            - Skills should be durable, explicit, and reusable across tasks.
        """

        async def _impl() -> Any:
            memories = await m.memories_by_tag("skill", tool_context)
            return m.fmt.format_memories(memories, preview=True)

        return await aguard(_impl)

    async def skills_read(name: str, tool_context: ToolContext) -> dict[str, Any]:
        """
        Reads a specific agent skill by name.

        Args:
            name: The name of the skill memory. The index's own name (e.g.
                "trace") returns the index; a reference is named
                "<skill>/references/<topic>". A trailing ".md" and a missing
                "<skill>/" prefix are tolerated, so "references/sinks" and
                "sinks" both resolve to "trace/references/sinks" when only the
                trace skill is loaded.

        Returns:
            The full content of the matching memory if it exists and is tagged with "skill".

        Behavior:
            - Only returns memories tagged with "skill".
            - Use this when the agent needs the full instructions for a known skill.
            - Skills should contain actionable guidance, not just descriptive notes.
        """

        async def _impl() -> Any:
            memory = await m.read_memory_by_tag(name, "skill", tool_context)
            if memory is None:
                skill_memories = await m.memories_by_tag("skill", tool_context)
                memory = _resolve_skill_reference(name, skill_memories)
            if memory is None:
                available = [
                    s.name for s in await m.memories_by_tag("skill", tool_context)
                ]
                return err(f"skill memory {name!r} not found", available=available)
            _push_skill_read(tool_context, memory.name)
            return m.fmt.format_memory(memory)

        return await aguard(_impl)

    async def inbox_list(tool_context: ToolContext) -> dict[str, Any]:
        """
        Lists incoming information captured in the inbox.

        Inbox items represent incoming messages or information that may contain
        important facts, requests, constraints, decisions, or other details worth keeping.

        Returns:
            A preview list of memories tagged with "inbox".

        Behavior:
            - Only returns memories tagged with "inbox".
            - Results are ordered by insertion order.
            - Use this to review captured incoming information.
            - Inbox is a triage view, not a store for reusable procedures.
        """

        async def _impl() -> Any:
            memories = await m.memories_by_tag("inbox", tool_context)
            return m.fmt.format_memories(memories, preview=True)

        return await aguard(_impl)

    async def inbox_read(name: str, tool_context: ToolContext) -> dict[str, Any]:
        """
        Reads a specific inbox item by name.

        Args:
            name: The name of the inbox memory.

        Returns:
            The full content of the matching memory if it exists and is tagged with "inbox".

        Behavior:
            - Only returns memories tagged with "inbox".
            - Use this when reviewing a specific incoming message or captured information item.
            - Inbox items may later be promoted into general memory or transformed into skills if appropriate.
        """

        async def _impl() -> Any:
            memory = await m.read_memory_by_tag(name, "inbox", tool_context)
            if memory is None:
                return err(f"inbox memory {name} not found")
            return m.fmt.format_memory(memory)

        return await aguard(_impl)

    return [
        append_memory,
        write_memory,
        read_memory,
        search_memory,
        list_tags,
        list_memories,
        skills_list,
        skills_read,
        inbox_list,
        inbox_read,
    ]

