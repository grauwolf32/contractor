from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Optional, Union
from xml.sax.saxutils import escape as xml_escape

import yaml
from google.adk.agents.callback_context import CallbackContext
from google.adk.artifacts import BaseArtifactService
from google.adk.tools.tool_context import ToolContext
from google.genai import types


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        output: Union[str, dict[str, Any], list[Any]],
        fmt: str,
        type_hint: bool = False,
    ) -> Union[str, dict[str, Any], list[Any]]:
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
    ) -> Union[str, dict[str, Any]]:
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
    ) -> Union[str, dict[str, Any]]:
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
    ) -> Union[str, list[dict[str, Any]]]:
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
    ) -> Union[str, list[str], dict[str, Any]]:
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
            raw = yaml.safe_load(artifact.text) or {}

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

            raw = yaml.safe_load(artifact.text) or {}
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
        return sorted(self.notes.values(), key=lambda m: (m.ordinal, m.name))

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
        tags: Optional[list[str]],
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
    ) -> Optional[MemoryNote]:
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
    ) -> Optional[MemoryNote]:
        await self.load(ctx)
        return self.notes.get(name)

    async def search_memory(
        self,
        tags: list[str],
        ctx: ToolContext | CallbackContext,
    ) -> list[MemoryNote]:
        await self.load(ctx)
        normalized = set(tags)
        return sorted(
            [
                memory
                for memory in self.notes.values()
                if any(tag in memory.tags for tag in normalized)
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
    ) -> Optional[MemoryNote]:
        note = await self.read_memory(name, ctx)
        if note is None:
            return None
        if tag not in note.tags:
            return None
        return note


def memory_tools(name: str, fmt: MemoryFormat = MemoryFormat("json")):
    m = MemoryTools(name=name, fmt=fmt)

    async def write_memory(
        name: str,
        memory: str,
        description: str,
        tags: Optional[list[str]],
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        """
        Writes a memory to the memory store.

        Args:
            name: A concise, meaningful title that clearly reflects the content of the memory.
            memory: The full content to store.
            description: A brief summary explaining what the memory contains and why it matters.
            tags: A list of 1–3 short tags representing the main topics of the memory.

        Guidelines:
            - Use clear, specific, and descriptive names.
            - Avoid vague titles like "note1" or "stuff".
            - Ensure the description adds context, not just repetition of the name.
            - Always include one to three short, relevant tags.
            - Tags should be short and reflect the topic of the memory.
            - The tags "skill" and "inbox" are reserved for system use and must not be assigned manually.
        """
        await m.write_memory(name, memory, description, tags, tool_context)
        return {"result": "ok"}

    async def append_memory(
        name: str,
        text: str,
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        """
        Appends text to an existing memory.

        Args:
            name: The name of the memory to append to.
            text: The text to append to the memory.
        """
        memory = await m.append_memory(name, text, tool_context)
        if memory is None:
            return {"error": f"memory {name} not found"}

        return {"result": m.fmt.format_memory(memory)}

    async def read_memory(
        name: str,
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        """
        Reads a memory by name from the memory store.

        Args:
            name: The exact name of the memory to read.

        Returns:
            The full stored memory, including content, tags, insertion order, and timestamps.

        Behavior:
            - Reads from the unified store regardless of category.
            - Returns the full content of the memory when found.
            - Use this when the memory name is known but the category is not important.
            - Prefer skills_read or inbox_read when the memory should belong to a specific
        """
        memory = await m.read_memory(name, tool_context)
        if memory is None:
            return {"error": f"memory {name} not found"}

        return {"result": m.fmt.format_memory(memory)}

    async def search_memory(
        tags: list[str],
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        """
        Searches for memories matching any of the provided tags.

        Args:
            tags: One or more tags to match.

        Returns:
            A preview list of matching memories.

        Behavior:
            - Matches memories that contain at least one of the requested tags.
            - Returns memory previews rather than full content.
            - Results are ordered by insertion order.
            - Use this for tag-based discovery across the whole store.
        """
        memories = await m.search_memory(tags, tool_context)
        return {"result": m.fmt.format_memories(memories, preview=True)}

    async def list_tags(tool_context: ToolContext) -> dict[str, Any]:
        """
        Lists all tags in the memory store.

        Returns:
            A list of all tags.
        """
        tags = await m.list_tags(tool_context)
        return {"result": m.fmt.format_tags(tags)}

    async def list_memories(tool_context: ToolContext) -> dict[str, Any]:
        """
        Lists all memories in insertion order.

        Returns:
            A preview list of all memories.
        """
        memories = await m.list_memories(tool_context)
        return {"result": m.fmt.format_memories(memories, preview=True)}

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
        memories = await m.memories_by_tag("skill", tool_context)
        return {"result": m.fmt.format_memories(memories, preview=True)}

    async def skills_read(name: str, tool_context: ToolContext) -> dict[str, Any]:
        """
        Reads a specific agent skill by name.

        Args:
            name: The name of the skill memory.

        Returns:
            The full content of the matching memory if it exists and is tagged with "skill".

        Behavior:
            - Only returns memories tagged with "skill".
            - Use this when the agent needs the full instructions for a known skill.
            - Skills should contain actionable guidance, not just descriptive notes.
        """

        memory = await m.read_memory_by_tag(name, "skill", tool_context)
        if memory is None:
            return {"error": f"skill memory {name} not found"}

        return {"result": m.fmt.format_memory(memory)}

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
        memories = await m.memories_by_tag("inbox", tool_context)
        return {"result": m.fmt.format_memories(memories, preview=True)}

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
        memory = await m.read_memory_by_tag(name, "inbox", tool_context)
        if memory is None:
            return {"error": f"inbox memory {name} not found"}

        return {"result": m.fmt.format_memory(memory)}

    registry = [
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

    return registry
