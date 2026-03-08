from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional, Union
from xml.sax.saxutils import escape as xml_escape

import yaml
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.tool_context import ToolContext


@dataclass
class MemoryNote:
    name: str
    memory: str
    description: str
    tags: list[str] = field(default_factory=list)


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
        }

    @staticmethod
    def _memory_to_markdown(memory: MemoryNote, **kwargs) -> str:
        tags = ", ".join(memory.tags) if memory.tags else "-"
        return (
            f"### {memory.name}\n"
            f"**Description**: {memory.description}\n"
            f"**Tags**: {tags}\n"
            f"**Memory**:\n{memory.memory}\n"
        )

    @staticmethod
    def _memory_preview_to_markdown(memory: MemoryNote, **kwargs) -> str:
        tags = ", ".join(memory.tags) if memory.tags else "-"
        return (
            f"### {memory.name}\n"
            f"**Description**: {memory.description}\n"
            f"**Tags**: {tags}\n"
        )

    @staticmethod
    def _memory_to_yaml(memory: MemoryNote, **kwargs) -> str:
        payload = {
            f"memory_{memory.name}": {
                "name": memory.name,
                "memory": memory.memory,
                "description": memory.description,
                "tags": memory.tags,
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
        tags = "\n".join(
            f"{pad2}    <tag>{xml_escape(tag)}</tag>" for tag in memory.tags
        )

        if tags:
            tags_block = f"\n{pad2}<tags>\n{tags}\n{pad2}</tags>"
        else:
            tags_block = f"\n{pad2}<tags />"

        return (
            f'{pad}<memory name="{name}">\n'
            f"{pad2}<description>{description}</description>"
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

    def state_key(self) -> str:
        return f"{self.__class__.__name__}::{self.name}"

    def list_memories(self, ctx: ToolContext | CallbackContext) -> list[MemoryNote]:
        sk = self.state_key()
        ctx.state.setdefault(sk, {})

        memories: list[MemoryNote] = []
        for name, memory in ctx.state[sk].items():
            memories.append(
                MemoryNote(
                    name=name,
                    memory=memory.get("memory", ""),
                    description=memory.get("description", ""),
                    tags=memory.get("tags", []) or [],
                )
            )

        return memories

    def list_tags(self, ctx: ToolContext | CallbackContext) -> list[str]:
        sk = self.state_key()
        ctx.state.setdefault(sk, {})

        tags = set()
        for _, memory in ctx.state[sk].items():
            tags.update(memory.get("tags", []) or [])

        return sorted(tags)

    def write_memory(
        self,
        name: str,
        memory: str,
        description: str,
        tags: Optional[list[str]],
        ctx: ToolContext | CallbackContext,
    ):
        sk = self.state_key()
        ctx.state.setdefault(sk, {})

        m = MemoryNote(
            name=name,
            memory=memory,
            description=description,
            tags=tags or [],
        )

        # HACK: You should rewrite whole object inside the state for changes to take effect
        mempool = ctx.state[sk]
        mempool[name] = asdict(m)
        ctx.state[sk] = mempool
        return

    def read_memory(
        self, name: str, ctx: ToolContext | CallbackContext
    ) -> Optional[MemoryNote]:
        sk = self.state_key()
        ctx.state.setdefault(sk, {})

        if name not in ctx.state[sk]:
            return None

        raw = ctx.state[sk][name]
        return MemoryNote(
            name=name,
            memory=raw.get("memory", ""),
            description=raw.get("description", ""),
            tags=raw.get("tags", []) or [],
        )

    def search_memory(
        self, tags: list[str], ctx: ToolContext | CallbackContext
    ) -> list[MemoryNote]:
        sk = self.state_key()
        ctx.state.setdefault(sk, {})

        memories: list[MemoryNote] = []
        for name, memory in ctx.state[sk].items():
            memory_tags = memory.get("tags", []) or []
            if any(tag in memory_tags for tag in tags):
                memories.append(
                    MemoryNote(
                        name=name,
                        memory=memory.get("memory", ""),
                        description=memory.get("description", ""),
                        tags=memory_tags,
                    )
                )
        return memories


def memory_tools(name: str, fmt: MemoryFormat = MemoryFormat("json")):
    m = MemoryTools(name=name, fmt=fmt)

    def write_memory(
        name: str,
        memory: str,
        description: str,
        tags: Optional[list[str]],
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        """
        Writes a memory to the memory store.
        Args:
            name: The name of the memory.
            memory: The memory to write.
            description: Brief description of the memory.
            tags: The tags of the memory.
        Reminder:
        - Always add one to three tags to the memory.
        - Tags should be short and reflect the topic of the memory.
        """

        m.write_memory(name, memory, description, tags, tool_context)
        return {"result": "ok"}

    def read_memory(name: str, tool_context: ToolContext) -> dict[str, Any]:
        """
        Reads a memory from the memory store.
        Args:
            name: The name of the memory.
        Returns:
            The memory content, if it exists.
        """

        memory = m.read_memory(name, tool_context)
        if memory is None:
            return {"error": f"memory {name} not found"}

        return {"result": m.fmt.format_memory(memory)}

    def search_memory(tags: list[str], tool_context: ToolContext) -> dict[str, Any]:
        """
        Searches for memories in the memory store.
        Args:
            tags: The tags to search for.
        Returns:
           A list of memory names that match the tags with description.
        """

        memories = m.search_memory(tags, tool_context)
        return {"result": m.fmt.format_memories(memories, preview=True)}

    def list_tags(tool_context: ToolContext) -> dict[str, Any]:
        """
        Lists all tags in the memory store.
        Returns:
            A list of all tags.
        """
        return {"result": m.fmt.format_tags(m.list_tags(tool_context))}

    def list_memories(tool_context: ToolContext) -> dict[str, Any]:
        """
        Lists all memories in the memory store.
        Returns:
            A list of all memories.
        """

        memories = m.list_memories(tool_context)
        return {"result": m.fmt.format_memories(memories, preview=True)}

    return [
        write_memory,
        read_memory,
        search_memory,
        list_tags,
        list_memories,
    ]
