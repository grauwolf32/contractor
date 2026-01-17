from typing import Any, Optional, Literal
from dataclasses import dataclass, field, asdict
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


# TODO: Use memory format to format the memory output


@dataclass
class MemoryTools:
    name: str

    def state_key(self) -> str:
        return f"{self.__class__.__name__}::{self.name}"

    def list_memories(self, ctx: ToolContext | CallbackContext) -> list[str]:
        sk = self.state_key()
        ctx.state.setdefault(sk, {})

        memories = []
        for name, memory in ctx.state[sk].items():
            memories.append(
                {
                    "name": name,
                    "description": memory.get("description", ""),
                    "tags": memory.get("tags", []),
                }
            )

        return memories

    def list_tags(self, ctx: ToolContext | CallbackContext) -> list[str]:
        sk = self.state_key()
        ctx.state.setdefault(sk, {})

        tags = set()
        for _, memory in ctx.state[sk].items():
            tags.update(memory.get("tags", []))

        return list(tags)

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

        m = MemoryNote(name, memory, description, tags)
        ctx.state[sk][name] = asdict(m)
        return

    def read_memory(
        self, name: str, ctx: ToolContext | CallbackContext
    ) -> Optional[str]:
        sk = self.state_key()
        ctx.state.setdefault(sk, {})

        if name in ctx.state[sk]:
            return ctx.state[sk][name].get("memory", "")
        return None

    def search_memory(
        self, tags: list[str], ctx: ToolContext | CallbackContext
    ) -> list[dict[str, Any]]:
        sk = self.state_key()
        ctx.state.setdefault(sk, {})

        memories = []
        for _, memory in ctx.state[sk].items():
            if any([tag in memory.get("tags", []) for tag in tags]):
                memories.append(memory)
        return memories


def memory_tools(name: str, fmt: MemoryFormat = MemoryFormat("json")):
    m = MemoryTools(name)

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

        return {"result": memory}

    def search_memory(tags: list[str], tool_context: ToolContext) -> dict[str, Any]:
        """
        Searches for memories in the memory store.
        Args:
            tags: The tags to search for.
        Returns:
           A list of memory names that match the tags with description.
        """

        memories = m.search_memory(tags, tool_context)
        return {"result": memories}

    def list_tags(tool_context: ToolContext) -> dict[str, Any]:
        """
        Lists all tags in the memory store.
        Returns:
            A list of all tags.
        """
        return {"result": m.list_tags(tool_context)}

    def list_memories(tool_context: ToolContext) -> dict[str, Any]:
        """
        Lists all memories in the memory store.
        Returns:
            A list of all memories.
        """

        return {"result": m.list_memories(tool_context)}

    return [
        write_memory,
        read_memory,
        search_memory,
        list_tags,
        list_memories,
    ]
