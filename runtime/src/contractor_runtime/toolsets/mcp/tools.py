"""A narrow ADK BaseTool over an allocation's frozen MCP declaration."""

from __future__ import annotations

import asyncio
from typing import Any

from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from jsonschema import Draft202012Validator
from mcp.types import Result

from contractor_runtime.toolsets.common.deadlines import invocation_deadline
from contractor_runtime.toolsets.mcp.errors import MCPError
from contractor_runtime.toolsets.mcp.schema import ToolDefinition, bounded_json
from contractor_runtime.toolsets.mcp.session import MCPSessionOwner


class MCPTool(BaseTool):
    def __init__(self, name: str, owner: MCPSessionOwner, *, timeout_seconds: float) -> None:
        super().__init__(name=name, description="")
        self._owner = owner
        self._definition: ToolDefinition | None = None
        self._input_validator: Draft202012Validator | None = None
        self._output_validator: Draft202012Validator | None = None
        self._timeout_seconds = timeout_seconds

    @property
    def schema_digest(self) -> str | None:
        return self._owner.schema_digest

    async def prepare_allocation(self, *, deadline: float) -> None:
        await self._owner.prepare(deadline=deadline)
        if self._definition is None:
            self._definition = self._owner.definition(self.name)
            self.description = self._definition.description
            self._input_validator = Draft202012Validator(self._definition.input_schema)
            if self._definition.output_schema is not None:
                self._output_validator = Draft202012Validator(self._definition.output_schema)

    def as_adk_tool(self) -> BaseTool:
        if self._definition is None:
            raise MCPError("session_not_ready")
        return self

    def _get_declaration(self) -> types.FunctionDeclaration:
        self.as_adk_tool()
        assert self._definition is not None
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters_json_schema=self._definition.declaration()["inputSchema"],
        )

    async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> dict[str, Any]:
        del tool_context
        self.as_adk_tool()
        bounded_json(args, 64 * 1024, code="invalid_arguments")
        if not isinstance(args, dict) or not self._input_validator.is_valid(args):
            raise MCPError("invalid_arguments")
        deadline = asyncio.get_running_loop().time() + self._timeout_seconds
        enclosing = invocation_deadline.get()
        if enclosing is not None:
            deadline = min(deadline, enclosing)
        result = await self._owner.call(self.name, args, deadline=deadline)
        return self._result(result)

    def _result(self, response: Result) -> dict[str, Any]:
        raw = response.model_dump(mode="json", by_alias=True, exclude_unset=True)
        bounded_json(raw, 256 * 1024, code="result_too_large")
        content = raw.get("content")
        is_error = raw.get("isError", False)
        if not isinstance(content, list) or type(is_error) is not bool:
            raise MCPError("invalid_result")
        if len(content) > 64:
            raise MCPError("result_too_large")
        blocks = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "text":
                raise MCPError("unsupported_content")
            if not isinstance(block.get("text"), str):
                raise MCPError("invalid_result")
            # Content annotations/_meta are transport metadata, not model input.
            blocks.append({"type": "text", "text": block["text"]})
        result: dict[str, Any] = {"content": blocks, "isError": is_error}
        if "structuredContent" in raw:
            if not isinstance(raw["structuredContent"], dict):
                raise MCPError("invalid_result")
            result["structuredContent"] = raw["structuredContent"]
        if (
            not is_error
            and self._output_validator is not None
            and not self._output_validator.is_valid(result.get("structuredContent"))
        ):
            raise MCPError("invalid_result")
        self._owner.reject_private_content(result)
        return result

    async def close_before(self, *, deadline: float) -> None:
        await self._owner.close(deadline=deadline)
        self._erase()

    async def close(self) -> None:
        await self._owner.close()
        self._erase()

    def _erase(self) -> None:
        self._definition = None
        self._input_validator = self._output_validator = None
        self.description = ""

    def __repr__(self) -> str:
        return f"MCPTool(name={self.name!r}, prepared={self._definition is not None!r})"
