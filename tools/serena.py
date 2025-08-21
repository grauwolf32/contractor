import os
from functools import lru_cache

from google.adk.tools.mcp_tool.mcp_toolset import (
    MCPToolset,
    StdioConnectionParams,
    StdioServerParameters,
)
from pydantic_settings import SettingsConfigDict

from helpers import Settings


class SerenaConfig(Settings):
    model_config = SettingsConfigDict(
        env_prefix="SERENA_MCP_",
    )

    path: str = "/tools/mcp/serena"
    timeout: int = 120

    @classmethod
    @lru_cache(maxsize=1)
    def get_settings(cls):
        return cls()


def serena_mcp_tools(tool_filter: list[str] | None = None) -> MCPToolset:
    config = SerenaConfig.get_settings()

    return MCPToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command="uv",
                args=["run", "--directory", config.path, "serena-mcp-server"],
            ),
            timeout=config.timeout,
        ),
        tool_filter=tool_filter,
        errlog=None,
    )
