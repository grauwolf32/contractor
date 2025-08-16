from functools import lru_cache

from google.adk.tools.mcp_tool.mcp_toolset import (MCPToolset,
                                                   StdioConnectionParams,
                                                   StdioServerParameters)
from pydantic_settings import BaseSettings, SettingsConfigDict


class SerenaConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="SERENA_MCP_",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return super().settings_customise_sources(
            settings_cls,
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )

    path: str = "/tools/mcp/serena"

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
        ),
        tool_filter=tool_filter,
    )
