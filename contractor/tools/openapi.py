import yaml
from typing import Any, Final
from dataclasses import dataclass, field

from google.genai import types
from google.adk.tools.tool_context import ToolContext

openapi_base_schema: Final[dict[str, Any]] = {
    "openapi": "3.0.3",
    "info": {
        "title": "",
        "description": "",
        "version": "1.0.0",
    },
    "paths": {},
    "components": {
        "schemas": {},
        "parameters": {},
        "responses": {},
        "securitySchemes": {},
        "examples": {},
        "requestBodies": {},
        "headers": {},
        "links": {},
        "callbacks": {},
    },
}


@dataclass
class OpenAPITools:
    name: str = "openapi"
    schema: dict[str, Any] = field(default_factory=dict)
    version: int | None = None

    def dump(self) -> str:
        return yaml.safe_dump(
            self.schema,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )

    def meta(self) -> dict[str, Any]:
        return {}

    async def save_schema(self, ctx: ToolContext) -> int:
        artifact = types.Part.from_text(self.dump())
        meta = self.meta()
        self.version = await ctx.save_artifact(self.name, artifact, meta)
        return self.version

    async def load_schema(self, ctx: ToolContext) -> int:
        artifact = await ctx.load_artifact(self.name)
        artifact = await ctx.load_artifact(self.name, self.version)
        self.schema = yaml.safe_load(artifact.text)
        return self.version
