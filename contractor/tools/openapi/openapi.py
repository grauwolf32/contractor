from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Final, Literal, Optional

import yaml
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from pydantic import ValidationError

from contractor.utils import DictDiff, deep_merge, dict_diff
from contractor.tools.openapi.models import (
    PathItem,
    SecurityScheme,
    RequestBody,
    Response,
)

COMPONENT_KEY_ERROR: Final[str] = "got key={key} but only keys {keys} are allowed"
COMPONENT_VALIDATION_ERROR: Final[str] = (
    "{exception}\n{component} should be formatted as:\n{schema}"
)
COMPONENT_NOT_FOUND_OR_ALREADY_REMOVED: Final[str] = (
    "{name} in {key} is not found, or already removed"
)
PATH_NOT_FOUND_OR_ALREADY_REMOVED: Final[str] = (
    "{path} is not found, or already removed"
)
FILE_VALIDATION_BANNED_EXTENSIONS: Final[str] = (
    "Using files with extensions {extensions} as the evidence is PROHIBITED.\n"
    "You MUST provide only code files as evidences.\n"
    "PROHIBITED FILES:\n{banned}"
)
FILE_VALIDATION_NO_FILES_PROVIDED: Final[str] = (
    "No files provided! You MUST provide at least one file as the evidence.\n"
)

SERVER_ALREADY_EXISTS: Final[str] = "Server with url {url} already exists."

SERVER_NOT_EXISTS: Final[str] = "Server with url {url} is not exists."

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
class OpenAPIFormat: ...


@dataclass
class OpenApiArtifact:
    name: str
    schema: dict[str, Any] = field(default_factory=dict)
    version: int | None = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def dump(self) -> str:
        return yaml.safe_dump(
            self.schema,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )

    def meta(self) -> dict[str, Any]:
        return {"version": self.version or 0}

    def openapi_key(self) -> str:
        return f"user:oas-{self.name}"

    async def save_schema(self, ctx: ToolContext) -> int:
        async with self._lock:
            artifact = types.Part.from_text(text=self.dump())
            meta = self.meta()
            self.version = await ctx.save_artifact(self.openapi_key(), artifact, meta)

        return self.version

    async def load_schema(self, ctx: ToolContext) -> dict[str, Any]:
        async with self._lock:
            artifact = await ctx.load_artifact(filename=self.openapi_key())
            if artifact is None:
                return openapi_base_schema
            self.schema = yaml.safe_load(artifact.text)

        return self.schema

    async def update_schema(self, diff: dict[str, Any], ctx: ToolContext) -> DictDiff:
        await self.load_schema(ctx)

        schema = copy.deepcopy(self.schema)
        schema = deep_merge(schema, diff)
        schema_diff: DictDiff = dict_diff(self.schema, schema)

        self.schema = schema

        await self.save_schema(ctx)

        return schema_diff


def validate_model(model, item: dict[str, Any]) -> tuple[bool, Optional[str]]:
    try:
        model.model_validate(item)
    except ValidationError as exc:
        schema = json.dumps(model.model_json_schema())
        err = COMPONENT_VALIDATION_ERROR.format(
            exception=str(exc),
            component=model.__name__,
            schema=schema,
        )
        return False, err
    return True, None


def validate_files(
    files: list[str], ext: list[str] = [".json", ".md", ".yaml", ".yml"]
) -> Optional[str]:
    if not files:
        return FILE_VALIDATION_NO_FILES_PROVIDED

    banned_ext = tuple(ext)
    banned = [f for f in files if f.endswith(banned_ext)]

    if banned:
        return FILE_VALIDATION_BANNED_EXTENSIONS.format(
            extensions=",".join(ext),
            banned="\n".join(banned),
        )

    return None


def openapi_tools(name: str) -> list[Callable]:
    oas = OpenApiArtifact(name=name)

    async def upsert_path(
        path: str,
        path_def: PathItem,
        path_files: list[str],
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        """
        Upsert a path definition in the OpenAPI schema
        Args:
            path: str
                The path to update
            path_def: PathItem
                The path definition to update
            files: list[str]
                REQUIRED: The list of files in the project where the path is defined
        """

        if err := validate_files(path_files):
            return {"error": err}

        ok, err = validate_model(PathItem, path_def)
        if not ok:
            return {"error": err}

        diff = {"paths": {path.strip(): path_def}}
        schema_diff: DictDiff = await oas.update_schema(diff, tool_context)
        return {"result": asdict(schema_diff)}

    async def remove_path(path: str, tool_context: ToolContext) -> dict[str, Any]:
        """
        Remove an existing API path from the schema.
        Args:
            path (str): API path to remove
        """

        schema = await oas.load_schema(tool_context)
        current = copy.deepcopy(schema)

        if "paths" in oas.schema and path in oas.schema["paths"]:
            del oas.schema["paths"][path]
        else:
            return {"error": PATH_NOT_FOUND_OR_ALREADY_REMOVED.format(path=path)}

        diff = dict_diff(oas.schema, current)
        await oas.save_schema(tool_context)

        return {"result": asdict(diff)}

    async def list_paths(tool_context: ToolContext) -> dict[str, Any]:
        """
        List all API paths defined in the schema.
        """

        schema = await oas.load_schema(tool_context)
        schema.setdefault("paths", {})
        return {"result": list(schema["paths"].keys())}

    async def get_path(path: str, tool_context: ToolContext) -> dict[str, Any]:
        """
        Retrieve a specific API path definition.
        Args:
            path (str): API path to retrieve
        """

        schema = await oas.load_schema(tool_context)
        schema.setdefault("paths", {})

        if path.strip() not in schema["paths"]:
            return {
                "error": PATH_NOT_FOUND_OR_ALREADY_REMOVED.format(path=path.strip())
            }

        return {"result": schema["paths"][path.strip()]}

    async def upsert_component(
        key: Literal[
            "schemas", "securitySchemes", "requestBodies", "headers", "responses"
        ],
        name: str,
        component_def: dict[str, Any],
        component_files: list[str],
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        """
        Upsert (create or update) an OpenAPI component definition.
        Args:
            key (Literal[schemas, securitySchemes, requestBodies, headers, responses]): Component type
            name (str): The name of the component
            component_def (dict): Component definition
            component_files (list[str]): Required list of project files where the component is defined
        """

        allowed_keys = {
            "schemas",
            "securitySchemes",
            "requestBodies",
            "headers",
            "responses",
        }
        if key not in allowed_keys:
            keys: str = ",".join(allowed_keys)
            return {"error": COMPONENT_KEY_ERROR.format(key=key, keys=keys)}

        if err := validate_files(component_files):
            return {"error": err}

        match key:
            case "securitySchemes":
                ok, err = validate_model(SecurityScheme, component_def)
                if not ok:
                    return {"error": err}
            case "requestBodies":
                ok, err = validate_model(RequestBody, component_def)
                if not ok:
                    return {"error": err}
            case "responses":
                ok, err = validate_model(Response, component_def)
                if not ok:
                    return {"error": err}

        diff = {"components": {key: {name: component_def}}}
        schema_diff: DictDiff = await oas.update_schema(diff, tool_context)
        return {"result": asdict(schema_diff)}

    async def remove_component(
        key: Literal[
            "schemas", "securitySchemes", "requestBodies", "headers", "responses"
        ],
        component_name: str,
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        """
        Remove an existing OpenAPI component.
        Args:
            key (Literal[schemas, securitySchemes, requestBodies, headers, responses]): Component type
            component_name (str): Component name
        """

        schema = await oas.load_schema(tool_context)
        current = copy.deepcopy(schema)

        if (
            "components" in oas.schema
            and key in oas.schema["components"]
            and component_name in oas.schema["components"][key]
        ):
            del oas.schema["components"][key][component_name]
        else:
            return {
                "error": COMPONENT_NOT_FOUND_OR_ALREADY_REMOVED.format(
                    name=component_name, key=key
                )
            }

        diff = dict_diff(oas.schema, current)
        await oas.save_schema(tool_context)

        return {"result": asdict(diff)}

    async def list_components(
        key: Literal[
            "schemas", "securitySchemes", "requestBodies", "headers", "responses"
        ],
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        """
        List component names for a given component type.
        """

        allowed_keys = {
            "schemas",
            "securitySchemes",
            "requestBodies",
            "headers",
            "responses",
        }
        if key not in allowed_keys:
            keys: str = ",".join(allowed_keys)
            return {"error": COMPONENT_KEY_ERROR.format(key=key, keys=keys)}

        schema = await oas.load_schema(tool_context)
        schema.setdefault("components", {})
        schema["components"].setdefault(key, {})

        return {"result": list(schema["components"][key].keys())}

    async def get_component(
        key: Literal[
            "schemas", "securitySchemes", "requestBodies", "headers", "responses"
        ],
        component_name: str,
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        """
        Retrieve a specific component definition.
        Args:
            key (Literal[schemas, securitySchemes, requestBodies, headers, responses]): Component type
            component_name (str): Component name
        """

        await oas.load_schema(tool_context)

        if not (
            "components" in oas.schema
            and key in oas.schema["components"]
            and component_name in oas.schema["components"][key]
        ):
            return {
                "error": COMPONENT_NOT_FOUND_OR_ALREADY_REMOVED.format(
                    name=component_name, key=key
                )
            }

        return {"result": oas.schema["components"][key][component_name]}

    async def set_info(
        title: str,
        framework: Optional[str],
        code_language: Optional[str],
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        """
        Set info section of the OpenAPI schema
        Args:
            title: The title of the API
            framework: The framework used to build service
            code_language: The code languague used to build service backend
        Returns:
            dict: result of the operation
        """

        extra: dict[str, Any] = dict()
        if framework is not None:
            extra["x-framework"] = framework
        if code_language is not None:
            extra["x-code-language"] = code_language

        schema = await oas.load_schema(tool_context)
        current = copy.deepcopy(schema)

        oas.schema["info"] = {"title": title, **extra}
        diff = dict_diff(current, oas.schema)
        await oas.save_schema(tool_context)

        return {"result": asdict(diff)}

    async def get_info(tool_context: ToolContext) -> dict[str, Any]:
        """
        Retrieve the OpenAPI info section.
        """

        schema = await oas.load_schema(tool_context)
        schema.setdefault("info", {})
        return {"result": schema["info"]}

    async def add_server(
        url: str, description: Optional[str], tool_context: ToolContext
    ) -> dict[str, Any]:
        """
        Add a new server entry to the OpenAPI schema.
        Args:
            url (str): Server URL
            description (Optional[str]): Server description
        """

        schema = await oas.load_schema(tool_context)
        current = copy.deepcopy(schema)

        oas.schema.setdefault("servers", [])
        servers = oas.schema["servers"]

        if any(url.strip() == server.get("url") for server in servers):
            return {"error": SERVER_ALREADY_EXISTS.format(url=url.strip())}

        oas.schema["servers"].append(
            {"url": url.strip(), "description": description or ""}
        )
        diff = dict_diff(current, oas.schema)
        await oas.save_schema(tool_context)
        return {"result": asdict(diff)}

    async def remove_server(url: str, tool_context) -> dict[str, Any]:
        """
        Remove a server entry from the schema.
        """

        schema = await oas.load_schema(tool_context)
        current = copy.deepcopy(schema)

        oas.schema.setdefault("servers", [])
        servers = oas.schema["servers"]

        if not any(url.strip() == server.get("url") for server in servers):
            return {"error": SERVER_NOT_EXISTS.format(url=url.strip())}

        servers = [server for server in servers if not server.get("url") == url.strip()]
        oas.schema["servers"] = servers

        diff = dict_diff(current, oas.schema)
        await oas.save_schema(tool_context)
        return {"result": asdict(diff)}

    async def list_servers(tool_context: ToolContext) -> dict[str, Any]:
        """
        List all configured servers.
        """

        schema = await oas.load_schema(tool_context)
        schema.setdefault("servers", [])
        return {"result": schema["servers"]}

    async def get_full_openapi_schema(tool_context: ToolContext) -> dict[str, Any]:
        """
        Retrieve the complete OpenAPI schema.
        IMPORTANT: This is very heavy operation, use it when absoutely necessary.
        """

        await oas.load_schema(tool_context)
        return {"result": oas.dump()}

    return [
        list_paths,
        list_components,
        list_servers,
        get_info,
        get_path,
        get_component,
        set_info,
        add_server,
        upsert_path,
        upsert_component,
        remove_server,
        remove_path,
        remove_component,
        get_full_openapi_schema,
    ]
