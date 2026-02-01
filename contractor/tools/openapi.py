from __future__ import annotations

import json
import yaml
import copy
from typing import Any, Literal, Optional, Final, Callable
from dataclasses import dataclass, field, asdict
from pydantic import BaseModel, Field, ValidationError

from google.genai import types
from google.adk.tools.tool_context import ToolContext

from contractor.utils import deep_merge, dict_diff, DictDiff

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

    async def load_schema(self, ctx: ToolContext) -> dict[str, Any]:
        artifact = await ctx.load_artifact(self.name)
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


def validate_model(model, item: dict[str, Any]) -> Optional[str]:
    try:
        model.validate_model(item)
    except ValidationError as exc:
        schema = json.dumps(model.model_json_schema())
        err = COMPONENT_VALIDATION_ERROR.format(
            exception=str(exc), component=model.__name__, schema=schema
        )
        return False, err
    return True, None


def validate_files(
    files: list[str], ext: list[str] = [".json", ".md", ".yaml", ".yml"]
) -> Optional[str]:
    banned = [f for f in files if f.endswith(ext)]
    if banned:
        banned = "\n".join(banned)
        return FILE_VALIDATION_BANNED_EXTENSIONS.format(
            extensions=",".join(ext), banned=banned
        )

    if not files:
        return FILE_VALIDATION_NO_FILES_PROVIDED
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

        if err := validate_model(PathItem, path_def):
            return {"error": err}

        diff = {"paths": {path.strip(): path_def}}
        schema_diff: DictDiff = await oas.update_schema(diff, tool_context)
        return {"result": asdict(schema_diff)}

    async def remove_path(path: str, tool_context: ToolContext) -> dict[str, Any]:
        schema = oas.load_schema(tool_context)
        current = copy.deepcopy(schema)

        if "paths" in oas.schema and path in oas.schema["paths"]:
            del oas.schema["paths"][path]
        else:
            return {"error": PATH_NOT_FOUND_OR_ALREADY_REMOVED.format(path=path)}

        diff = dict_diff(oas.schema, current)
        await oas.save_schema(tool_context)

        return {"result": asdict(diff)}

    async def list_paths(tool_context: ToolContext) -> dict[str, Any]:
        schema = await oas.load_schema(tool_context)
        schema.setdefault("paths", {})
        return {"result": list(schema["paths"].keys())}

    async def get_path(path: str, tool_context: ToolContext) -> dict[str.Any]:
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
        component_def: dict[str, Any],
        component_files: list[str],
        tool_context: ToolContext,
    ) -> dict[str, Any]:
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
                if err := validate_model(SecurityScheme, component_def):
                    return {"error": err}
            case "requestBodies":
                if err := validate_model(RequestBody, component_def):
                    return {"error": err}
            case "responses":
                if err := validate_model(Response, component_def):
                    return {"error": err}

        diff = {"components": {key: component_def}}
        schema_diff: DictDiff = await oas.update_schema(diff, tool_context)
        return {"result": asdict(schema_diff)}

    async def remove_component(
        key: Literal[
            "schemas", "securitySchemes", "requestBodies", "headers", "responses"
        ],
        component_name: str,
        tool_context: ToolContext,
    ) -> dict[str, Any]:
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
        return {"result": list(schema["components"][key].keys())}

    async def get_component(
        key: Literal[
            "schemas", "securitySchemes", "requestBodies", "headers", "responses"
        ],
        component_name: str,
        tool_context: ToolContext,
    ) -> dict[str, Any]:
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
        schema = await oas.load_schema(tool_context)
        schema.setdefault("info", {})
        return {"result": schema["info"]}

    async def add_server(
        url: str, description: Optional[str], tool_context: ToolContext
    ) -> dict[str, Any]:
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
        schema = await oas.load_schema(tool_context)
        schema.setdefault("servers", [])
        return {"result": schema["servers"]}

    async def get_full_openapi_schema(tool_context: ToolContext) -> dict[str, Any]:
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


# ==================== DATA MODELS ====================


class SecurityScheme(BaseModel):
    """
    Defines a security scheme that can be used by the operations.

    Supported schemes are HTTP authentication,
    an API key (either as a header, a cookie parameter or as a query parameter),
    mutual TLS (use of a client certificate),
    OAuth2's common flows (implicit, password, client credentials and authorization code)
    as defined in [RFC6749](https://tools.ietf.org/html/rfc6749),
    and [OpenID Connect Discovery](https://tools.ietf.org/html/draft-ietf-oauth-discovery-06).
    """

    type: Literal["apiKey", "http", "mutualTLS", "oauth2", "openIdConnect"] = ...

    description: Optional[str] = Field(
        description="A short description for security scheme.", default=None
    )
    name: Optional[str] = Field(
        description="**REQUIRED** for `apiKey`. The name of the header, query or cookie parameter to be used.",
        default=None,
    )
    security_scheme_in: Optional[Literal["query", "header", "cookie"]] = Field(
        alias="in",
        description="**REQUIRED** for `apiKey`. The location of the API key.",
        default=None,
    )

    scheme: Optional[str] = Field(
        description=(
            """
        **REQUIRED** for `http` with the value `basic`.
        The name of the HTTP Authorization scheme to be used in the
        [Authorization header as defined in RFC7235](https://tools.ietf.org/html/rfc7235#section-5.1).
        
        The values used SHOULD be registered in the
        [IANA Authentication Scheme registry](https://www.iana.org/assignments/http-authschemes/http-authschemes.xhtml).
        """
        ),
        default=None,
    )

    bearerFormat: Optional[str] = Field(  # noqa: N815
        description="A hint to the client to identify how the bearer token is formatted. Bearer tokens are usually generated by an authorization server, so this information is primarily for documentation purposes.",
        default=None,
    )
    flows: Optional[dict[str, Any]] = Field(
        description="**REQUIRED** for `oauth2`. An object containing configuration information for the flow types supported.",
        default=None,
    )

    openIdConnectUrl: Optional[str] = Field(  # noqa: N815
        description=(
            """
            **REQUIRED** for `openIdConnect`. OpenId Connect URL to discover OAuth2 configuration values.
            This MUST be in the form of a URL. The OpenID Connect standard requires the use of TLS.
            """
        ),
        default=None,
    )

    class Config:
        extra = "allow"
        allow_population_by_field_name = True
        schema_extra = {
            "examples": [
                {
                    "type": "http",
                    "scheme": "basic",
                },
                {
                    "type": "apiKey",
                    "name": "api_key",
                    "in": "header",
                },
                {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT",
                },
                {
                    "type": "oauth2",
                    "flows": {
                        "implicit": {
                            "authorizationUrl": "https://example.com/api/oauth/dialog",
                            "scopes": {
                                "write:pets": "modify pets in your account",
                                "read:pets": "read your pets",
                            },
                        }
                    },
                },
                {
                    "type": "openIdConnect",
                    "openIdConnectUrl": "https://example.com/openIdConnect",
                },
                {
                    "type": "openIdConnect",
                    "openIdConnectUrl": "openIdConnect",
                },
            ]
        }


class Response(BaseModel):
    """
    Describes a single response from an API Operation, including design-time,
    static `links` to operations based on the response.
    """

    description: str = Field(
        description=(
            """
            **REQUIRED**. A short description of the response.
            """
        )
    )

    headers: Optional[dict[str, Any]] = Field(
        description=(
            """
            A map containing descriptions of potential response headers.
            """
        ),
        default=None,
    )

    content: Optional[dict[str, Any]] = Field(
        description=(
            """
            A map containing descriptions of potential response payloads.
            The key is a media type or [media type range](https://tools.ietf.org/html/rfc7231#appendix-D)
            and the value describes it.  
            
            For responses that match multiple keys, only the most specific key is applicable. e.g. text/plain overrides text/*
            """
        ),
        default=None,
    )

    links: Optional[dict[str, str]] = Field(
        description=(
            """
            A map of operations links that can be followed from the response.
            The key of the map is a short name for the link,
            following the naming constraints of the names for [Component Objects](#componentsObject).
            """
        )
    )

    class Config:
        extra = "allow"
        schema_extra = {
            "examples": [
                {
                    "description": "A complex object array response",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "array",
                                "items": {
                                    "$ref": "#/components/schemas/VeryComplexType"
                                },
                            }
                        }
                    },
                },
                {
                    "description": "A simple string response",
                    "content": {"text/plain": {"schema": {"type": "string"}}},
                },
                {
                    "description": "A simple string response",
                    "content": {
                        "text/plain": {"schema": {"type": "string", "example": "whoa!"}}
                    },
                    "headers": {
                        "X-Rate-Limit-Limit": {
                            "description": "The number of allowed requests in the current period",
                            "schema": {"type": "integer"},
                        },
                        "X-Rate-Limit-Remaining": {
                            "description": "The number of remaining requests in the current period",
                            "schema": {"type": "integer"},
                        },
                        "X-Rate-Limit-Reset": {
                            "description": "The number of seconds left in the current period",
                            "schema": {"type": "integer"},
                        },
                    },
                },
                {
                    "description": "object created",
                },
            ]
        }


class RequestBody(BaseModel):
    """Describes a single request body."""

    description: Optional[str] = Field(
        description=(
            "A brief description of the request body."
            "This could contain examples of use."
            "CommonMark syntax MAY be used for rich text representation."
        )
    )

    content: dict[str, Any] = Field(
        description=(
            "**REQUIRED**. The content of the request body."
            "The key is a media type or [media type range](https://tools.ietf.org/html/rfc7231#appendix-D)"
            "and the value describes it."
            "For requests that match multiple keys, only the most specific key is applicable. e.g. text/plain overrides text/*"
        )
    )

    required: bool = Field(
        description=(
            "Determines if the request body is required in the request.Default is `false`."
        ),
        default=False,
    )

    class Config:
        extra = "allow"
        schema_extra = {
            "examples": [
                {
                    "description": "user to add to the system",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/User"},
                            "examples": {
                                "user": {
                                    "summary": "User Example",
                                    "externalValue": "http://foo.bar/examples/user-example.json",
                                }
                            },
                        },
                        "application/xml": {
                            "schema": {"$ref": "#/components/schemas/User"},
                            "examples": {
                                "user": {
                                    "summary": "User example in XML",
                                    "externalValue": "http://foo.bar/examples/user-example.xml",
                                }
                            },
                        },
                        "text/plain": {
                            "examples": {
                                "user": {
                                    "summary": "User example in Plain text",
                                    "externalValue": "http://foo.bar/examples/user-example.txt",
                                }
                            }
                        },
                        "*/*": {
                            "examples": {
                                "user": {
                                    "summary": "User example in other format",
                                    "externalValue": "http://foo.bar/examples/user-example.whatever",
                                }
                            }
                        },
                    },
                },
                {
                    "description": "user to add to the system",
                    "content": {
                        "text/plain": {
                            "schema": {"type": "array", "items": {"type": "string"}}
                        }
                    },
                },
            ]
        }


class Operation(BaseModel):
    """Describes a single API operation on a path."""

    tags: Optional[list[str]] = Field(
        description=(
            """
        A list of tags for API documentation control.
        Tags can be used for logical grouping of operations by resources or any other qualifier.
        """
        ),
        default=None,
    )

    summary: Optional[str] = Field(
        description=(
            """
            A short summary of what the operation does.
            """
        ),
        default=None,
    )
    description: Optional[str] = Field(
        description=(
            """
            A verbose explanation of the operation behavior.
            [CommonMark syntax](https://spec.commonmark.org/) MAY be used for rich text representation.
            """
        ),
        default=None,
    )

    externalDocs: Optional[dict[str, str]] = Field(  # noqa: N815
        description=(
            """
            Additional external documentation for this operation.
            """
        ),
        default=None,
    )

    operationId: str = Field(  # noqa: N815
        description=(
            """
            Unique string used to identify the operation.
            The id MUST be unique among all operations described in the API.
            The operationId value is case-sensitive.
            Tools and libraries MAY use the operationId to uniquely identify an operation, therefore, it is RECOMMENDED to follow common programming naming conventions.
            """
        )
    )

    parameters: Optional[list[dict[str, Any]]] = Field(
        description=(
            """
            A list of parameters that are applicable for this operation.
            If a parameter is already defined at the [Path Item](#pathItem), the new definition will override it but can never remove it.
            The list MUST NOT include duplicated parameters.
            A unique parameter is defined by a combination of a name and location.
            The list can use the [Reference Object](#referenceObject) to link to parameters that are defined at the [OpenAPI Object's components/parameters](#componentsParameters).
            """
        ),
        default=None,
    )

    requestBody: Optional[dict[str, Any]] = Field(  # noqa: N815
        description=(
            """
            The request body applicable for this operation.  
    
            The `requestBody` is fully supported in HTTP methods where the HTTP 1.1 specification
            [RFC7231](https://tools.ietf.org/html/rfc7231#section-4.3.1) has explicitly defined semantics for request bodies.

            In other cases where the HTTP spec is vague (such as [GET](https://tools.ietf.org/html/rfc7231#section-4.3.1),
            [HEAD](https://tools.ietf.org/html/rfc7231#section-4.3.2) and [DELETE](https://tools.ietf.org/html/rfc7231#section-4.3.5)),
            `requestBody` is permitted but does not have well-defined semantics and SHOULD be avoided if possible.
            """
        ),
        default=None,
    )

    responses: Optional[dict[str, any]] = Field(
        description=(
            """
            The map of possible responses as they are returned from executing this operation.
            Also could be a map of references to a responses defined in the [Components Object](#componentsResponses).
            """
        ),
        default=None,
    )

    callbacks: Optional[dict[str, Any]] = Field(
        description=(
            """
            A map of possible out-of band callbacks related to the parent operation.
            Each value in the map is a Path Item Object that describes a set of requests that may be initiated by the API provider and the expected responses.
            The key value used to identify the path item object is an expression, evaluated at runtime, that identifies a URL to use for the callback operation.
            """
        ),
        default=None,
    )

    deprecated: bool = Field(
        description=(
            """
            Declares this operation to be deprecated.
            Consumers SHOULD refrain from usage of the declared operation.
            Default value is `false`.
            """
        ),
        default=False,
    )

    security: Optional[list[dict[str, Any]]] = Field(
        description=(
            """
            A declaration of which security mechanisms can be used for this operation.
            The list of values includes alternative security requirement objects that can be used.
            Only one of the security requirement objects need to be satisfied to authorize a request.
            This definition overrides any declared top-level [security](#openapi-security)
            To remove a top-level security declaration, an empty array can be used.
            """
        ),
        default=None,
    )

    class Config:
        extra = "allow"
        arbitrary_types_allowed = True
        schema_extra = {
            "examples": [
                {
                    "tags": ["pet"],
                    "summary": "Updates a pet in the store with form data",
                    "operationId": "updatePetWithForm",
                    "parameters": [
                        {
                            "name": "petId",
                            "in": "path",
                            "description": "ID of pet that needs to be updated",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "requestBody": {
                        "content": {
                            "application/x-www-form-urlencoded": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "name": {
                                            "description": "Updated name of the pet",
                                            "type": "string",
                                        },
                                        "status": {
                                            "description": "Updated status of the pet",
                                            "type": "string",
                                        },
                                    },
                                    "required": ["status"],
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Pet updated.",
                            "content": {"application/json": {}, "application/xml": {}},
                        },
                        "405": {
                            "description": "Method Not Allowed",
                            "content": {"application/json": {}, "application/xml": {}},
                        },
                    },
                    "security": [{"petstore_auth": ["write:pets", "read:pets"]}],
                }
            ]
        }


class PathItem(BaseModel):
    """
    Describes the operations available on a single path.
    A Path Item MAY be empty, due to [ACL constraints](#securityFiltering).
    The path itself is still exposed to the documentation viewer
    but they will not know which operations and parameters are available.
    """

    ref: Optional[str] = Field(
        description=(
            "Allows for an external definition of this path item."
            "The referenced structure MUST be in the format of a [Path Item Object](#pathItemObject)."
            "In case a Path Item Object field appears both in the defined object and the referenced object,"
            "the behavior is undefined."
        ),
        default=None,
        alias="$ref",
    )

    summary: Optional[str] = Field(
        description="An optional, string summary, intended to apply to all operations in this path.",
        default=None,
    )
    description: Optional[str] = Field(
        description="An optional, string description, intended to apply to all operations in this path.",
        default=None,
    )

    get: Optional[Operation] = Field(
        description="A definition of a GET operation on this path.", default=None
    )
    put: Optional[Operation] = Field(
        description="A definition of a PUT operation on this path.", default=None
    )
    post: Optional[Operation] = Field(
        description="A definition of a POST operation on this path.", default=None
    )
    delete: Optional[Operation] = Field(
        description="A definition of a DELETE operation on this path.", default=None
    )
    options: Optional[Operation] = Field(
        description="A definition of a OPTIONS operation on this path.", default=None
    )
    head: Optional[Operation] = Field(
        description="A definition of a HEAD operation on this path.", default=None
    )
    patch: Optional[Operation] = Field(
        description="A definition of a PATCH operation on this path.", default=None
    )
    trace: Optional[Operation] = Field(
        description="A definition of a TRACE operation on this path.", default=None
    )
    parameters: Optional[list[dict[str, Any]]] = Field(
        description=(
            "A list of parameters that are applicable for all the operations described under this path."
            "These parameters can be overridden at the operation level, but cannot be removed there."
            "The list MUST NOT include duplicated parameters. A unique parameter is defined by a combination of a name and location."
            "The list can use the [Reference Object](#referenceObject) to link to parameters that are defined at the OpenAPI Object's components/parameters."
        ),
        default=None,
    )

    class Config:
        extra = "allow"
        allow_population_by_field_name = True
        schema_extra = {
            "examples": [
                {
                    "get": {
                        "description": "Returns pets based on ID",
                        "summary": "Find pets by ID",
                        "operationId": "getPetsById",
                        "responses": {
                            "200": {
                                "description": "pet response",
                                "content": {
                                    "*/*": {
                                        "schema": {
                                            "type": "array",
                                            "items": {
                                                "$ref": "#/components/schemas/Pet"
                                            },
                                        }
                                    }
                                },
                            },
                            "default": {
                                "description": "error payload",
                                "content": {
                                    "text/html": {
                                        "schema": {
                                            "$ref": "#/components/schemas/ErrorModel"
                                        }
                                    }
                                },
                            },
                        },
                    },
                    "parameters": [
                        {
                            "name": "id",
                            "in": "path",
                            "description": "ID of pet to use",
                            "required": True,
                            "schema": {"type": "array", "items": {"type": "string"}},
                            "style": "simple",
                        }
                    ],
                }
            ]
        }
