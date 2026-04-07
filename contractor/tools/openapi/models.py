from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Literal, Optional

# ==================== DATA MODELS ====================


class SecurityScheme(BaseModel):
    """
    Defines a security scheme that can be used by the operations.
    """

    type: Literal["apiKey", "http", "mutualTLS", "oauth2", "openIdConnect"]

    description: Optional[str] = Field(
        description="A short description for security scheme.",
        default=None,
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
            "**REQUIRED** for `http` with the value `basic`.\n"
            "The name of the HTTP Authorization scheme to be used in the Authorization header."
        ),
        default=None,
    )

    bearerFormat: Optional[str] = Field(  # noqa: N815
        description="A hint to the client to identify how the bearer token is formatted.",
        default=None,
    )

    flows: Optional[dict[str, Any]] = Field(
        description="**REQUIRED** for `oauth2`.",
        default=None,
    )

    openIdConnectUrl: Optional[str] = Field(  # noqa: N815
        description="**REQUIRED** for `openIdConnect`.",
        default=None,
    )

    model_config = ConfigDict(
        extra="allow",
        validate_by_name=True,
        json_schema_extra={
            "examples": [
                {"type": "http", "scheme": "basic"},
                {"type": "apiKey", "name": "api_key", "in": "header"},
                {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"},
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
        },
    )


class Response(BaseModel):
    """
    Describes a single response from an API Operation.
    """

    description: str = Field(
        description="**REQUIRED**. A short description of the response."
    )

    headers: Optional[dict[str, Any]] = Field(default=None)
    content: Optional[dict[str, Any]] = Field(default=None)
    links: Optional[dict[str, str]] = Field(default=None)

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
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
                            "description": "The number of allowed requests",
                            "schema": {"type": "integer"},
                        },
                        "X-Rate-Limit-Remaining": {
                            "description": "The number of remaining requests",
                            "schema": {"type": "integer"},
                        },
                        "X-Rate-Limit-Reset": {
                            "description": "The number of seconds left",
                            "schema": {"type": "integer"},
                        },
                    },
                },
                {"description": "object created"},
            ]
        },
    )


class RequestBody(BaseModel):
    """Describes a single request body."""

    description: Optional[str] = Field(default=None)
    content: dict[str, Any]
    required: bool = Field(default=False)

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
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
                            "schema": {
                                "type": "array",
                                "items": {"type": "string"},
                            }
                        }
                    },
                },
            ]
        },
    )


class Operation(BaseModel):
    """Describes a single API operation on a path."""

    tags: Optional[list[str]] = None
    summary: Optional[str] = None
    description: Optional[str] = None
    externalDocs: Optional[dict[str, str]] = None
    operationId: str
    parameters: Optional[list[dict[str, Any]]] = None
    requestBody: Optional[dict[str, Any]] = None
    responses: Optional[dict[str, Any]] = None
    callbacks: Optional[dict[str, Any]] = None
    deprecated: bool = False
    security: Optional[list[dict[str, Any]]] = None

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        json_schema_extra={
            "examples": [
                {
                    "tags": ["pet"],
                    "summary": "Updates a pet in the store with form data",
                    "operationId": "updatePetWithForm",
                    "parameters": [
                        {
                            "name": "petId",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Pet updated.",
                            "content": {
                                "application/json": {},
                                "application/xml": {},
                            },
                        },
                        "405": {
                            "description": "Method Not Allowed",
                            "content": {
                                "application/json": {},
                                "application/xml": {},
                            },
                        },
                    },
                    "security": [{"petstore_auth": ["write:pets", "read:pets"]}],
                }
            ]
        },
    )


class PathItem(BaseModel):
    """
    Describes the operations available on a single path.
    """

    ref: Optional[str] = Field(default=None, alias="$ref")
    summary: Optional[str] = None
    description: Optional[str] = None

    get: Optional[Operation] = None
    put: Optional[Operation] = None
    post: Optional[Operation] = None
    delete: Optional[Operation] = None
    options: Optional[Operation] = None
    head: Optional[Operation] = None
    patch: Optional[Operation] = None
    trace: Optional[Operation] = None

    parameters: Optional[list[dict[str, Any]]] = None

    model_config = ConfigDict(
        extra="allow",
        validate_by_name=True,
        json_schema_extra={
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
                            "required": True,
                            "schema": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "style": "simple",
                        }
                    ],
                }
            ]
        },
    )
