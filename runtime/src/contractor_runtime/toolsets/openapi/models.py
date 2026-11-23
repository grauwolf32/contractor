"""Focused OpenAPI 3 mutation-boundary models used by ``openapi@1``."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SecurityScheme(BaseModel):
    type: Literal["apiKey", "http", "mutualTLS", "oauth2", "openIdConnect"]
    description: str | None = None
    name: str | None = None
    location: Literal["query", "header", "cookie"] | None = Field(default=None, alias="in")
    scheme: str | None = None
    bearer_format: str | None = Field(default=None, alias="bearerFormat")
    flows: dict[str, Any] | None = None
    open_id_connect_url: str | None = Field(default=None, alias="openIdConnectUrl")

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    @model_validator(mode="after")
    def validate_conditional_fields(self) -> Self:
        if self.type == "apiKey" and (not self.name or self.location is None):
            raise ValueError("apiKey security scheme requires name and in")
        if self.type == "http" and not self.scheme:
            raise ValueError("http security scheme requires scheme")
        if self.type == "oauth2" and not self.flows:
            raise ValueError("oauth2 security scheme requires flows")
        if self.type == "openIdConnect" and not self.open_id_connect_url:
            raise ValueError("openIdConnect security scheme requires openIdConnectUrl")
        return self


class Response(BaseModel):
    description: str
    headers: dict[str, Any] | None = None
    content: dict[str, Any] | None = None
    links: dict[str, Any] | None = None

    model_config = ConfigDict(extra="allow")


class RequestBody(BaseModel):
    description: str | None = None
    content: dict[str, Any]
    required: bool = False

    model_config = ConfigDict(extra="allow")


class Operation(BaseModel):
    tags: list[str] | None = None
    summary: str | None = None
    description: str | None = None
    external_docs: dict[str, Any] | None = Field(default=None, alias="externalDocs")
    operation_id: str | None = Field(default=None, alias="operationId")
    parameters: list[dict[str, Any]] | None = None
    request_body: dict[str, Any] | None = Field(default=None, alias="requestBody")
    responses: dict[str, Any]
    callbacks: dict[str, Any] | None = None
    deprecated: bool = False
    security: list[dict[str, Any]] | None = None

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    @model_validator(mode="after")
    def validate_responses(self) -> Self:
        if not self.responses:
            raise ValueError("operation responses must not be empty")
        for status, response in self.responses.items():
            if not isinstance(status, str) or not status:
                raise ValueError("response status keys must be non-empty strings")
            if not isinstance(response, dict):
                raise ValueError("response values must be objects")
            if "$ref" not in response:
                Response.model_validate(response)
        return self


class PathItem(BaseModel):
    ref: str | None = Field(default=None, alias="$ref")
    summary: str | None = None
    description: str | None = None
    get: Operation | None = None
    put: Operation | None = None
    post: Operation | None = None
    delete: Operation | None = None
    options: Operation | None = None
    head: Operation | None = None
    patch: Operation | None = None
    trace: Operation | None = None
    parameters: list[dict[str, Any]] | None = None

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    @model_validator(mode="after")
    def validate_has_operation_or_ref(self) -> Self:
        operations = (
            self.get,
            self.put,
            self.post,
            self.delete,
            self.options,
            self.head,
            self.patch,
            self.trace,
        )
        if self.ref is None and all(operation is None for operation in operations):
            raise ValueError("path item requires at least one operation or $ref")
        return self
