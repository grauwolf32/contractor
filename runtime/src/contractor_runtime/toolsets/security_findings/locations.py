"""Closed source/web location contract shared by authoring and collection reads."""

from __future__ import annotations

import re
import unicodedata
from typing import Annotated, Any
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema

from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.toolsets.http.limits import MAX_URL_BYTES

# V63-001 / docs/spec/27: match the retained Go contract, not tool display limits.
MAX_LOCATIONS = 256  # Same bound as finding evidence links.
MAX_PATH_BYTES = 4096
MAX_METHOD_BYTES = 64
MAX_EXACT_INTEGER = 2**53 - 1  # JSON/JavaScript lossless integer boundary.
HTTP_TOKEN = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
INVALID_PERCENT_ESCAPE = re.compile(r"%(?![0-9a-fA-F]{2})")
LineNumber = Annotated[int, Field(strict=True, ge=1, le=MAX_EXACT_INTEGER)]


class ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ExactEvidenceRef(ArtifactRef):
    revision: str = Field(min_length=1)


class StandardReference(ClosedModel):
    scheme: str
    version: str
    requirement_id: str


class LineRange(ClosedModel):
    start_line: LineNumber
    end_line: LineNumber

    @model_validator(mode="after")
    def check_order(self) -> LineRange:
        if self.end_line < self.start_line:
            raise ValueError("end_line must be >= start_line")
        return self


class SourceLocation(ClosedModel):
    file: str
    line: LineNumber | SkipJsonSchema[None] = None
    range: LineRange | SkipJsonSchema[None] = None

    @field_validator("file")
    @classmethod
    def check_path(cls, value: str) -> str:
        if not value or len(value.encode("utf-8")) > MAX_PATH_BYTES:
            raise ValueError("file must be a bounded relative POSIX path")
        if "\\" in value or ":" in value or any(_is_control(char) for char in value):
            raise ValueError("file must be a relative POSIX path without control characters")
        if any(part in {"", ".", ".."} for part in value.split("/")):
            raise ValueError("file cannot contain empty, dot or parent components")
        return value

    @model_validator(mode="before")
    @classmethod
    def check_optional_coordinates(cls, value: Any) -> Any:
        if isinstance(value, dict):
            for name in ("line", "range"):
                if name in value and value[name] is None:
                    raise ValueError(f"omit {name} instead of passing null")
        return value

    @model_validator(mode="after")
    def check_region(self) -> SourceLocation:
        if self.line is not None and self.range is not None:
            raise ValueError("provide line or range, not both")
        return self


class WebLocation(ClosedModel):
    url: str
    method: str | SkipJsonSchema[None] = None

    @field_validator("url")
    @classmethod
    def check_url(cls, value: str) -> str:
        validate_web_url(value)
        return value

    @field_validator("method")
    @classmethod
    def check_method(cls, value: str | None) -> str:
        if value is None or len(value) > MAX_METHOD_BYTES or not HTTP_TOKEN.fullmatch(value):
            raise ValueError("method must be an HTTP token; omit it when unknown")
        return value


def validate_web_url(value: str) -> None:
    if len(value.encode("utf-8")) > MAX_URL_BYTES or "\\" in value:
        raise ValueError("url must be a bounded absolute HTTP/HTTPS URL")
    if any(char.isspace() or _is_control(char) for char in value):
        raise ValueError("url cannot contain whitespace or control characters")
    if INVALID_PERCENT_ESCAPE.search(value):
        raise ValueError("url contains an invalid percent escape")
    try:
        parsed = urlsplit(value)
        valid = (
            parsed.scheme in {"http", "https"}
            and bool(parsed.hostname)
            and parsed.username is None
            and not parsed.netloc.endswith(":")
            and parsed.port != 0
        )
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("url requires an HTTP/HTTPS host, valid port and no userinfo")


def _is_control(character: str) -> bool:
    return unicodedata.category(character) == "Cc"


Location = SourceLocation | WebLocation


def normalize_locations(values: list[Location | dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(values, list) or len(values) > MAX_LOCATIONS:
        raise ValueError("locations exceeds the finding location limit")
    result = []
    for value in values:
        if isinstance(value, (SourceLocation, WebLocation)):
            location = value
        elif isinstance(value, dict) and "file" in value:
            location = SourceLocation.model_validate(value)
        else:
            location = WebLocation.model_validate(value)
        result.append(location.model_dump(exclude_none=True))
    return result
