"""Private Runtime protocol base models and validation."""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime
from enum import StrEnum
from typing import Annotated, ClassVar, Literal, Self
from urllib.parse import urlsplit

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    ValidationInfo,
    field_validator,
    model_validator,
)

API_VERSION = "contractor/v1alpha1"
ARTIFACT_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
# Opaque private Artifact list cursor (LIM-ART-5): unpadded base64url issued by
# the Server; api/testdata/v1alpha1/artifact-list-cases.json pins the grammar.
MAX_ARTIFACT_LIST_CURSOR_LENGTH = 512
ARTIFACT_LIST_CURSOR_PATTERN = re.compile(rf"[A-Za-z0-9_-]{{1,{MAX_ARTIFACT_LIST_CURSOR_LENGTH}}}")
# RFC 6838 restricted names in canonical lowercase, without parameters or
# wildcards, at most 255 characters as in the public MediaType schema;
# api/testdata/v1alpha1/media-type-cases.json pins the grammar and bound.
MEDIA_TYPE_PATTERN = re.compile(
    r"(?=.{3,255}\Z)[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*"
)
# Values that Operations shows. Tool names and logical Agent names use the
# public ConfigId grammar, instance and allocation IDs the public ResourceId
# grammar; api/testdata/v1alpha1/runtime-capability-cases.json pins them and
# the capability bounds to the Go contracts and the schemas.
NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
MAX_RESOURCE_ID_LENGTH = 256
RESOURCE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}")
MAX_RUNTIME_CAPABILITY_REFS = 128
MAX_RUNTIME_CAPABILITY_REF_LENGTH = 256
MAX_TOOLSET_CAPABILITY_TOOLS = 256
ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")
VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
NATIVE_SKILL_TOOL_NAMES = frozenset({"list_skills", "load_skill", "load_skill_resource"})
WORKER_SUBTASK_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
WORKER_FAILURE_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
RUN_METADATA_LABEL_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
MAX_RUN_METADATA_LABELS = 32
MAX_RUN_METADATA_LABEL_KEY_BYTES = 63
MAX_RUN_METADATA_LABEL_VALUE_BYTES = 256
MAX_WORKER_RESULT_BYTES = 64 * 1024
MAX_WORKER_FAILURE_MESSAGE_BYTES = 4 * 1024
MAX_WORKER_RESULT_ARTIFACTS = 128
MAX_WORKER_OBSERVATION_TOOLS = 256
MAX_WORKER_FILES_READ = 25
MAX_WORKER_COMPLETION_BYTES = 256 * 1024
MAX_AGENT_STATE_SNAPSHOT_BYTES = 4 * 1024 * 1024
MAX_STATE_WORKSPACE_PATHS = 10_000
MAX_STATE_WORKSPACE_PATH_BYTES = 2 * 1024 * 1024
# The relative workspace path grammar (WSP-R13): workspace targets and the
# files a Worker reports reading, and the larger Worker State paths.
MAX_WORKSPACE_TARGET_BYTES = 1024
MAX_WORKSPACE_TARGET_COMPONENTS = 32
MAX_WORKER_STATE_PATH_BYTES = 4096
MAX_WORKER_STATE_PATH_COMPONENTS = 128
MAX_UINT64 = 2**64 - 1
_STATE_METRIC_NAME_PATTERN = re.compile(r"^[a-z0-9_]+(?:\.[a-z0-9_]+)*$")
# Whitespace of the private wire's text checks: the Unicode White_Space
# characters, which the Server's strings.TrimSpace (unicode.IsSpace) trims.
# str.strip() and str.isspace() also treat the information separators
# U+001C..U+001F as whitespace, which the Server accepts as text.
# api/testdata/v1alpha1/blank-text-cases.json pins the set for both sides.
WHITESPACE = (
    "\t\n\v\f\r \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005"
    "\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000"
)


def _to_camel(value: str) -> str:
    first, *rest = value.split("_")
    return first + "".join(part.capitalize() for part in rest)


def trim_whitespace(value: str) -> str:
    """Remove leading and trailing WHITESPACE, as the Server's strings.TrimSpace does."""

    return value.strip(WHITESPACE)


def is_blank(value: str) -> bool:
    """Return whether value is empty or consists only of WHITESPACE."""

    return not trim_whitespace(value)


def json_byte_size(value: object) -> int:
    """Count compact UTF-8 JSON bytes, the measure of every private-wire byte bound.

    The Server measures the same with contracts.ResultJSONSize (WRK-R37): U+2028,
    U+2029 and other non-ASCII characters count their UTF-8 bytes, and only the
    escapes JSON requires count extra. api/testdata/v1alpha1/json-byte-size-cases.json
    pins the rule for both sides. NaN, infinities, unsupported types and lone
    surrogates raise ValueError or TypeError.
    """

    return len(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    )


def relative_path_problem(value: str, *, max_bytes: int, max_components: int) -> str | None:
    """Return why value is not a relative workspace path, or None when it is one (WSP-R13).

    A relative path is non-empty NFC UTF-8 of at most ``max_bytes`` bytes without a
    leading slash, backslash, NUL or URI; it has at most ``max_components``
    components, none empty, "." or ".." or holding a control character, and its
    first component is not a Windows drive (an ASCII letter and a colon). This is the
    one Python definition of the rule; the Server's contracts.RelativePathProblem
    applies it with the same problem texts, and
    api/testdata/v1alpha1/workspace-path-cases.json pins both. The byte bound is
    checked first, so the work is bounded by ``max_bytes``.
    """

    if not value:
        return "must not be empty"
    try:
        size = len(value.encode("utf-8"))
    except UnicodeEncodeError:
        return "must be valid UTF-8"
    if size > max_bytes:
        return f"exceeds the byte bound of {max_bytes} UTF-8 bytes"
    if value != unicodedata.normalize("NFC", value):
        return "must be NFC-normalized"
    if value.startswith("/") or "\\" in value or "\x00" in value or "://" in value:
        return "must be a relative POSIX path without a backslash, NUL or URI"
    parts = value.split("/")
    if len(parts) > max_components:
        return f"exceeds the depth bound of {max_components} components"
    for part in parts:
        if part in {"", ".", ".."}:
            return "contains an invalid component: empty, '.' or '..'"
        if any(ord(character) < 0x20 or ord(character) == 0x7F for character in part):
            return "contains a control character"
    first = parts[0]
    if len(first) >= 2 and first[0].isascii() and first[0].isalpha() and first[1] == ":":
        return "must not start with a Windows drive"
    return None


def _require_text(field: str, value: str, *, max_bytes: int | None = None) -> str:
    if is_blank(value):
        raise ValueError(f"{field} must not be empty")
    if max_bytes is not None and len(value.encode("utf-8")) > max_bytes:
        raise ValueError(f"{field} exceeds {max_bytes} UTF-8 bytes")
    return value


def _require_tool_arguments(value: dict[str, object]) -> None:
    try:
        size = json_byte_size(value)
    except (TypeError, ValueError) as error:
        raise ValueError("tool arguments are invalid") from error
    if size > 4096:
        raise ValueError("tool arguments exceed 4096 JSON bytes")


def _require_resource_id(field: str, value: str) -> str:
    if RESOURCE_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{field} must be a resource identifier of at most {MAX_RESOURCE_ID_LENGTH} characters"
        )
    return value


def _require_url(field: str, value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username is not None:
        raise ValueError(f"{field} must be an absolute HTTP(S) URL without user information")
    return value


def _require_aware_datetime(field: str, value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must include an offset")
    return value


class AgentObservedState(StrEnum):
    IDLE = "idle"
    ALLOCATED = "allocated"
    DRAINING = "draining"
    FENCED = "fenced"


class WorkerSessionMode(StrEnum):
    ISOLATED = "isolated"
    SHARED = "shared"


class ReconciliationAction(StrEnum):
    CONTINUE = "continue"
    DRAIN = "drain"
    RELEASE = "release"
    REREGISTER = "reregister"


def _bounded_completion(value):
    # The optional completion object is bounded before its strict model
    # validation (LIM-AWC-12). An unknown kind or phase is not dropped: the
    # model rejects it like any other malformed value (AWC-R115).
    if isinstance(value, dict) and json_byte_size(value) > 4096:
        raise ValueError("completion diagnostics exceed their bound")
    return value


def _require_state_workspace_path(value: str) -> str:
    if (
        relative_path_problem(
            value,
            max_bytes=MAX_WORKER_STATE_PATH_BYTES,
            max_components=MAX_WORKER_STATE_PATH_COMPONENTS,
        )
        is not None
    ):
        raise ValueError("Worker State workspace path is invalid")
    return value


RUNTIME_ADAPTER_REFS = frozenset({"caido-graphql@1", "http-proxy@1", "otlp-http@1"})
_RUNTIME_AGENT_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_HEADER_NAME_PATTERN = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_RUNTIME_ADAPTER_ERROR_CODES = frozenset(
    {
        "close_failed",
        "delivery_failed",
        "flush_failed",
        "flush_timeout",
        "queue_overflow",
        "request_failed",
    }
)
# Byte-for-byte the same expressions as internal/cabundle/validate.go. Every
# construct is ASCII-only, so Python re and Go RE2 agree on what they match.
_CERTIFICATE_BLOCK = (
    r"-----BEGIN CERTIFICATE-----\r?\n((?:[A-Za-z0-9+/=]+\r?\n)+)-----END CERTIFICATE-----"
)
_CERTIFICATE_BLOCK_PATTERN = re.compile(_CERTIFICATE_BLOCK)
_CA_BUNDLE_PATTERN = re.compile(
    r"(?:[\t\n\f\r ]*\n)?"
    + _CERTIFICATE_BLOCK
    + r"(?:[\t\n\f\r ]*\n"
    + _CERTIFICATE_BLOCK
    + r")*[\t\n\f\r ]*"
)
_FORBIDDEN_RUNTIME_HEADERS = frozenset(
    {
        "connection",
        "content-length",
        "host",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
RuntimeCredentialKind = Literal[
    "caido-bearer@1",
    "http-origin-basic@1",
    "http-origin-bearer@1",
    "http-proxy-basic@1",
    "http-proxy-bearer@1",
    "otlp-headers@1",
]
HTTPProxyTarget = Literal["llm-gateway", "tool-http", "tool-subprocess"]
WorkspaceMode = Literal["direct", "overlay"]
WorkspaceStorage = Literal["local", "memory"]


def _require_sorted_unique(field: str, values: list[str], *, maximum: int) -> None:
    if len(values) > maximum or values != sorted(set(values)):
        raise ValueError(f"{field} must be sorted, unique, and contain at most {maximum} items")


def _require_runtime_endpoint(field: str, value: str) -> str:
    parsed = urlsplit(value)
    if (
        value != trim_whitespace(value)
        or not 1 <= len(value.encode("utf-8")) <= 2048
        or parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or "?" in value
        or "#" in value
    ):
        raise ValueError(
            f"{field} must be a bounded absolute HTTP(S) URL without userinfo, query, or fragment"
        )
    return value


def _require_workspace_target(value: str) -> str:
    """Accept "" (the workspace root) or a relative path of the target grammar."""

    if value == "":
        return value
    problem = relative_path_problem(
        value,
        max_bytes=MAX_WORKSPACE_TARGET_BYTES,
        max_components=MAX_WORKSPACE_TARGET_COMPONENTS,
    )
    if problem is not None:
        raise ValueError(f"workspace target {problem}")
    return value


def _require_runtime_label(field: str, value: str) -> str:
    if (
        len(value.encode("ascii", errors="ignore")) != len(value)
        or not 1 <= len(value) <= 63
        or ID_PATTERN.fullmatch(value) is None
        or value == "default"
    ):
        raise ValueError(f"{field} contains an invalid Runtime label")
    return value


def _require_selector(field: str, value: str) -> str:
    if value.count("@") != 1:
        raise ValueError(f"{field} must use exact <id>@<version> syntax")
    identifier, version = value.split("@")
    if ID_PATTERN.fullmatch(identifier) is None or VERSION_PATTERN.fullmatch(version) is None:
        raise ValueError(f"{field} has an invalid exact selector")
    return value


def _require_capability_ref(field: str, value: str) -> str:
    if len(value) > MAX_RUNTIME_CAPABILITY_REF_LENGTH:
        raise ValueError(
            f"{field} must contain at most {MAX_RUNTIME_CAPABILITY_REF_LENGTH} characters"
        )
    return _require_selector(field, value)


def _require_digest(field: str, value: str) -> str:
    if DIGEST_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be sha256 followed by 64 lowercase hex characters")
    return value


def _require_worker_subtask_id(value: str) -> str:
    if (
        len(value.encode("ascii", errors="ignore")) != len(value)
        or not 1 <= len(value) <= 128
        or WORKER_SUBTASK_ID_PATTERN.fullmatch(value) is None
    ):
        raise ValueError("subtaskId is invalid")
    return value


def normalize_run_metadata_labels(value: dict[str, str]) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("runMetadataLabels must be an object")
    if len(value) > MAX_RUN_METADATA_LABELS:
        raise ValueError("runMetadataLabels exceed 32 entries")
    if any(not isinstance(key, str) for key in value):
        raise ValueError("runMetadataLabels contain an invalid key")
    result: dict[str, str] = {}
    for key in sorted(value):
        label_value = value[key]
        if (
            not key
            or len(key.encode("utf-8")) > MAX_RUN_METADATA_LABEL_KEY_BYTES
            or RUN_METADATA_LABEL_KEY_PATTERN.fullmatch(key) is None
            or key.startswith("contractor.")
        ):
            raise ValueError("runMetadataLabels contain an invalid key")
        if not isinstance(label_value, str):
            raise ValueError("runMetadataLabels contain an invalid value")
        encoded = label_value.encode("utf-8")
        if len(encoded) > MAX_RUN_METADATA_LABEL_VALUE_BYTES or "\0" in label_value:
            raise ValueError("runMetadataLabels contain an invalid value")
        result[key] = label_value
    return result


def _require_worker_result_text(value: str) -> str:
    if is_blank(value) or len(value.encode("utf-8")) > MAX_WORKER_RESULT_BYTES:
        raise ValueError("Worker result must contain 1..65536 UTF-8 bytes")
    return value


class WireModel(BaseModel):
    # Optional fields mean omission unless their owning wire schema allows null.
    nullable_fields: ClassVar[frozenset[str]] = frozenset()

    model_config = ConfigDict(
        alias_generator=_to_camel,
        populate_by_name=True,
        extra="forbid",
        strict=True,
    )

    @field_validator("*", mode="after")
    @classmethod
    def reject_optional_null(cls, value, info: ValidationInfo):
        name = info.field_name
        if (
            value is None
            and name is not None
            and not cls.model_fields[name].is_required()
            and name not in cls.nullable_fields
        ):
            raise ValueError(f"{name} must be omitted, not null")
        return value


def _require_runtime_adapter_ref(value: str) -> str:
    if value not in RUNTIME_ADAPTER_REFS:
        raise ValueError("unknown RuntimeAdapter ref")
    return value


class VersionedWireModel(WireModel):
    api_version: Literal[API_VERSION]


class TerminationError(WireModel):
    code: str
    message: str
    retryable: bool

    @model_validator(mode="after")
    def validate_error(self) -> Self:
        _require_text("code", self.code)
        _require_text("message", self.message)
        return self


RuntimeAdapterRef = Annotated[str, AfterValidator(_require_runtime_adapter_ref)]
