"""Private Runtime protocol base models and validation."""

from __future__ import annotations

import ipaddress
import json
import re
import unicodedata
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Self
from urllib.parse import urlsplit

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    model_validator,
)

API_VERSION = "contractor/v1alpha1"
ARTIFACT_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
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
MAX_UINT64 = 2**64 - 1
_STATE_METRIC_NAME_PATTERN = re.compile(r"^[a-z0-9_]+(?:\.[a-z0-9_]+)*$")


def _to_camel(value: str) -> str:
    first, *rest = value.split("_")
    return first + "".join(part.capitalize() for part in rest)


def _require_text(field: str, value: str) -> str:
    if not value.strip():
        raise ValueError(f"{field} must not be empty")
    return value


def _require_url(field: str, value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username is not None:
        raise ValueError(f"{field} must be an absolute HTTP(S) URL without user information")
    return value


def _require_inference_gateway_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        value != value.strip()
        or parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or "?" in value
        or "#" in value
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith("/")
    ):
        raise ValueError(
            "llmGatewayConfig.url must be an absolute HTTP(S) URL with an explicit "
            "path and no userinfo, query, or fragment"
        )
    return value


def _require_management_gateway_origin(value: str) -> str:
    parsed = urlsplit(value)
    if (
        value != value.strip()
        or parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or "?" in value
        or "#" in value
        or parsed.query
        or parsed.fragment
        or parsed.path
    ):
        raise ValueError("managementUrl must be a canonical HTTP(S) origin")
    if parsed.scheme == "http":
        try:
            loopback = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            loopback = False
        if not loopback:
            raise ValueError("HTTP managementUrl is allowed only for a loopback IP origin")
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


def _known_completion(value):
    # Optional future diagnostics must not make historical reports unreadable.
    if isinstance(value, dict) and len(json.dumps(value).encode()) > 4096:
        raise ValueError("completion diagnostics exceed their bound")
    if (
        isinstance(value, dict)
        and isinstance(value.get("kind"), str)
        and isinstance(value.get("phase"), str)
        and value["kind"] != ""
        and value["phase"] != ""
        and (
            value.get("kind") != "audit-check-results@1"
            or value.get("phase")
            not in {"collecting", "sealed", "publishing", "published", "failed"}
        )
    ):
        return None
    return value


def _require_state_workspace_path(value: str) -> str:
    if (
        not value
        or value != unicodedata.normalize("NFC", value)
        or len(value.encode("utf-8")) > 4096
        or value.startswith("/")
        or "\\" in value
        or "\x00" in value
        or "://" in value
    ):
        raise ValueError("Worker State workspace path is invalid")
    parts = value.split("/")
    if len(parts) > 128:
        raise ValueError("Worker State workspace path is invalid")
    for part in parts:
        if part in {"", ".", ".."} or any(
            ord(character) < 0x20 or ord(character) == 0x7F for character in part
        ):
            raise ValueError("Worker State workspace path is invalid")
    if (
        len(parts[0]) >= 2
        and parts[0][0].isascii()
        and parts[0][0].isalpha()
        and parts[0][1] == ":"
    ):
        raise ValueError("Worker State workspace path is invalid")
    return value


def _encoded_state_path_list_size(paths: list[str]) -> int:
    return len(json.dumps(paths, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


RUNTIME_ADAPTER_REFS = frozenset({"caido-graphql@1", "http-proxy@1", "otlp-http@1"})
RUNTIME_CREDENTIAL_KINDS = frozenset(
    {"caido-bearer@1", "http-proxy-basic@1", "http-proxy-bearer@1", "otlp-headers@1"}
)
PROXY_TARGETS = frozenset({"llm-gateway", "tool-http", "tool-subprocess"})
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
_CERTIFICATE_PATTERN = re.compile(
    r"-----BEGIN CERTIFICATE-----\s+.+?\s+-----END CERTIFICATE-----", re.DOTALL
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
        value != value.strip()
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
    if value == "":
        return value
    if (
        value != unicodedata.normalize("NFC", value)
        or len(value.encode("utf-8")) > 1024
        or value.startswith("/")
        or "\\" in value
        or "\x00" in value
        or "://" in value
    ):
        raise ValueError("workspace target is invalid")
    parts = value.split("/")
    if len(parts) > 32:
        raise ValueError("workspace target is invalid")
    for part in parts:
        if part in {"", ".", ".."} or any(
            ord(character) < 0x20 or ord(character) == 0x7F for character in part
        ):
            raise ValueError("workspace target is invalid")
    if len(parts[0]) >= 2 and parts[0][0].isalpha() and parts[0][1] == ":":
        raise ValueError("workspace target is invalid")
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
    if not value.strip() or len(value.encode("utf-8")) > MAX_WORKER_RESULT_BYTES:
        raise ValueError("Worker result must contain 1..65536 UTF-8 bytes")
    return value


class WireModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel,
        populate_by_name=True,
        extra="forbid",
        strict=True,
    )


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
