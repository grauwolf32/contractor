"""Allocation-local loading and script-free ADK exposure of Agent Skills."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import mimetypes
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from google.adk.skills import load_skill_from_dir, prompt
from google.adk.skills.models import Skill
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.skill_toolset import SkillToolset
from google.genai import types

from contractor_runtime.artifacts import (
    ArtifactAPIError,
    ArtifactClient,
    ArtifactClientError,
)
from contractor_runtime.contracts import ResolvedSkill, RuntimeSettings
from contractor_runtime.workspace import AllocationWorkspace

from .package import (
    MAXIMUM_PATH_BYTES,
    MAXIMUM_PATH_COMPONENTS,
    MEDIA_TYPE,
    SkillPackage,
    SkillPackageError,
    validate_package,
)

if TYPE_CHECKING:
    from google.adk.models.llm_request import LlmRequest
    from google.adk.tools.tool_context import ToolContext


EXACT_SKILL_TOOL_NAMES = ("list_skills", "load_skill", "load_skill_resource")
MAXIMUM_PACKAGES = 32
MAXIMUM_STORED_BYTES = 64 << 20
MAXIMUM_EXPANDED_BYTES = 128 << 20
MAXIMUM_DISCLOSURE_BYTES = 16 << 20
DISCLOSURE_ENVELOPE_BYTES = 64 << 10
EXTRACTION_DIRECTORY = ".agent-skills"

SKILL_SYSTEM_INSTRUCTION = (
    "Specialized Agent Skills are available through three progressive-disclosure "
    "functions. Use list_skills to see their names and descriptions. When one is "
    "relevant, call load_skill with its exact name to read its SKILL.md guidance, "
    "then continue the task in the same turn. Use load_skill_resource only for an "
    "exact references/... or assets/... path named by that guidance. These resources "
    "belong to the selected skill and are not user-provided Run artifacts. If a skill "
    "or resource call returns an error, do not guess alternate names or paths."
)

_FORBIDDEN_ADK_SURFACE = ("run_skill_script", "search_skills", "scripts/")
_PORTABLE_COMPONENT = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


class AgentSkillPreparationError(RuntimeError):
    """Secret- and content-free allocation preparation failure."""

    def __init__(self, code: str, *, retryable: bool, status_code: int) -> None:
        self.code = code
        self.retryable = retryable
        self.status_code = status_code
        super().__init__(f"Agent Skill preparation failed ({code})")


class AgentSkillCleanupError(RuntimeError):
    """Signals that allocation-local Skill state could not be erased."""


class SkillToolError(RuntimeError):
    """Stable error used only for the content-free metric projection."""

    def __init__(self, code: str) -> None:
        self.code = code
        self.retryable = False
        super().__init__(code)


@dataclass(slots=True)
class DisclosureBudget:
    """Atomic allocation-lifetime model-visible Skill content budget."""

    limit: int = MAXIMUM_DISCLOSURE_BYTES
    used: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def __post_init__(self) -> None:
        if self.limit <= 0 or self.used < 0 or self.used > self.limit:
            raise ValueError("invalid Agent Skill disclosure budget")

    async def reserve(self, charge: int) -> bool:
        if charge < 0:
            raise ValueError("Agent Skill disclosure charge must not be negative")
        async with self._lock:
            if charge > self.limit - self.used:
                return False
            self.used += charge
            return True


@dataclass(slots=True, repr=False)
class PreparedAgentSkills:
    """Owns all package, native ADK and extraction state for one allocation."""

    extraction_root: Path
    native_toolset: SkillToolset
    skills: list[Skill]
    native_tools: dict[str, BaseTool]
    charges: dict[tuple[str, str, str], int]
    binary_resources: dict[tuple[str, str], bytes]
    disclosure: DisclosureBudget = field(default_factory=DisclosureBudget)
    _adapter: ContractorSkillToolset | None = field(default=None, init=False, repr=False)
    _binary_authorizations: dict[tuple[str, str, str], int] = field(
        default_factory=dict, init=False, repr=False
    )
    _closed: bool = field(default=False, init=False, repr=False)
    _remove_tree: Callable[[Path], None] = field(default=shutil.rmtree, repr=False)

    def __repr__(self) -> str:
        return f"PreparedAgentSkills(package_count={len(self.skills)}, closed={self._closed})"

    @property
    def selected_names(self) -> frozenset[str]:
        return frozenset(skill.name for skill in self.skills)

    def build_adapter(
        self,
        *,
        budget: Callable[[], Any | None],
        metrics: Any,
    ) -> ContractorSkillToolset:
        if self._closed:
            raise RuntimeError("Agent Skill owner is closed")
        if self._adapter is not None:
            raise RuntimeError("Agent Skill adapter already exists")
        adapter = ContractorSkillToolset(self, budget=budget, metrics=metrics)
        self._adapter = adapter
        return adapter

    def charge_for(self, tool_name: str, skill_name: str = "", file_path: str = "") -> int:
        charge = self.charges.get((tool_name, skill_name, file_path))
        if charge is not None:
            return charge
        # A valid but absent resource produces only a bounded native error. The
        # argument itself is already bounded by the portable-path validator.
        return DISCLOSURE_ENVELOPE_BYTES + len(skill_name.encode()) + len(file_path.encode())

    def authorize_binary(self, invocation_id: str, skill_name: str, file_path: str) -> None:
        key = (invocation_id, skill_name, file_path)
        self._binary_authorizations[key] = self._binary_authorizations.get(key, 0) + 1

    def consume_binary(self, invocation_id: str, skill_name: str, file_path: str) -> bytes | None:
        key = (invocation_id, skill_name, file_path)
        count = self._binary_authorizations.get(key, 0)
        if count <= 0:
            return None
        if count == 1:
            del self._binary_authorizations[key]
        else:
            self._binary_authorizations[key] = count - 1
        return self.binary_resources.get((skill_name, file_path))

    async def close(self) -> None:
        if self._closed:
            return
        failures: list[Exception] = []
        try:
            await self.native_toolset.close()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            failures.append(error)

        if self._adapter is not None:
            self._adapter.detach()
            self._adapter = None
        self.native_tools.clear()
        self._binary_authorizations.clear()
        self.binary_resources.clear()
        self.charges.clear()
        for skill in self.skills:
            skill.instructions = ""
            skill.resources.references.clear()
            skill.resources.assets.clear()
            skill.resources.scripts.clear()
            skill._uri = None
        self.skills.clear()
        native_skills = getattr(self.native_toolset, "_skills", None)
        if isinstance(native_skills, dict):
            native_skills.clear()
        native_tools = getattr(self.native_toolset, "_tools", None)
        if isinstance(native_tools, list):
            native_tools.clear()

        if self.extraction_root.exists() or self.extraction_root.is_symlink():
            try:
                await asyncio.to_thread(self._remove_tree, self.extraction_root)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                failures.append(error)

        if failures:
            raise AgentSkillCleanupError(
                f"Agent Skill cleanup failed ({type(failures[0]).__name__})"
            ) from None
        self._closed = True


class ContractorSkillToolset(BaseToolset):
    """Safe ADK surface around the native three non-executable Skill tools."""

    def __init__(
        self,
        owner: PreparedAgentSkills,
        *,
        budget: Callable[[], Any | None],
        metrics: Any,
    ) -> None:
        super().__init__()
        self._owner: PreparedAgentSkills | None = owner
        self._tools = [
            ContractorSkillTool(
                name=name,
                native=owner.native_tools[name],
                owner=owner,
                budget=budget,
                metrics=metrics,
            )
            for name in EXACT_SKILL_TOOL_NAMES
        ]

    async def get_tools(self, readonly_context: Any | None = None) -> list[BaseTool]:
        del readonly_context
        return list(self._tools)

    async def process_llm_request(
        self, *, tool_context: ToolContext, llm_request: LlmRequest
    ) -> None:
        del tool_context
        llm_request.append_instructions([SKILL_SYSTEM_INSTRUCTION])

    async def close(self) -> None:
        owner = self._owner
        if owner is not None:
            await owner.close()

    def detach(self) -> None:
        for tool in self._tools:
            if isinstance(tool, ContractorSkillTool):
                tool.detach()
        self._tools.clear()
        self._owner = None


class ContractorSkillTool(BaseTool):
    def __init__(
        self,
        *,
        name: str,
        native: BaseTool,
        owner: PreparedAgentSkills,
        budget: Callable[[], Any | None],
        metrics: Any,
    ) -> None:
        descriptions = {
            "list_skills": "List the exact Agent Skills selected for this Worker.",
            "load_skill": "Load the SKILL.md guidance for one selected Agent Skill.",
            "load_skill_resource": (
                "Load one references/... or assets/... file from a selected Agent Skill."
            ),
        }
        super().__init__(name=name, description=descriptions[name])
        self._native: BaseTool | None = native
        self._owner: PreparedAgentSkills | None = owner
        self._budget = budget
        self._metrics = metrics

    def _get_declaration(self) -> types.FunctionDeclaration:
        properties: dict[str, Any] = {}
        required: list[str] = []
        if self.name in {"load_skill", "load_skill_resource"}:
            properties["skill_name"] = {
                "type": "string",
                "description": "Exact selected Agent Skill name.",
            }
            required.append("skill_name")
        if self.name == "load_skill_resource":
            properties["file_path"] = {
                "type": "string",
                "description": "Portable references/... or assets/... path.",
            }
            required.append("file_path")
        schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
        }
        if required:
            schema["required"] = required
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters_json_schema=schema,
        )

    async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> Any:
        started = time.monotonic()
        budget = self._budget()
        if budget is not None:
            budget.before_tool_call()

        owner = self._owner
        native = self._native
        safe_arguments: dict[str, Any] = {"arguments_valid": False}
        error: SkillToolError | None = None
        result: Any
        visible_bytes = 0
        if owner is None or native is None:
            error = SkillToolError("SKILL_TOOL_UNAVAILABLE")
            result = _tool_error("SKILL_TOOL_UNAVAILABLE", "Agent Skill tools are unavailable.")
        else:
            validated = _validate_tool_arguments(self.name, args, owner.selected_names)
            if validated is None:
                error = SkillToolError("INVALID_ARGUMENTS")
                result = _tool_error("INVALID_ARGUMENTS", "Skill function arguments are invalid.")
            else:
                skill_name, file_path, safe_arguments = validated
                charge = owner.charge_for(self.name, skill_name, file_path)
                if not await owner.disclosure.reserve(charge):
                    error = SkillToolError("SKILL_DISCLOSURE_LIMIT")
                    result = _tool_error(
                        "SKILL_DISCLOSURE_LIMIT",
                        "Agent Skill disclosure limit was reached.",
                    )
                else:
                    try:
                        result = await native.run_async(args=args, tool_context=tool_context)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        error = SkillToolError("SKILL_TOOL_ERROR")
                        result = _tool_error("SKILL_TOOL_ERROR", "Agent Skill function failed.")
                    native_code = _native_error_code(result)
                    if native_code is not None:
                        error = SkillToolError(native_code)
                    if (
                        error is None
                        and self.name == "load_skill_resource"
                        and isinstance(result, Mapping)
                        and isinstance(result.get("status"), str)
                    ):
                        owner.authorize_binary(
                            tool_context.invocation_id,
                            skill_name,
                            file_path,
                        )
                    binary = (
                        owner.binary_resources.get((skill_name, file_path))
                        if error is None
                        and self.name == "load_skill_resource"
                        and isinstance(result, Mapping)
                        and isinstance(result.get("status"), str)
                        else None
                    )
                    visible_bytes = _model_visible_size(
                        result,
                        binary,
                    )
                    if visible_bytes > charge:
                        error = SkillToolError("SKILL_DISCLOSURE_ESTIMATE_INVALID")
                        result = _tool_error(
                            "SKILL_DISCLOSURE_ESTIMATE_INVALID",
                            "Agent Skill result was suppressed by Runtime policy.",
                        )
                        visible_bytes = _json_bytes(result)

        if visible_bytes == 0:
            visible_bytes = _json_bytes(result)
        self._metrics.record_tool_call(
            self.name,
            arguments=safe_arguments,
            error=error,
            duration_ms=max(0, int((time.monotonic() - started) * 1000)),
            result_size_bytes=visible_bytes,
        )
        return result

    async def process_llm_request(
        self, *, tool_context: ToolContext, llm_request: LlmRequest
    ) -> None:
        await super().process_llm_request(tool_context=tool_context, llm_request=llm_request)
        if self.name != "load_skill_resource" or not llm_request.contents:
            return
        owner = self._owner
        if owner is None:
            return
        last = llm_request.contents[-1]
        for part in last.parts or []:
            response = getattr(part, "function_response", None)
            if response is None or response.name != self.name:
                continue
            value = response.response or {}
            if not isinstance(value, Mapping) or not isinstance(value.get("status"), str):
                continue
            skill_name = value.get("skill_name")
            file_path = value.get("file_path")
            if not isinstance(skill_name, str) or not isinstance(file_path, str):
                continue
            content = owner.consume_binary(tool_context.invocation_id, skill_name, file_path)
            if content is None:
                continue
            media_type, _ = mimetypes.guess_type(file_path)
            llm_request.contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text="The selected Agent Skill binary resource is:"),
                        types.Part(
                            inline_data=types.Blob(
                                data=content,
                                mime_type=media_type or "application/octet-stream",
                            )
                        ),
                    ],
                )
            )

    def detach(self) -> None:
        self._native = None
        self._owner = None
        self._metrics = None


async def prepare_agent_skills(
    resolved: Sequence[ResolvedSkill],
    *,
    allocation_id: str,
    runtime_settings: RuntimeSettings,
    workspace: AllocationWorkspace,
    artifact_client_factory: Callable[[str, RuntimeSettings], ArtifactClient] | None,
) -> PreparedAgentSkills | None:
    """Fetch, independently validate, extract and load one immutable Skill set."""

    if not resolved:
        return None
    if len(resolved) > MAXIMUM_PACKAGES or artifact_client_factory is None:
        raise AgentSkillPreparationError(
            "skill_runtime_unsupported", retryable=False, status_code=422
        )

    root = workspace.path / EXTRACTION_DIRECTORY
    if root.parent != workspace.path:
        raise AgentSkillPreparationError(
            "skill_runtime_unsupported", retryable=False, status_code=422
        )
    try:
        root.mkdir(mode=0o700)
    except OSError:
        raise AgentSkillPreparationError(
            "skill_runtime_unsupported", retryable=False, status_code=422
        ) from None

    native: SkillToolset | None = None
    loaded_skills: list[Skill] = []
    try:
        client = artifact_client_factory(allocation_id, runtime_settings)
        packages: list[tuple[ResolvedSkill, SkillPackage]] = []
        stored_total = 0
        expanded_total = 0
        for selected in resolved:
            try:
                value = await client.read_artifact(selected.artifact)
            except asyncio.CancelledError:
                raise
            except ArtifactClientError as error:
                retryable = not isinstance(error, ArtifactAPIError) or error.retryable
                raise AgentSkillPreparationError(
                    "skill_artifact_unavailable",
                    retryable=retryable,
                    status_code=503 if retryable else 422,
                ) from None
            except Exception:
                raise AgentSkillPreparationError(
                    "skill_artifact_unavailable", retryable=True, status_code=503
                ) from None

            if value.artifact != selected.artifact:
                raise AgentSkillPreparationError(
                    "skill_artifact_unavailable", retryable=True, status_code=503
                )
            if value.media_type != MEDIA_TYPE:
                raise AgentSkillPreparationError(
                    "skill_media_type_invalid", retryable=False, status_code=422
                )
            digest = f"sha256:{hashlib.sha256(value.data).hexdigest()}"
            if digest != selected.package_digest:
                raise AgentSkillPreparationError(
                    "skill_digest_mismatch", retryable=False, status_code=422
                )
            try:
                package = validate_package(value.data, selected.name)
            except SkillPackageError as error:
                raise AgentSkillPreparationError(
                    error.code, retryable=False, status_code=422
                ) from None
            stored_total += package.stored_bytes
            expanded_total += package.expanded_bytes
            if stored_total > MAXIMUM_STORED_BYTES or expanded_total > MAXIMUM_EXPANDED_BYTES:
                raise AgentSkillPreparationError(
                    "skill_limit_exceeded", retryable=False, status_code=422
                )
            packages.append((selected, package))

        charges: dict[tuple[str, str, str], int] = {}
        binary_resources: dict[tuple[str, str], bytes] = {}
        for selected, package in packages:
            skill_directory = _extract_package(root, selected.name, package)
            try:
                skill = load_skill_from_dir(skill_directory)
            except Exception:
                raise AgentSkillPreparationError(
                    "skill_runtime_unsupported", retryable=False, status_code=422
                ) from None
            if (
                skill.name != selected.name
                or skill.resources.scripts
                or set(skill.resources.references)
                != {
                    resource.path.removeprefix("references/")
                    for resource in package.resources
                    if resource.path.startswith("references/")
                }
                or set(skill.resources.assets)
                != {
                    resource.path.removeprefix("assets/")
                    for resource in package.resources
                    if resource.path.startswith("assets/")
                }
            ):
                raise AgentSkillPreparationError(
                    "skill_runtime_unsupported", retryable=False, status_code=422
                )
            skill._uri = None
            loaded_skills.append(skill)

        native = SkillToolset(
            skills=loaded_skills,
            registry=None,
            tool_filter=list(EXACT_SKILL_TOOL_NAMES),
        )
        tools = await native.get_tools()
        native_tools = {tool.name: tool for tool in tools}
        if tuple(tool.name for tool in tools) != EXACT_SKILL_TOOL_NAMES:
            raise AgentSkillPreparationError(
                "skill_runtime_unsupported", retryable=False, status_code=422
            )
        _build_disclosure_plan(loaded_skills, charges, binary_resources)
        return PreparedAgentSkills(
            extraction_root=root,
            native_toolset=native,
            skills=loaded_skills,
            native_tools=native_tools,
            charges=charges,
            binary_resources=binary_resources,
        )
    except asyncio.CancelledError:
        await _cleanup_failed_preparation(root, native, loaded_skills)
        raise
    except AgentSkillPreparationError:
        await _cleanup_failed_preparation(root, native, loaded_skills)
        raise
    except Exception:
        await _cleanup_failed_preparation(root, native, loaded_skills)
        raise AgentSkillPreparationError(
            "skill_runtime_unsupported", retryable=False, status_code=422
        ) from None


async def probe_native_agent_skills() -> bool:
    """Prove the pinned ADK loader and exact script-free adapter assumptions."""

    if any(value in SKILL_SYSTEM_INSTRUCTION for value in _FORBIDDEN_ADK_SURFACE):
        return False
    with tempfile.TemporaryDirectory(prefix="contractor-skill-probe-") as temporary:
        directory = Path(temporary) / "probe-skill"
        directory.mkdir(mode=0o700)
        (directory / "SKILL.md").write_text(
            "---\nname: probe-skill\ndescription: Probe native Agent Skills support.\n---\n"
            "Follow the supplied task.\n",
            encoding="utf-8",
        )
        skill = load_skill_from_dir(directory)
        skill._uri = None
        native = SkillToolset(
            skills=[skill],
            registry=None,
            tool_filter=list(EXACT_SKILL_TOOL_NAMES),
        )
        try:
            tools = await native.get_tools()
            if tuple(tool.name for tool in tools) != EXACT_SKILL_TOOL_NAMES:
                return False
            if (
                getattr(native, "_registry", object()) is not None
                or getattr(native, "_code_executor", object()) is not None
                or getattr(native, "_env", object()) is not None
                or getattr(native, "_skills_folder", object()) is not None
                or getattr(native, "_provided_tools_by_name", object()) != {}
                or getattr(native, "_provided_toolsets", object()) != []
            ):
                return False
        finally:
            await native.close()
    return True


def _build_disclosure_plan(
    skills: Sequence[Skill],
    charges: dict[tuple[str, str, str], int],
    binary_resources: dict[tuple[str, str], bytes],
) -> None:
    listed = prompt.format_skills_as_xml(list(skills))
    charges[("list_skills", "", "")] = _charge(listed)
    for skill in skills:
        loaded = {
            "skill_name": skill.name,
            "instructions": skill.instructions,
            "frontmatter": skill.frontmatter.model_dump(),
        }
        charges[("load_skill", skill.name, "")] = _charge(loaded)
        for prefix, resources in (
            ("references", skill.resources.references),
            ("assets", skill.resources.assets),
        ):
            for relative_path, content in resources.items():
                file_path = f"{prefix}/{relative_path}"
                if isinstance(content, bytes):
                    binary_resources[(skill.name, file_path)] = content
                    result: Any = {
                        "skill_name": skill.name,
                        "file_path": file_path,
                        "status": "Binary file detected. Content will be attached to the request.",
                    }
                    charges[("load_skill_resource", skill.name, file_path)] = _charge(
                        result, binary=content
                    )
                else:
                    result = {
                        "skill_name": skill.name,
                        "file_path": file_path,
                        "content": content,
                    }
                    charges[("load_skill_resource", skill.name, file_path)] = _charge(result)


def _charge(value: Any, *, binary: bytes | None = None) -> int:
    result = DISCLOSURE_ENVELOPE_BYTES + _json_bytes(value)
    if binary is not None:
        result += len(base64.b64encode(binary)) + DISCLOSURE_ENVELOPE_BYTES
    return result


def _model_visible_size(value: Any, binary: bytes | None) -> int:
    result = _json_bytes(value)
    if binary is not None:
        result += len(base64.b64encode(binary)) + DISCLOSURE_ENVELOPE_BYTES
    return result


def _json_bytes(value: Any) -> int:
    return len(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    )


def _validate_tool_arguments(
    tool_name: str,
    args: Mapping[str, Any],
    selected_names: frozenset[str],
) -> tuple[str, str, dict[str, Any]] | None:
    if tool_name == "list_skills":
        return ("", "", {}) if not args else None
    if set(args) not in ({"skill_name"}, {"skill_name", "file_path"}):
        return None
    skill_name = args.get("skill_name")
    if not isinstance(skill_name, str) or skill_name not in selected_names:
        return None
    if tool_name == "load_skill":
        if set(args) != {"skill_name"}:
            return None
        return skill_name, "", {"skill_name": skill_name}
    if tool_name != "load_skill_resource" or set(args) != {"skill_name", "file_path"}:
        return None
    file_path = args.get("file_path")
    if not isinstance(file_path, str) or not _valid_resource_path(file_path):
        return None
    return skill_name, file_path, {"skill_name": skill_name, "file_path": file_path}


def _valid_resource_path(value: str) -> bool:
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError:
        return False
    if not value or len(encoded) > MAXIMUM_PATH_BYTES or "\\" in value or value.startswith("/"):
        return False
    components = value.split("/")
    return (
        2 <= len(components) <= MAXIMUM_PATH_COMPONENTS
        and components[0] in {"references", "assets"}
        and all(
            component not in {"", ".", ".."} and _PORTABLE_COMPONENT.fullmatch(component)
            for component in components
        )
    )


def _native_error_code(result: Any) -> str | None:
    if not isinstance(result, Mapping) or "error" not in result:
        return None
    code = result.get("error_code")
    if isinstance(code, str) and re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", code):
        return code
    return "SKILL_TOOL_ERROR"


def _tool_error(code: str, message: str) -> dict[str, str]:
    return {"error": message, "error_code": code}


def _extract_package(root: Path, skill_name: str, package: SkillPackage) -> Path:
    directory = root / skill_name
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.mkdir(skill_name, mode=0o700, dir_fd=root_fd)
        skill_fd = os.open(
            skill_name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=root_fd,
        )
        try:
            for member in package.members:
                _write_exclusive_member(skill_fd, member.path, member.data())
        finally:
            os.close(skill_fd)
    finally:
        os.close(root_fd)
    return directory


def _write_exclusive_member(root_fd: int, path: str, data: bytes) -> None:
    components = path.split("/")
    directory_fd = os.dup(root_fd)
    try:
        for component in components[:-1]:
            with contextlib.suppress(FileExistsError):
                os.mkdir(component, mode=0o700, dir_fd=directory_fd)
            next_fd = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        file_fd = os.open(
            components[-1],
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        try:
            view = memoryview(data)
            while view:
                written = os.write(file_fd, view)
                if written <= 0:
                    raise OSError("short Agent Skill extraction write")
                view = view[written:]
        finally:
            os.close(file_fd)
    finally:
        os.close(directory_fd)


async def _cleanup_failed_preparation(
    root: Path,
    native: SkillToolset | None,
    skills: Sequence[Skill],
) -> None:
    if native is not None:
        with contextlib.suppress(Exception):
            await native.close()
    for skill in skills:
        skill.instructions = ""
        skill.resources.references.clear()
        skill.resources.assets.clear()
        skill.resources.scripts.clear()
        skill._uri = None
    if root.exists() or root.is_symlink():
        with contextlib.suppress(Exception):
            await asyncio.to_thread(shutil.rmtree, root)
