"""Typed local producer output; persisted identities belong to the Audit owner."""

from types import MappingProxyType

import jcs
from jsonschema.exceptions import ValidationError

from contractor_runtime import threads
from contractor_runtime.artifacts import ArtifactAPIError
from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.contracts.audit_documents import decode_audit_document
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.redaction import runtime_secrets
from contractor_runtime.strict_json import strict_json_loads
from contractor_runtime.toolsets.common.artifact_visibility import require_model_visible_binding
from contractor_runtime.toolsets.common.factory import require_selected_tools
from contractor_runtime.toolsets.common.tool_base import ArtifactTool


class SubmitHypothesisProposalsTool(ArtifactTool):
    name = "submit_hypothesis_proposals"
    description = """Save proposed hypotheses and optional checks as an ordinary Run output.

    Read the configured proposal context and relevant exact materials first.
    hypotheses_json maps local names to statement/scope/grounds, with optional
    sources listing material handles. work_json lists key/workflow/objective,
    optional hypothesis alias and inputs mapping checking slots to handles.
    selected_materials_json optionally maps new handles to exact own-Run refs,
    such as snapshots selected with the Project Findings reader. Scope and
    persisted IDs/revisions/origin are assigned by the Server. Repeated calls
    keep prior selected materials and require expected_revision for correction.
    Empty hypotheses and work express a successful decision to stop.
    Returns saved proposal metadata; hypotheses are published after Run success.
    """

    def __init__(self, client, metrics, secrets, namespace, run_id):
        super().__init__(client, metrics, secrets)
        self._namespace = namespace
        self._run_id = run_id

    async def __call__(
        self,
        hypotheses_json: str,
        work_json: str,
        expected_revision: str | None = None,
        selected_materials_json: str = "{}",
    ) -> dict:
        with self._recorded({"expected_revision": expected_revision}) as call:
            roles_value = await self._client.read_artifact(
                ArtifactRef(namespace="audit-context", name="hypothesis-proposals"),
                max_bytes=8192,
            )
            try:
                roles = decode_audit_document(
                    roles_value.data, schema="contractor.audit.proposal-binding.v1"
                )
                target = ArtifactRef.model_validate(roles["result"])
                if (
                    roles["run_id"] != self._run_id
                    or target.namespace != self._namespace
                    or target.revision is not None
                    or roles["context"]["namespace"] != "inputs"
                ):
                    raise ValueError("proposal role")
            except (ValueError, TypeError, KeyError, ValidationError):
                raise ToolInputError("Invalid configured proposal role.") from None
            context = await self._client.read_artifact(
                ArtifactRef.model_validate(roles["context"]), max_bytes=8 << 20
            )
            prior = None
            try:
                prior = await self._client.read_artifact(target, max_bytes=1 << 20)
            except ArtifactAPIError as error:
                if error.status_code != 404:
                    raise
            try:
                if len(selected_materials_json.encode()) > 1 << 20:
                    raise ValueError("selection bytes")
                selected = strict_json_loads(selected_materials_json)
                if not isinstance(selected, dict) or len(selected) > 32:
                    raise ValueError("selection shape")
                if prior is not None:
                    saved = strict_json_loads(prior.data.decode())
                    previous = saved.get("selected_materials", {})
                    if any(k in selected and selected[k] != v for k, v in previous.items()):
                        raise ValueError("selection handle changed")
                    selected = previous | selected
                if len(selected) > 32:
                    raise ValueError("selection bound")
                trusted = await threads.to_thread_until_done(
                    lambda: decode_audit_document(
                        context.data, schema="contractor.audit.proposal-context.v1"
                    ),
                    name="audit-proposal-context",
                )
                known_inputs = {jcs.canonicalize(m["ref"]) for m in trusted["materials"].values()}
                metadata = {}
                for handle, value in selected.items():
                    ref = ArtifactRef.model_validate(value)
                    ref.require_exact()
                    require_model_visible_binding(ref.namespace, ref.name)
                    if (
                        ref.namespace == "audit-context"
                        or (ref.namespace == "inputs" and ref.name == roles["context"]["name"])
                        or (ref.namespace == target.namespace and ref.name == target.name)
                    ):
                        raise ValueError("system or cyclic selection")
                    if ref.namespace == "inputs" and jcs.canonicalize(value) not in known_inputs:
                        raise ValueError("unselected system input")
                    exact = await self._client.read_artifact_metadata(ref)
                    metadata[handle] = {
                        "scope_kind": "run",
                        "scope_id": self._run_id,
                        "ref": exact.artifact.model_dump(by_alias=True, exclude_none=True),
                        # The private API deliberately hides digest; its fixed width
                        # suffices for the early byte bound. Server derives identity.
                        "digest": "sha256:" + "0" * 64,
                        "media_type": exact.media_type,
                        "size_bytes": exact.size,
                    }
                data = await threads.to_thread_until_done(
                    lambda: assemble_proposals(
                        hypotheses_json,
                        work_json,
                        context.data,
                        selected_materials_json=jcs.canonicalize(selected).decode(),
                        selected_metadata=metadata,
                    ),
                    name="audit-proposal-assembly",
                )
            except (ValueError, TypeError, KeyError, UnicodeError, ValidationError):
                raise ToolInputError("Invalid proposal content or selected binding.") from None
            if prior is not None and prior.data == data:
                result = {
                    "artifact": prior.artifact.model_dump(by_alias=True),
                    "mediaType": prior.media_type,
                    "size": len(data),
                }
                call.succeed(result)
                return result
            saved = await self._client.write_artifact(
                target,
                data=data,
                media_type="application/json",
                expected_revision=expected_revision,
            )
            result = saved.model_dump(by_alias=True)
            call.succeed({"artifact": result["artifact"], "size": saved.size})
            return result


def assemble_proposals(
    hypotheses_json: str,
    work_json: str,
    context: bytes,
    *,
    selected_materials_json: str = "{}",
    selected_metadata: dict | None = None,
) -> bytes:
    if len(hypotheses_json.encode()) + len(work_json.encode()) > 1 << 20:
        raise ValueError("proposal bytes")
    trusted = decode_audit_document(context, schema="contractor.audit.proposal-context.v1")
    seeds, work = strict_json_loads(hypotheses_json), strict_json_loads(work_json)
    selected = strict_json_loads(selected_materials_json)
    value = {"schema": "contractor.audit.proposals.v1", "hypotheses": seeds, "work": work}
    if selected:
        value["selected_materials"] = selected
    data = jcs.canonicalize(value)
    proposed = decode_audit_document(data, schema="contractor.audit.proposals.v1")
    materials = dict(trusted["materials"])
    original_by_local = {
        jcs.canonicalize(m["ref"]): trusted["input_sources"][key] for key, m in materials.items()
    }
    distinct = {jcs.canonicalize(m) for m in trusted["input_sources"].values()}
    for key, ref in proposed.get("selected_materials", {}).items():
        if key in materials or selected_metadata is None or key not in selected_metadata:
            raise ValueError("unknown or colliding selected material")
        material = selected_metadata[key]
        original = original_by_local.get(jcs.canonicalize(ref), material)
        distinct.add(jcs.canonicalize(original))
        materials[key] = material
    if len(distinct) > 32 or sum(map(len, distinct)) > 1 << 20:
        raise ValueError("cumulative material limit")
    for seed in seeds.values():
        if any(handle not in materials for handle in seed.get("sources", [])):
            raise ValueError("unknown grounds")
    program = trusted["program"]
    if (
        len(data) > 1 << 20
        or len(seeds) + len(work) > 128
        or set(seeds) & set(program["hypotheses"])
        or any(len(jcs.canonicalize(seed)) > 32768 for seed in seeds.values())
    ):
        raise ValueError("proposal bounds or alias collision")
    aliases = set(seeds) | set(program["hypotheses"])
    keys = set()
    for item in proposed["work"]:
        if item["key"] in keys:
            raise ValueError("duplicate work key")
        keys.add(item["key"])
        binding = program["workflows"][item["workflow"]]
        if "proposals" in binding["outputs"]:
            raise ValueError("producer response cannot call another producer")
        if "hypothesis" in item and (
            item["hypothesis"] not in aliases
            or "result" not in binding["outputs"]
            or not any(source["source"] == "task-context" for source in binding["inputs"].values())
        ):
            raise ValueError("checking binding")
        for slot, handle in item.get("inputs", {}).items():
            if binding["inputs"][slot]["source"] not in {"audit-input", "task-input"}:
                raise ValueError("system input")
            if handle not in materials:
                raise ValueError("foreign material")
            mapping = binding["inputs"][slot]
            if mapping["source"] == "audit-input":
                media = materials[handle]["media_type"]
                allowed = program["inputs"][mapping["name"]]["media_types"]
                if (
                    media not in allowed
                    and "*/*" not in allowed
                    and media.split("/")[0] + "/*" not in allowed
                ):
                    raise ValueError("incompatible material")
    return data


class AuditProposalsToolsetFactory:
    ref = "audit-proposals@1"
    exported_tools = frozenset({"submit_hypothesis_proposals"})
    infrastructure_channels = MappingProxyType({})

    def __init__(self, client_factory):
        self._client_factory = client_factory

    async def probe(self):
        return self.exported_tools if self._client_factory is not None else frozenset()

    async def create_selected(
        self, *, selected, allocation_id, namespace, runtime_settings, resources, **kwargs
    ):
        require_selected_tools(selected, self.exported_tools)
        return {
            "submit_hypothesis_proposals": SubmitHypothesisProposalsTool(
                self._client_factory(allocation_id, runtime_settings),
                resources.metrics,
                runtime_secrets(runtime_settings),
                namespace,
                kwargs["run_id"],
            )
        }
