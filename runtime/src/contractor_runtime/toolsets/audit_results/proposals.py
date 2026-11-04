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
from contractor_runtime.toolsets.common.factory import require_selected_tools
from contractor_runtime.toolsets.common.tool_base import ArtifactTool


class SubmitHypothesisProposalsTool(ArtifactTool):
    name = "submit_hypothesis_proposals"
    description = """Record named hypotheses and next work from the selected generation context.

    Read inputs/generation_context and its explicitly selected material inputs first.
    hypotheses_json is an object of local names to statement/scope/grounds objects.
    work_json is an array of key/workflow/objective objects, with optional hypothesis
    local name and inputs mapping checking slots to selected material handles.
    Choose allowed Workflow aliases only. Never supply persisted IDs or provenance.
    Empty hypotheses and work are a successful decision to propose no more work.
    Returns the saved proposal output metadata. Use expected_revision to correct it.
    """

    def __init__(self, client, metrics, secrets, namespace):
        super().__init__(client, metrics, secrets)
        self._namespace = namespace

    async def __call__(
        self, hypotheses_json: str, work_json: str, expected_revision: str | None = None
    ) -> dict:
        with self._recorded({"expected_revision": expected_revision}) as call:
            context = await self._client.read_artifact(
                ArtifactRef(namespace="inputs", name="generation_context"), max_bytes=8 << 20
            )
            try:
                data = await threads.to_thread_until_done(
                    lambda: assemble_proposals(hypotheses_json, work_json, context.data),
                    name="audit-proposal-assembly",
                )
            except (ValueError, TypeError, KeyError, UnicodeError, ValidationError):
                raise ToolInputError("Invalid proposal content or selected binding.") from None
            target = ArtifactRef(namespace=self._namespace, name="proposals")
            try:
                prior = await self._client.read_artifact(target, max_bytes=1 << 20)
            except ArtifactAPIError as error:
                if error.status_code != 404:
                    raise
            else:
                if prior.data == data:
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


def assemble_proposals(hypotheses_json: str, work_json: str, context: bytes) -> bytes:
    if len(hypotheses_json.encode()) + len(work_json.encode()) > 1 << 20:
        raise ValueError("proposal bytes")
    trusted = decode_audit_document(context, schema="contractor.audit.proposal-context.v1")
    seeds, work = strict_json_loads(hypotheses_json), strict_json_loads(work_json)
    data = jcs.canonicalize(
        {"schema": "contractor.audit.proposals.v1", "hypotheses": seeds, "work": work}
    )
    proposed = decode_audit_document(data, schema="contractor.audit.proposals.v1")
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
            if handle not in trusted["materials"]:
                raise ValueError("foreign material")
            mapping = binding["inputs"][slot]
            if mapping["source"] == "audit-input":
                media = trusted["materials"][handle]["media_type"]
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
            )
        }
