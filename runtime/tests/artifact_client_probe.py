"""Cross-language probe used by the Go private Artifact API integration test."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from contractor_runtime.allocation import WorkerState
from contractor_runtime.artifacts import ArtifactAPIError, ArtifactClient, MTLSArtifactTransport
from contractor_runtime.contracts import ArtifactRef, RuntimeSettings
from contractor_runtime.mtls import runtime_agent_client_context
from contractor_runtime.toolsets.run_artifacts.tools import RunArtifactsToolsetFactory
from contractor_runtime.workspace import AllocationWorkspace


async def run(args: argparse.Namespace) -> dict[str, object]:
    context = runtime_agent_client_context(
        ca_file=args.ca,
        certificate_file=args.certificate,
        private_key_file=args.private_key,
    )
    client = ArtifactClient(
        args.allocation_id,
        MTLSArtifactTransport(
            args.api_url, context, timeout_seconds=3, runtime_instance_id=args.instance_id
        ),
    )
    if args.mode == "page":
        return await page_through(client, args.limit)
    if args.mode == "fenced":
        try:
            await client.write_artifact(
                ArtifactRef(namespace="inputs", name="source"),
                data=b"must not commit",
                media_type="text/plain",
                expected_revision=args.expected_revision,
            )
        except ArtifactAPIError as error:
            return {"code": error.code, "retryable": error.retryable}
        raise RuntimeError("fenced write unexpectedly succeeded")

    created = await client.write_artifact(
        ArtifactRef(namespace="inputs", name="source"),
        data=b"first payload",
        media_type="text/plain",
        expected_revision=None,
    )
    current = await client.read_artifact(ArtifactRef(namespace="inputs", name="source"))
    exact = await client.read_artifact(created.artifact)
    listed = (await client.list_artifact_page("inputs", limit=10)).artifacts
    updated = await client.write_artifact(
        ArtifactRef(namespace="inputs", name="source"),
        data=b"second payload",
        media_type="text/plain",
        expected_revision=created.artifact.revision,
    )
    try:
        await client.write_artifact(
            ArtifactRef(namespace="outputs", name="result"),
            data=b"forbidden",
            media_type="text/plain",
            expected_revision=None,
        )
    except ArtifactAPIError as error:
        output_error = error.code
    else:
        raise RuntimeError("reserved output write unexpectedly succeeded")
    return {
        "createdRevision": created.artifact.revision,
        "createdBindingAt": created.binding_created_at.isoformat(),
        "createdRevisionAt": created.revision_created_at.isoformat(),
        "updatedRevision": updated.artifact.revision,
        "updatedBindingAt": updated.binding_created_at.isoformat(),
        "updatedRevisionAt": updated.revision_created_at.isoformat(),
        "currentRevision": current.artifact.revision,
        "currentBindingAt": current.binding_created_at.isoformat(),
        "currentRevisionAt": current.revision_created_at.isoformat(),
        "exactRevision": exact.artifact.revision,
        "exactBindingAt": exact.binding_created_at.isoformat(),
        "exactRevisionAt": exact.revision_created_at.isoformat(),
        "payload": current.data.decode(),
        "listed": [ref.model_dump(by_alias=True, exclude_none=True) for ref in listed],
        "outputError": output_error,
    }


async def page_through(client: ArtifactClient, limit: int) -> dict[str, object]:
    """Follow every page through the client and through the list_artifacts tool."""

    client_refs: list[list[str]] = []
    client_pages = 0
    cursor: str | None = None
    while True:
        page = await client.list_artifact_page(limit=limit, cursor=cursor)
        client_pages += 1
        client_refs += [[ref.namespace, ref.name] for ref in page.artifacts]
        cursor = page.next_cursor
        if cursor is None:
            break
    state = WorkerState()
    tools = await RunArtifactsToolsetFactory(lambda _allocation, _settings: client).create_selected(
        selected=["list_artifacts"],
        allocation_id="allocation-1",
        run_id="run-a",
        namespace="builder",
        runtime_settings=RuntimeSettings(
            llmGatewayUrl="https://llm.invalid/v1",
            artifactApiUrl="https://control.invalid/private/v1",
            requestTimeoutSeconds=30,
        ),
        workspace=AllocationWorkspace(
            root=Path("/nonexistent/contractor-probe"),
            path=Path("/nonexistent/contractor-probe/allocation-1"),
        ),
        state=state,
    )
    tool_refs: list[list[str]] = []
    tool_pages = 0
    tool_cursor: str | None = None
    while True:
        result = await tools["list_artifacts"](cursor=tool_cursor)
        tool_pages += 1
        tool_refs += [[ref["namespace"], ref["name"]] for ref in result["artifacts"]]
        tool_cursor = result["nextCursor"]
        if tool_cursor is None:
            break
    return {
        "client": client_refs,
        "clientPages": client_pages,
        "tool": tool_refs,
        "toolPages": tool_pages,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--allocation-id", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--ca", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--mode", choices=("normal", "fenced", "page"), default="normal")
    parser.add_argument("--expected-revision")
    parser.add_argument("--limit", type=int, default=256)
    print(json.dumps(asyncio.run(run(parser.parse_args())), separators=(",", ":")))


if __name__ == "__main__":
    main()
