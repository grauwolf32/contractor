"""Cross-language probe used by the Go private Artifact API integration test."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from contractor_runtime.artifacts import ArtifactAPIError, ArtifactClient, MTLSArtifactTransport
from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.mtls import runtime_agent_client_context


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
    listed = await client.list_artifacts("inputs")
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
        "updatedRevision": updated.artifact.revision,
        "currentRevision": current.artifact.revision,
        "exactRevision": exact.artifact.revision,
        "payload": current.data.decode(),
        "listed": [ref.model_dump(by_alias=True, exclude_none=True) for ref in listed],
        "outputError": output_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--allocation-id", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--ca", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--mode", choices=("normal", "fenced"), default="normal")
    parser.add_argument("--expected-revision")
    print(json.dumps(asyncio.run(run(parser.parse_args())), separators=(",", ":")))


if __name__ == "__main__":
    main()
