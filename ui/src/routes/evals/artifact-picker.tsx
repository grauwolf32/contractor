import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import {
  listArtifacts,
  downloadExactArtifact,
  type ArtifactMetadata,
} from "../../api/artifacts";
import { listProjectArtifacts } from "../../api/project-artifacts";
import { usePublicAPI } from "../../api/context";
import type { EvalArtifact } from "../../api/evals";
import { useSession } from "../../auth/session";
import { ProjectArtifactWriteForm } from "../projects/common";
import { EvalError, EvalField, EvalPages } from "./common";

export function EvalArtifactPicker({
  projectId,
  onSelect,
}: {
  projectId: string;
  onSelect: (ref: EvalArtifact) => void;
}) {
  const api = usePublicAPI();
  const { session } = useSession();
  const [scope, setScope] = useState<"project" | "user">("project");
  const [cursors, setCursors] = useState<string[]>([]);
  const [upload, setUpload] = useState(false);
  const cursor = cursors.at(-1);
  const inventory = useQuery({
    queryKey: ["evals", "input-artifacts", scope, projectId, cursor],
    queryFn: () =>
      scope === "project"
        ? listProjectArtifacts(api, {
            projectId,
            ...(cursor ? { cursor } : {}),
          })
        : listArtifacts(api, cursor ? { cursor } : {}),
  });
  const select = useMutation({
    mutationFn: async (metadata: ArtifactMetadata) => {
      const scopeId =
        scope === "project" ? projectId : session!.principal.userId;
      const prefix =
        scope === "project"
          ? `/v1/projects/${encodeURIComponent(projectId)}`
          : "/v1";
      const artifact = metadata.artifact;
      const path = `${prefix}/artifacts/${encodeURIComponent(artifact.namespace)}/${encodeURIComponent(artifact.name)}?revision=${encodeURIComponent(artifact.revision)}`;
      const { blob } = await downloadExactArtifact(api, metadata, path);
      const hash = await crypto.subtle.digest(
        "SHA-256",
        await blob.arrayBuffer(),
      );
      const sha256 =
        "sha256:" +
        Array.from(new Uint8Array(hash), (x) =>
          x.toString(16).padStart(2, "0"),
        ).join("");
      return {
        scope,
        scopeId,
        ...artifact,
        sha256,
        mediaType: metadata.mediaType,
        sizeBytes: blob.size,
      };
    },
    onSuccess: onSelect,
  });
  return (
    <div className="eval-artifact-picker">
      <EvalField label="Input source">
        <select
          value={scope}
          onChange={(e) => {
            setScope(e.target.value as typeof scope);
            setCursors([]);
            setUpload(false);
          }}
        >
          <option value="project">Evaluation workspace</option>
          <option value="user">My artifacts</option>
        </select>
      </EvalField>
      <EvalError error={inventory.error} />
      <EvalError error={select.error} />
      {select.isPending ? (
        <p role="status">Pinning exact input bytes…</p>
      ) : null}
      <ul className="eval-choice-list">
        {inventory.data?.items.map((item) => (
          <li key={item.artifact.revision}>
            <span>
              {item.artifact.namespace}/{item.artifact.name}
              <small>
                {item.mediaType} · revision {item.artifact.revision}
              </small>
            </span>
            <button
              type="button"
              className="secondary"
              disabled={select.isPending}
              onClick={() => select.mutate(item)}
            >
              Use input
            </button>
          </li>
        ))}
      </ul>
      <EvalPages
        previous={
          cursors.length ? () => setCursors((c) => c.slice(0, -1)) : undefined
        }
        next={
          inventory.data?.page.hasMore && inventory.data.page.nextCursor
            ? () => setCursors((c) => [...c, inventory.data!.page.nextCursor!])
            : undefined
        }
      />
      {scope === "project" ? (
        <>
          <button
            type="button"
            className="secondary"
            onClick={() => setUpload(!upload)}
          >
            Upload an input
          </button>
          {upload ? (
            <ProjectArtifactWriteForm
              projectId={projectId}
              onWritten={() => {
                setUpload(false);
                void inventory.refetch();
              }}
            />
          ) : null}
        </>
      ) : null}
    </div>
  );
}
