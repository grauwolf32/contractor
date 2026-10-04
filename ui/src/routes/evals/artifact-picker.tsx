import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { listArtifacts, type ArtifactMetadata } from "../../api/artifacts";
import { listProjectArtifacts } from "../../api/project-artifacts";
import { usePublicAPI } from "../../api/context";
import { scopedArtifactAPI } from "../../api/scoped-artifacts";
import type { EvalArtifact } from "../../api/evals";
import { CursorControls } from "../../app/cursor-controls";
import { useCursorStack } from "../../app/pagination";
import { useSession } from "../../auth/session";
import { ArtifactWriteForm } from "../artifacts/common";
import { EvalError, EvalField } from "./common";
import { queryKeys } from "../../api/query-keys";
import { sha256Hex } from "../../app/digest";

/** Skill packages live on the Skills surface, not among Eval inputs. */
const EXCLUDED_SKILL_NAMESPACE = "skills";

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
  const pages = useCursorStack();
  const [upload, setUpload] = useState(false);
  const cursor = pages.cursor;
  const inventory = useQuery({
    queryKey: queryKeys.evals.inputArtifacts(scope, projectId, cursor),
    queryFn: () =>
      scope === "project"
        ? listProjectArtifacts(api, {
            projectId,
            ...(cursor ? { cursor } : {}),
          })
        : listArtifacts(api, {
            excludeNamespace: EXCLUDED_SKILL_NAMESPACE,
            ...(cursor ? { cursor } : {}),
          }),
  });
  const select = useMutation({
    mutationFn: async (metadata: ArtifactMetadata) => {
      const scopeId =
        scope === "project" ? projectId : session!.principal.userId;
      const { blob } = await scopedArtifactAPI(
        api,
        scope === "project"
          ? { kind: "project", id: projectId }
          : { kind: "user" },
      ).download(metadata);
      const sha256 = `sha256:${await sha256Hex(await blob.arrayBuffer())}`;
      return {
        scope,
        scopeId,
        ...metadata.artifact,
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
            pages.reset();
            setUpload(false);
          }}
        >
          <option value="project">Evaluation workspace</option>
          <option value="user">My artifacts</option>
        </select>
      </EvalField>
      <EvalError error={inventory.error} />
      <EvalError error={select.error} />
      {select.isPending ? <p role="status">Pinning input…</p> : null}
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
              className="secondary-button"
              disabled={select.isPending}
              onClick={() => select.mutate(item)}
            >
              Use input
            </button>
          </li>
        ))}
      </ul>
      <CursorControls
        label="Input artifact pages"
        {...pages.controls(inventory.data?.page)}
      />
      {scope === "project" ? (
        <>
          <button
            type="button"
            className="secondary-button"
            onClick={() => setUpload(!upload)}
          >
            Upload an input
          </button>
          {upload ? (
            <ArtifactWriteForm
              scope={{ kind: "project", id: projectId }}
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
