import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { listArtifacts, type ArtifactMetadata } from "../../api/artifacts";
import { listProjectArtifacts } from "../../api/project-artifacts";
import { usePublicAPI } from "../../api/context";
import { scopedArtifactAPI } from "../../api/scoped-artifacts";
import type { EvalArtifact } from "../../api/evals";
import { CursorControls } from "../../app/cursor-controls";
import { useCursorStack } from "../../app/pagination";
import { MaterialChooser } from "../artifacts/material-chooser";
import { UploadDialog } from "../artifacts/upload-dialog";
import { uploadCancelNote } from "../artifacts/upload-operation";
import { useSession } from "../../auth/session";
import { Button, Field } from "../../ui";
import { ArtifactWriteForm } from "../artifacts/common";
import { EvalError } from "./common";
import { queryKeys } from "../../api/query-keys";
import { sha256Hex } from "../../app/digest";

/** Skill packages live on the Skills surface, not among Eval inputs. */
const EXCLUDED_SKILL_NAMESPACE = "skills";

export function EvalArtifactPicker({
  projectId,
  onSelect,
  onClose,
  role,
}: {
  projectId: string;
  onSelect: (ref: EvalArtifact) => void;
  onClose: () => void;
  role: string;
}) {
  const api = usePublicAPI();
  const client = useQueryClient();
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
  const owner =
    scope === "project"
      ? { kind: "project" as const, id: projectId }
      : { kind: "user" as const };
  const select = useMutation({
    mutationFn: async (metadata: ArtifactMetadata) => {
      const scopeId =
        scope === "project" ? projectId : session!.principal.userId;
      const { blob } = await scopedArtifactAPI(api, owner).download(metadata);
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
  });
  const items = inventory.data?.items ?? [];
  const pageOnly = cursor !== undefined || inventory.data?.page.hasMore;
  const context = inventory.data
    ? `${items.length} ${items.length === 1 ? "file" : "files"}${pageOnly ? " on this page" : ""}`
    : "";
  return (
    <>
      <MaterialChooser
        key={JSON.stringify([projectId, scope, cursor])}
        title={`Choose input · ${role}`}
        context={context}
        items={items}
        scope={owner}
        projectRoot="/evals/workspaces"
        sourceControl={
          <Field label="Input source">
            {(control) => (
              <select
                {...control}
                value={scope}
                disabled={select.isPending}
                onChange={(e) => {
                  setScope(e.target.value as typeof scope);
                  pages.reset();
                  select.reset();
                  setUpload(false);
                }}
              >
                <option value="project">Evaluation workspace</option>
                <option value="user">My files</option>
              </select>
            )}
          </Field>
        }
        listLabel="Input files"
        searchLabel={
          pageOnly ? "Search files on this page" : "Search input files"
        }
        attachLabel="Use input"
        pending={select.isPending}
        onClose={onClose}
        onAttach={(item) => select.mutate(item, { onSuccess: onSelect })}
        empty={
          inventory.isPending ? (
            <p role="status">Loading files…</p>
          ) : inventory.error ? (
            <EvalError
              error={inventory.error}
              reload={() => void inventory.refetch()}
            />
          ) : items.length === 0 ? (
            <p className="ui-caption">No files here yet.</p>
          ) : undefined
        }
        notice={
          <>
            {inventory.error && items.length > 0 && (
              <EvalError
                error={inventory.error}
                reload={() => void inventory.refetch()}
              />
            )}
            <EvalError error={select.error} />
            {select.isPending ? (
              <p className="check-picker-notice" role="status">
                Pinning input…
              </p>
            ) : null}
          </>
        }
        listActions={
          <>
            {!select.isPending && (
              <CursorControls
                label="Input file pages"
                {...pages.controls(inventory.data?.page)}
              />
            )}
            {scope === "project" ? (
              <Button
                size="sm"
                disabled={select.isPending}
                onClick={() => setUpload(true)}
              >
                Upload an input
              </Button>
            ) : null}
          </>
        }
      />
      {upload ? (
        <UploadDialog
          eyebrow="Evaluation workspace"
          title="Upload an input"
          closeLabel="Close upload dialog"
          note={uploadCancelNote("file")}
          onClose={() => setUpload(false)}
        >
          {(operation) => (
            <ArtifactWriteForm
              scope={{ kind: "project", id: projectId }}
              headingId={operation.headingId}
              startOperation={operation.start}
              onCancel={operation.close}
              onWritten={() => {
                setUpload(false);
                pages.reset();
                void client.invalidateQueries({
                  queryKey: queryKeys.evals.inputArtifacts(
                    "project",
                    projectId,
                    undefined,
                  ),
                });
              }}
            />
          )}
        </UploadDialog>
      ) : null}
    </>
  );
}
