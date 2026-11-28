import { materialKey } from "../../artifacts/material-library-filter";
import { useMutation } from "@tanstack/react-query";
import { useState } from "react";

import {
  ENGAGEMENT_MEDIA_TYPE,
  type ArtifactWriteResponse,
} from "../../../api/artifacts";
import { usePublicAPI } from "../../../api/context";
import { getProjectArtifactMetadata } from "../../../api/project-artifacts";
import { ErrorNotice } from "../../../app/error-notice";
import { Icon } from "../../../app/icon";
import { Button } from "../../../ui";
import { ArtifactWriteForm } from "../../artifacts/common";
import { materialKindOf, MATERIAL_UPLOAD_KINDS } from "../../artifacts/kinds";
import { UploadDialog } from "../../artifacts/upload-dialog";
import { uploadCancelNote } from "../../artifacts/upload-operation";
import { candidatesFor, inSentence } from "./check-types";
import type { InputSlot, StartCheck } from "./use-start-check";
import { MaterialChooser } from "../../artifacts/material-chooser";
import "./material-picker.css";

/** Previewing is tentative. Only Attach commits the exact input to the composer. */
export function MaterialPicker({
  model,
  slot,
  previewInitially,
  onClose,
}: {
  model: StartCheck;
  slot: InputSlot;
  previewInitially: boolean;
  onClose: () => void;
}) {
  const api = usePublicAPI();
  const [uploadOpen, setUploadOpen] = useState(false);
  const created = useMutation({
    mutationFn: (result: ArtifactWriteResponse) =>
      getProjectArtifactMetadata(api, {
        projectId: model.projectId,
        ...result.artifact,
      }),
  });
  const scope = { kind: "project" as const, id: model.projectId };
  const suggestedKind = slot.input.media_types.includes(ENGAGEMENT_MEDIA_TYPE)
    ? "engagement"
    : slot.name === "source"
      ? "source"
      : slot.name === "openapi"
        ? "api"
        : slot.selected === undefined
          ? "other"
          : materialKindOf(slot.selected);
  const suggested = MATERIAL_UPLOAD_KINDS.find(
    (kind) => kind.id === suggestedKind,
  )!;
  function readAndAttach(result: ArtifactWriteResponse) {
    created.mutate(result, {
      onSuccess: (material) => {
        // Mutation observers stop on unmount: a late read cannot change another check.
        if (candidatesFor(slot.input, [material]).length === 0) return;
        model.registerAuthoredMaterial(slot.name, material);
        onClose();
      },
    });
  }
  return (
    <>
      <MaterialChooser
        title={`Choose ${inSentence(slot.label)}`}
        context={`${model.project?.name ?? "Project"} · ${slot.candidates.length} compatible materials${model.materials.complete ? "" : " loaded"}`}
        items={slot.candidates}
        scope={scope}
        initialSelection={slot.selection}
        previewInitially={previewInitially}
        onClose={onClose}
        onAttach={(item) => {
          model.chooseMaterial(slot.name, materialKey(item));
          onClose();
        }}
        listLabel="Compatible materials"
        searchLabel="Search compatible materials"
        attachLabel="Attach material"
        pending={created.isPending}
        notice={
          <>
            {" "}
            {created.isPending ? (
              <p className="ui-caption check-picker-notice" role="status">
                Material added. Reading its saved version…
              </p>
            ) : null}
            {created.error === null ? null : (
              <div className="check-picker-notice">
                <ErrorNotice
                  error={created.error}
                  context="The material was added, but its exact version could not be read."
                  onRetry={() => {
                    if (created.variables !== undefined)
                      readAndAttach(created.variables);
                  }}
                  retryPending={created.isPending}
                />
              </div>
            )}
            {created.data !== undefined &&
            candidatesFor(slot.input, [created.data]).length === 0 ? (
              <p className="notice notice-warning check-picker-notice">
                The material was added, but its format does not match this
                input. Choose a compatible material.
              </p>
            ) : null}
          </>
        }
        listActions={
          <>
            {" "}
            <p className="ui-caption">Accepted formats: {slot.formats}.</p>
            {model.materials.error === null ? null : (
              <ErrorNotice
                error={model.materials.error}
                context="More materials could not be read."
                onRetry={model.materials.retry}
                retryPending={model.materials.retrying}
              />
            )}
            {model.materials.hasMore ? (
              <Button
                size="sm"
                pending={model.materials.loadingMore}
                onClick={model.materials.loadMore}
              >
                Load more materials
              </Button>
            ) : null}
            <Button
              variant="ghost"
              size="sm"
              icon={<Icon name="plus" />}
              onClick={() => setUploadOpen(true)}
            >
              Add material
            </Button>
          </>
        }
      />
      {uploadOpen ? (
        <UploadDialog
          eyebrow="New check material"
          title={`Add ${inSentence(slot.label)}`}
          closeLabel="Close upload dialog"
          note={uploadCancelNote("material")}
          onClose={() => setUploadOpen(false)}
        >
          {(upload) => (
            <ArtifactWriteForm
              scope={scope}
              suggested={{
                ...suggested,
                mediaType: slot.input.media_types.includes(suggested.mediaType)
                  ? suggested.mediaType
                  : (slot.input.media_types.find((type) => type !== "*/*") ??
                    suggested.mediaType),
              }}
              acceptedMediaTypes={slot.input.media_types}
              headingId={upload.headingId}
              startOperation={upload.start}
              submitLabel="Add and attach"
              onCancel={upload.close}
              onWritten={(result) => {
                setUploadOpen(false);
                readAndAttach(result);
              }}
            />
          )}
        </UploadDialog>
      ) : null}
    </>
  );
}
