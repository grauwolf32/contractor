import { useMutation } from "@tanstack/react-query";
import { useId, useRef, useState } from "react";
import { Link } from "react-router";

import {
  ENGAGEMENT_MEDIA_TYPE,
  type ArtifactMetadata,
  type ArtifactWriteResponse,
} from "../../../api/artifacts";
import { usePublicAPI } from "../../../api/context";
import { getProjectArtifactMetadata } from "../../../api/project-artifacts";
import { queryKeys } from "../../../api/query-keys";
import { previewProjectArtifact } from "../../../api/project-artifacts";
import { downloadProjectArtifact } from "../../../api/project-artifacts";
import { artifactDetailPath } from "../../../app/artifact-path";
import { Dialog, DialogHeader } from "../../../app/dialog";
import { ErrorNotice } from "../../../app/error-notice";
import { saveBlob } from "../../../app/download";
import { formatBytes } from "../../../app/format";
import { Icon } from "../../../app/icon";
import { Button, EmptyState, IconButton, IdChip } from "../../../ui";
import { ArtifactWriteForm } from "../../artifacts/common";
import { MaterialKindMark } from "../../artifacts/icons";
import {
  formatLabel,
  materialKindOf,
  MATERIAL_UPLOAD_KINDS,
} from "../../artifacts/kinds";
import { ArtifactPreviewPanel } from "../../artifacts/preview";
import { UploadDialog } from "../../artifacts/upload-dialog";
import { uploadCancelNote } from "../../artifacts/upload-operation";
import { candidatesFor, inSentence, materialKey } from "./check-types";
import type { InputSlot, StartCheck } from "./use-start-check";
import "./material-picker.css";

export function MaterialSummary({ material }: { material: ArtifactMetadata }) {
  const { artifact } = material;
  return (
    <span className="check-material-summary">
      <strong>{artifact.name}</strong>
      <span className="check-material-meta">
        <code
          title={`${artifact.namespace}/${artifact.name}@${artifact.revision}`}
        >
          {artifact.revision}
        </code>
        <span>{formatLabel(material.mediaType)}</span>
        <span>{formatBytes(material.size)}</span>
      </span>
      <span className="check-material-namespace">
        {artifact.namespace}
        {material.gitSource === undefined ? null : (
          <>
            {" "}
            · Git{" "}
            <code title={material.gitSource.resolvedCommit}>
              {material.gitSource.resolvedCommit.slice(0, 8)}
            </code>
          </>
        )}
      </span>
    </span>
  );
}

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
  const heading = useId();
  const searchRef = useRef<HTMLInputElement>(null);
  const backRef = useRef<HTMLButtonElement>(null);
  const [search, setSearch] = useState("");
  const [selection, setSelection] = useState(slot.selection);
  const [preview, setPreview] = useState(previewInitially);
  const [uploadOpen, setUploadOpen] = useState(false);
  const matches = slot.candidates.filter((item) =>
    `${item.artifact.namespace}/${item.artifact.name}@${item.artifact.revision} ${item.mediaType}`
      .toLowerCase()
      .includes(search.trim().toLowerCase()),
  );
  const selected = matches.find((item) => materialKey(item) === selection);
  const created = useMutation({
    mutationFn: (result: ArtifactWriteResponse) =>
      getProjectArtifactMetadata(api, {
        projectId: model.projectId,
        ...result.artifact,
      }),
  });
  const download = useMutation({
    mutationFn: (material: ArtifactMetadata) =>
      downloadProjectArtifact(api, model.projectId, material),
    onSuccess: (result) => saveBlob(result.blob, result.filename),
  });
  const downloadShown =
    selected !== undefined &&
    download.variables !== undefined &&
    materialKey(download.variables) === materialKey(selected);
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
      <Dialog
        className="ui-dialog check-material-picker"
        labelledBy={heading}
        onRequestClose={onClose}
        initialFocusRef={
          previewInitially && window.innerWidth <= 820 ? backRef : searchRef
        }
      >
        <DialogHeader
          id={heading}
          title={`Choose ${inSentence(slot.label)}`}
          close={{ label: "Close material chooser", onClose }}
        />
        <p className="check-picker-context">
          {model.project?.name ?? "Project"} · {slot.candidates.length}{" "}
          compatible materials{model.materials.complete ? "" : " loaded"}
        </p>
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
            The material was added, but its format does not match this input.
            Choose a compatible material.
          </p>
        ) : null}
        <div className="check-picker-grid" data-preview={preview}>
          <section
            className="check-picker-list"
            aria-label="Compatible materials"
          >
            <label className="materials-search">
              <Icon name="search" size={17} />
              <span className="ui-visually-hidden">
                Search compatible materials
              </span>
              <input
                ref={searchRef}
                type="search"
                placeholder="Search compatible materials"
                value={search}
                onChange={(event) => {
                  setSearch(event.target.value);
                }}
              />
            </label>
            <div className="check-picker-rows">
              {matches.map((item) => (
                <Button
                  key={materialKey(item)}
                  className="check-picker-row"
                  variant="ghost"
                  aria-label={`${item.artifact.namespace}/${item.artifact.name}@${item.artifact.revision}`}
                  aria-pressed={selected === item}
                  onClick={() => {
                    setSelection(materialKey(item));
                    setPreview(true);
                    if (window.innerWidth <= 820)
                      window.requestAnimationFrame(() =>
                        backRef.current?.focus(),
                      );
                  }}
                >
                  <MaterialKindMark kind={materialKindOf(item)} />
                  <MaterialSummary material={item} />
                  <Icon
                    name={selected === item ? "check" : "chevron-right"}
                    size={16}
                  />
                </Button>
              ))}
              {matches.length > 0 ? null : (
                <EmptyState title="No matching materials">
                  Try another name, or add a compatible material.
                </EmptyState>
              )}
            </div>
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
          </section>
          <section
            className="check-picker-preview"
            aria-label="Selected material preview"
          >
            <Button
              ref={backRef}
              className="check-picker-back"
              variant="ghost"
              size="sm"
              icon={<Icon name="arrow-left" />}
              onClick={() => {
                setPreview(false);
                window.setTimeout(() => searchRef.current?.focus(), 0);
              }}
            >
              Back to choices
            </Button>
            {selected === undefined ? (
              <EmptyState title="Material preview">
                Choose a material to inspect its saved version.
              </EmptyState>
            ) : (
              <>
                <div className="check-picker-preview-head">
                  <div className="check-picker-preview-toolbar">
                    <span className="materials-kicker">Saved version</span>
                    <div className="ui-toolbar">
                      <span className="check-picker-copy">
                        <IdChip
                          value={`${selected.artifact.namespace}/${selected.artifact.name}@${selected.artifact.revision}`}
                          display=""
                          label="material identifier"
                        />
                      </span>
                      <IconButton
                        size="sm"
                        variant="ghost"
                        label="Download material"
                        icon={<Icon name="download" />}
                        pending={downloadShown && download.isPending}
                        onClick={() => download.mutate(selected)}
                      />
                    </div>
                  </div>
                  <h3>{selected.artifact.name}</h3>
                  <div className="check-material-meta">
                    <span className="materials-badge">
                      {selected.current ? "Current" : "Historical"}
                    </span>
                    <code>{selected.artifact.revision}</code>
                    <span>{formatBytes(selected.size)}</span>
                  </div>
                  {downloadShown && download.error !== null ? (
                    <ErrorNotice
                      error={download.error}
                      context="The download did not finish."
                    />
                  ) : null}
                </div>
                <ArtifactPreviewPanel
                  key={materialKey(selected)}
                  archiveScope={scope}
                  metadata={selected}
                  loadOnMountKey={[
                    ...queryKeys.projects.artifacts.metadata(
                      model.projectId,
                      selected.artifact.namespace,
                      selected.artifact.name,
                      selected.artifact.revision,
                    ),
                    "preview",
                  ]}
                  loadPreview={() =>
                    previewProjectArtifact(api, model.projectId, selected)
                  }
                  unavailableCopy="Download or open this material to inspect its saved file."
                />
                <div className="check-picker-details">
                  <span>
                    {selected.artifact.namespace}/{selected.artifact.name}
                  </span>
                  <Link
                    target="_blank"
                    rel="noopener"
                    to={artifactDetailPath(scope, selected.artifact)}
                  >
                    Open full page ↗
                  </Link>
                </div>
              </>
            )}
          </section>
        </div>
        <footer className="check-picker-footer">
          <div className="check-picker-selection">
            {selected === undefined ? (
              "Select a material to attach"
            ) : (
              <>
                <strong>{selected.artifact.name}</strong>
                <code>{selected.artifact.revision}</code>
              </>
            )}
          </div>
          <div className="ui-toolbar">
            <Button variant="ghost" onClick={onClose}>
              Cancel
            </Button>
            <Button
              variant="primary"
              icon={<Icon name="check" />}
              disabled={selected === undefined || created.isPending}
              onClick={() => {
                if (selected !== undefined) {
                  model.chooseMaterial(slot.name, materialKey(selected));
                  onClose();
                }
              }}
            >
              Attach material
            </Button>
          </div>
        </footer>
      </Dialog>
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
