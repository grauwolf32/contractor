import { useMutation } from "@tanstack/react-query";
import { useId, useRef, useState, type ReactNode } from "react";
import { Link } from "react-router";
import type { ArtifactMetadata } from "../../api/artifacts";
import { artifactScopeKeys } from "../../api/query-keys";
import {
  scopedArtifactAPI,
  type WritableArtifactScope,
} from "../../api/scoped-artifacts";
import { usePublicAPI } from "../../api/context";
import { artifactDetailPath } from "../../app/artifact-path";
import { Dialog, DialogHeader } from "../../app/dialog";
import { ErrorNotice } from "../../app/error-notice";
import { saveBlob } from "../../app/download";
import { formatBytes } from "../../app/format";
import { Icon } from "../../app/icon";
import { Button, EmptyState, IconButton, IdChip } from "../../ui";
import { MaterialKindMark } from "./icons";
import { formatLabel, materialKindOf } from "./kinds";
import { ArtifactPreviewPanel } from "./preview";
import { materialKey, matchingMaterials } from "./material-library-filter";
import "./material-library.css";
import "./material-chooser.css";

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

export function MaterialChooser({
  title,
  context,
  items,
  scope,
  projectRoot,
  initialSelection,
  previewInitially = false,
  onClose,
  onAttach,
  sourceControl,
  listActions,
  notice,
  empty,
  pending = false,
  listLabel,
  searchLabel,
  attachLabel,
}: {
  title: string;
  context: string;
  items: readonly ArtifactMetadata[];
  scope: WritableArtifactScope;
  projectRoot?: "/projects" | "/evals/workspaces";
  initialSelection?: string | undefined;
  previewInitially?: boolean;
  onClose: () => void;
  onAttach: (item: ArtifactMetadata) => void;
  sourceControl?: ReactNode;
  listActions: ReactNode;
  notice?: ReactNode;
  empty?: ReactNode;
  pending?: boolean;
  listLabel: string;
  searchLabel: string;
  attachLabel: string;
}) {
  const api = usePublicAPI();
  const reader = scopedArtifactAPI(api, scope);
  const heading = useId();
  const searchRef = useRef<HTMLInputElement>(null);
  const backRef = useRef<HTMLButtonElement>(null);
  const [search, setSearch] = useState("");
  const [selection, setSelection] = useState(initialSelection);
  const [preview, setPreview] = useState(previewInitially);
  const matches = matchingMaterials(items, "all", search);
  const selected = matches.find((item) => materialKey(item) === selection);
  const download = useMutation({
    mutationFn: (material: ArtifactMetadata) => reader.download(material),
    onSuccess: (result) => saveBlob(result.blob, result.filename),
  });
  const downloadShown =
    selected !== undefined &&
    download.variables !== undefined &&
    materialKey(download.variables) === materialKey(selected);
  return (
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
        title={title}
        close={{ label: "Close material chooser", onClose }}
      />
      {context && <p className="check-picker-context">{context}</p>}
      {notice}
      <div className="check-picker-grid" data-preview={preview}>
        <section className="check-picker-list" aria-label={listLabel}>
          {sourceControl}
          <label className="materials-search">
            <Icon name="search" size={17} />
            <span className="ui-visually-hidden">{searchLabel}</span>
            <input
              ref={searchRef}
              type="search"
              placeholder={searchLabel}
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
                disabled={pending}
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
            {matches.length > 0
              ? null
              : (empty ?? (
                  <EmptyState title="No matching materials">
                    Try another name or revision.
                  </EmptyState>
                ))}
          </div>
          {listActions}
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
                  ...artifactScopeKeys(scope).metadata(
                    selected.artifact.namespace,
                    selected.artifact.name,
                    selected.artifact.revision,
                  ),
                  "preview",
                ]}
                loadPreview={() => reader.preview(selected)}
                unavailableCopy="Download or open this material to inspect its saved file."
              />
              <div className="check-picker-details">
                <span>
                  {selected.artifact.namespace}/{selected.artifact.name}
                </span>
                <Link
                  target="_blank"
                  rel="noopener"
                  to={artifactDetailPath(scope, selected.artifact, projectRoot)}
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
            "Select a material"
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
            disabled={selected === undefined || pending}
            onClick={() => {
              if (selected !== undefined) {
                onAttach(selected);
              }
            }}
          >
            {attachLabel}
          </Button>
        </div>
      </footer>
    </Dialog>
  );
}
