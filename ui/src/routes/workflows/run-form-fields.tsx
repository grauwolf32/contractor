import { useState, type ReactNode } from "react";
import { Link } from "react-router";

import type { ArtifactMetadata } from "../../api/artifacts";
import type { WorkflowResource } from "../../api/workflows";
import { ErrorNotice } from "../../app/error-notice";
import { formatBytes, formatTimestamp } from "../../app/format";
import { Icon } from "../../app/icon";
import { LoadMoreButton } from "../../app/load-more";
import type { RunDraftState } from "../../run-drafts/memory";
import {
  artifactAccepts,
  artifactOptionKey,
} from "../../run-drafts/validation";
import { Button, Facts, Field, IconButton, StatusChip } from "../../ui";
import {
  GitImportDialog,
  GitSourceDetails,
} from "../artifacts/git-import-dialog";
import { artifactDetailPath } from "../artifacts/paths";
import type { RunDraftErrors } from "./run-form-state";
import type { RunFormInventories } from "./run-form-inventories";
import { RunInputUploadDialog } from "./run-input-upload-dialog";
import "./run-drafts.css";

type ParameterSlot = WorkflowResource["parameters"][string];

/** "source code" for `source_code`, with the exact name after it. */
function SlotLabel({ name, required }: { name: string; required: boolean }) {
  return (
    <span>
      {name.replaceAll("_", " ")} {required ? <strong>required</strong> : null}
      {name.includes("_") ? <small> ({name})</small> : null}
    </span>
  );
}

const PARAMETER_HINTS: Readonly<Record<string, string>> = {
  authorization_scope: "Allowed systems, actions, test accounts and exclusions",
  objective: "What should this Run produce or verify?",
  target: "For example, https://app.example.test",
};

/**
 * The string parameters `slots` in name order. An optional one is left out
 * of the request until "Include optional …" is checked; the objective and
 * authorization scope take several lines.
 */
export function ParameterFields({
  slots,
  title,
  parameters,
  errors,
  onChange,
}: {
  slots: readonly [string, ParameterSlot][];
  title: string;
  parameters: RunDraftState["parameters"];
  errors: RunDraftErrors;
  /** A value, or undefined to leave an optional parameter out. */
  onChange: (name: string, value: string | undefined) => void;
}) {
  return (
    <fieldset className="run-draft-section">
      <legend>{title}</legend>
      {slots.length === 0 ? (
        <p className="compact-empty">This Workflow declares no parameters.</p>
      ) : (
        <div className="run-field-grid">
          {[...slots]
            .sort(([left], [right]) => left.localeCompare(right))
            .map(([name, slot]) => {
              const included = parameters[name] !== undefined;
              const Control = ["objective", "authorization_scope"].includes(
                name,
              )
                ? "textarea"
                : "input";
              return (
                <div className="run-field" key={name}>
                  {slot.required ? null : (
                    <label className="run-field-check">
                      <input
                        type="checkbox"
                        checked={included}
                        onChange={(event) =>
                          onChange(name, event.target.checked ? "" : undefined)
                        }
                      />
                      Include optional <code>{name}</code>
                    </label>
                  )}
                  <Field
                    id={`parameter-${name}`}
                    label={<SlotLabel name={name} required={slot.required} />}
                    error={errors[`parameter:${name}`]}
                  >
                    {(control) => (
                      <Control
                        {...control}
                        name={`parameter-${name}`}
                        {...(Control === "textarea"
                          ? { rows: 3 }
                          : { type: "text" })}
                        placeholder={PARAMETER_HINTS[name]}
                        disabled={!slot.required && !included}
                        value={parameters[name] ?? ""}
                        onChange={(event) => onChange(name, event.target.value)}
                      />
                    )}
                  </Field>
                </div>
              );
            })}
        </div>
      )}
    </fieldset>
  );
}

function artifactLabel(metadata: ArtifactMetadata): string {
  return `${artifactOptionKey(metadata.artifact)} · ${metadata.mediaType} · ${formatBytes(metadata.size)}`;
}

/** The selected revision of one input, to confirm before the Run starts. */
function InputReview({
  name,
  selected,
  metadata,
  reviewed,
  suggested,
  projectId,
  inputNoun,
  onConfirm,
}: {
  name: string;
  selected: string;
  metadata: ArtifactMetadata;
  reviewed: boolean;
  suggested: boolean;
  projectId: string | undefined;
  inputNoun: "file" | "material";
  onConfirm: () => void;
}) {
  const needsReview = !reviewed;
  return (
    <section
      className={`run-input-review${needsReview ? " needs-review" : " is-reviewed"}`}
      aria-label={`Input review for ${name}`}
    >
      <div className="run-input-review-heading">
        <span>
          <small>
            {needsReview && suggested
              ? "Format-compatible suggestion"
              : reviewed
                ? "Reviewed selection"
                : "Selection"}
          </small>
          <code>{selected}</code>
        </span>
        <StatusChip tone={reviewed ? "success" : "review"} size="sm">
          {reviewed ? "Confirmed" : "Review needed"}
        </StatusChip>
      </div>
      <Facts
        size="sm"
        items={[
          {
            term: "Scope",
            value:
              projectId === undefined
                ? "Your library"
                : `Project · ${projectId}`,
          },
          { term: "Stored type", value: metadata.mediaType },
          { term: "Size", value: formatBytes(metadata.size) },
          {
            term: "Provenance",
            value:
              metadata.gitSource === undefined
                ? `Stored · ${formatTimestamp(metadata.createdAt)}`
                : `Git import · ${formatTimestamp(metadata.gitSource.importedAt)}`,
          },
        ]}
      />
      <p>
        {needsReview && suggested
          ? `Suggested only because ${metadata.mediaType} matches this slot. Confirm that this revision is the intended ${name} input.`
          : reviewed
            ? `Confirmed for ${name}. This records your selection, not a validation of its contents.`
            : `Review this revision before using it as ${name}.`}
      </p>
      <div className="run-input-review-actions">
        {needsReview ? (
          <Button size="sm" onClick={onConfirm}>
            Confirm input for {name}
          </Button>
        ) : null}
        <Link
          to={artifactDetailPath(
            projectId === undefined
              ? { kind: "user" }
              : { kind: "project", id: projectId },
            metadata.artifact,
          )}
        >
          Preview {inputNoun} details
        </Link>
      </div>
    </section>
  );
}

/**
 * One exact revision per declared input, from the library or the Project:
 * a format-compatible choice, its review, and a local upload or Git import
 * that selects what it stored. A dialog answer for a draft that is no
 * longer current (`isCurrent`) is dropped.
 */
export function InputFields({
  workflow,
  presentation,
  projectId,
  draft,
  errors,
  inventories,
  isCurrent,
  onSelect,
  onConfirm,
}: {
  workflow: WorkflowResource;
  presentation: "page" | "drawer";
  projectId: string | undefined;
  draft: RunDraftState;
  errors: RunDraftErrors;
  inventories: Pick<
    RunFormInventories,
    "artifactInventory" | "artifacts" | "artifactMap"
  >;
  isCurrent: () => boolean;
  onSelect: (
    slot: string,
    selected: string,
    metadata?: ArtifactMetadata,
  ) => void;
  onConfirm: (slot: string) => void;
}) {
  const [gitSlot, setGitSlot] = useState<string | null>(null);
  const [uploadSlot, setUploadSlot] = useState<string | null>(null);
  const { artifactInventory, artifacts, artifactMap } = inventories;
  const { artifactSelections, artifactSuggestions, artifactReviews } = draft;
  // Artifacts are files in the Library and materials in a Project (§3).
  const inputNoun = projectId === undefined ? "file" : "material";
  const inputNouns = projectId === undefined ? "files" : "materials";

  function selectStored(slot: string, metadata: ArtifactMetadata): void {
    onSelect(slot, artifactOptionKey(metadata.artifact), metadata);
  }

  let slots: ReactNode;
  if (Object.entries(workflow.inputs).length === 0) {
    slots = (
      <p className="run-draft-empty">
        This Workflow declares no input {inputNouns}.
      </p>
    );
  } else {
    slots = (
      <div className="run-field-grid">
        {Object.entries(workflow.inputs)
          .sort(
            ([left, leftSlot], [right, rightSlot]) =>
              (presentation === "drawer"
                ? Number(rightSlot.required) - Number(leftSlot.required)
                : 0) || left.localeCompare(right),
          )
          .map(([name, slot]) => {
            const compatible = artifacts.filter((metadata) =>
              artifactAccepts(slot.mediaTypes, metadata),
            );
            const selected = artifactSelections[name] ?? "";
            const selectedMetadata = artifactMap.get(selected);
            return (
              <div className="run-field" key={name}>
                <Field
                  id={`artifact-${name}`}
                  label={<SlotLabel name={name} required={slot.required} />}
                  hint={
                    <>
                      Slot <code>{name}</code> accepts{" "}
                      {slot.mediaTypes.join(", ")}. A format match does not
                      check what the {inputNoun} contains.
                    </>
                  }
                  error={
                    errors[`artifact:${name}`] ??
                    errors[`artifactReview:${name}`]
                  }
                >
                  {(control) => (
                    <select
                      {...control}
                      name={`artifact-${name}`}
                      value={selected}
                      disabled={artifactInventory.isPending}
                      onChange={(event) => {
                        const exactRef = event.target.value;
                        onSelect(name, exactRef, artifactMap.get(exactRef));
                      }}
                    >
                      <option value="">
                        {slot.required
                          ? "Select a revision"
                          : `No ${inputNoun} supplied`}
                      </option>
                      {compatible.map((metadata) => (
                        <option
                          key={artifactOptionKey(metadata.artifact)}
                          value={artifactOptionKey(metadata.artifact)}
                        >
                          {artifactLabel(metadata)}
                        </option>
                      ))}
                    </select>
                  )}
                </Field>
                {selectedMetadata === undefined ? null : (
                  <InputReview
                    name={name}
                    selected={selected}
                    metadata={selectedMetadata}
                    reviewed={artifactReviews[name] === selected}
                    suggested={artifactSuggestions[name] === selected}
                    projectId={projectId}
                    inputNoun={inputNoun}
                    onConfirm={() => onConfirm(name)}
                  />
                )}
                <div className="run-field-actions">
                  <Button size="sm" onClick={() => setUploadSlot(name)}>
                    Upload local file for {name}
                  </Button>
                  {slot.mediaTypes.some((type) =>
                    ["application/zip", "*/*"].includes(type),
                  ) ? (
                    <IconButton
                      size="sm"
                      label={`Import Git for ${name}`}
                      icon={<Icon name="git" size={16} />}
                      onClick={() => setGitSlot(name)}
                    />
                  ) : null}
                </div>
                <GitSourceDetails source={selectedMetadata?.gitSource} />
                {!artifactInventory.isPending && compatible.length === 0 ? (
                  <p className="field-guidance">
                    No matching {inputNouns} found in the loaded list.{" "}
                    <Link
                      to={
                        projectId === undefined
                          ? "/artifacts"
                          : `/projects/${encodeURIComponent(projectId)}/artifacts`
                      }
                    >
                      Upload one
                    </Link>
                    .
                  </p>
                ) : null}
              </div>
            );
          })}
      </div>
    );
  }

  return (
    <fieldset className="run-draft-section">
      <legend>
        {projectId === undefined ? "Input files" : "Input materials"}
      </legend>
      {gitSlot === null ? null : (
        <GitImportDialog
          {...(projectId === undefined ? {} : { projectId })}
          suggestedName={gitSlot}
          onClose={() => setGitSlot(null)}
          onImported={(result) => {
            if (!isCurrent()) return;
            selectStored(gitSlot, {
              artifact: result.artifact,
              mediaType: result.mediaType,
              size: result.size,
              gitSource: result.gitSource,
              current: true,
              frozen: false,
              createdAt: result.gitSource.importedAt,
            });
            setGitSlot(null);
          }}
        />
      )}
      {uploadSlot === null ? null : (
        <RunInputUploadDialog
          {...(projectId === undefined ? {} : { projectId })}
          slotName={uploadSlot}
          mediaTypes={workflow.inputs[uploadSlot]?.mediaTypes ?? []}
          onClose={() => setUploadSlot(null)}
          onUploaded={(result) => {
            if (!isCurrent()) return;
            selectStored(uploadSlot, {
              artifact: result.artifact,
              mediaType: result.mediaType,
              size: result.size,
              current: true,
              frozen: false,
              createdAt: new Date().toISOString(),
            });
            setUploadSlot(null);
          }}
        />
      )}
      {artifactInventory.error === null ? null : (
        <ErrorNotice error={artifactInventory.error} />
      )}
      {slots}
      <LoadMoreButton
        query={artifactInventory}
        label={`Load more ${inputNouns}`}
        pendingLabel={`Loading ${inputNouns}…`}
      />
    </fieldset>
  );
}
